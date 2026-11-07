"""Futures-Simulator für aktive Bots (Long und Short).

Getrennt vom Team-Simulator, damit dessen erprobter Code unverändert bleibt.

Ablauf pro Bar i:
  1. Ausstehender Ausstieg (Signal/Zeitlimit vom Close i-1) zum Open.
  2. Ausstehende Einstiegs-Order prüfen:
       market -> Fill zum Open (Taker-Gebühr + Slippage)
       limit  -> Fill, wenn der Kurs den Limitpreis erreicht (Maker-Gebühr,
                 keine Slippage; öffnet die Bar schon besser, zum Open)
       stop   -> Fill, wenn der Kurs den Stoppreis erreicht (Taker + Slippage)
     Nach einem Fill wird in derselben Bar nur noch der Stop-Loss geprüft
     (vorsichtig: Ziel in der Einstiegs-Bar zählt nie).
  3. Offene Position: Stop-Loss vor Take-Profit. Stop = Taker + Slippage,
     Take-Profit = Limit (Maker). Kurslücke über den Stop: Fill zum Open.
  4. Haltekosten (Funding-Näherung), Bewertung, Notbremse.
  5. Strategie entscheidet zum Close.

Positionsgröße: ``risk_per_trade`` vom Konto bis zum Stop (inkl. Kosten),
höchstens ``max_leverage`` x Konto.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FxCosts:
    maker: float = 0.0002
    taker: float = 0.0005
    slippage: float = 0.0002
    funding_8h: float = 0.0001   # Haltekosten pro 8 h (vorsichtig für beide Seiten)


@dataclass
class FxRisk:
    risk_per_trade: float = 0.01
    max_leverage: float = 3.0
    min_stop: float = 0.0015      # Stop mind. 0,15 % entfernt, sonst nur Gebühren-Lotterie
    halt_drawdown: Optional[float] = None


@dataclass
class FxState:
    cash: float
    peak: float
    position: Optional[dict] = None
    order: Optional[dict] = None
    pending_exit: Optional[str] = None
    pending_add: bool = False
    halted: bool = False
    last_ts: Optional[str] = None
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FxState":
        return FxState(**{k: v for k, v in d.items() if k in FxState.__dataclass_fields__})


def _finite(*v) -> bool:
    return all(x is not None and math.isfinite(x) for x in v)


class FxSimulator:
    def __init__(self, strategy, costs: Optional[FxCosts] = None, risk: Optional[FxRisk] = None,
                 start_cash: float = 1000.0, state: Optional[FxState] = None, entry_block=None):
        self.s = strategy
        # News-Filter: liefert True, wenn zu diesem Zeitpunkt kein Einstieg erlaubt ist
        self.entry_block = entry_block
        self.c = costs or FxCosts()
        self.r = risk or FxRisk()
        self.st = state or FxState(cash=start_cash, peak=start_cash)

    # --------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> FxState:
        st = self.st
        idx = df.index
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        # Median statt erster Abstand: Lücken (Wochenende) verfälschen ihn nicht
        bar_hours = (pd.Series(idx).diff().median().total_seconds() / 3600) if len(idx) > 1 else 0.25
        start = 0
        if st.last_ts is not None:
            start = int(idx.searchsorted(pd.Timestamp(st.last_ts), side="right"))
        arrays = {k: df[k].to_numpy() for k in df.columns}
        for i in range(start, len(idx)):
            if not (_finite(o[i], h[i], l[i], c[i]) and l[i] <= min(o[i], c[i]) and h[i] >= max(o[i], c[i])
                    and l[i] > 0):
                raise ValueError(f"Ungültige Kerze {idx[i]}")
            self._step(i, idx[i], o[i], h[i], l[i], c[i], bar_hours, arrays)
        return st

    # --------------------------------------------------------------------
    def _step(self, i, ts, o, h, l, c, bar_hours, a):
        st = self.st
        # 1. Ausstieg zum Open
        if st.pending_exit and st.position:
            side = st.position["side"]
            self._close(ts, o * (1 - side * self.c.slippage), self.c.taker, st.pending_exit)
        st.pending_exit = None
        # 1b. Nachkauf im Gewinn (Pyramide) zum Open
        if st.pending_add and st.position:
            self._add(ts, o)
        st.pending_add = False

        # 2. Einstiegs-Order
        opened_now = False
        if st.order and self.entry_block is not None and self.entry_block(ts):
            st.order = None  # Einstieg verworfen; die Strategie prüft zum Close neu
        if st.order and not st.position and not st.halted:
            fill, fee = self._try_fill(st.order, o, h, l)
            if fill is not None:
                order, st.order = st.order, None
                opened_now = self._open(ts, fill, fee, order)
            else:
                st.order["expire"] -= 1
                if st.order["expire"] <= 0:
                    st.order = None

        # 3. Stop-Loss / Take-Profit
        pos = st.position
        if pos:
            # Bester/schlechtester Kurs während des Trades (für die Auswertung MFE/MAE)
            pos["hi"] = max(pos.get("hi", h), h)
            pos["lo"] = min(pos.get("lo", l), l)
            side, sl, tp = pos["side"], pos["sl"], pos["tp"]
            hit_sl = (l <= sl) if side == 1 else (h >= sl)
            if hit_sl:
                gap = (o <= sl) if side == 1 else (o >= sl)
                price = (o if gap and not opened_now else sl) * (1 - side * self.c.slippage)
                self._close(ts, price, self.c.taker, "stop")
            else:
                tp1 = pos.get("tp1")
                if tp1 is not None and not opened_now and ((h >= tp1) if side == 1 else (l <= tp1)):
                    gap = (o >= tp1) if side == 1 else (o <= tp1)
                    self._partial(ts, o if gap else tp1, pos["tp1_frac"])
                if tp is not None and not opened_now and st.position:
                    hit_tp = (h >= tp) if side == 1 else (l <= tp)
                    if hit_tp:
                        gap = (o >= tp) if side == 1 else (o <= tp)
                        self._close(ts, o if gap else tp, self.c.maker, "take_profit")

        # 4. Haltekosten, Bewertung, Notbremse
        pos = st.position
        if pos:
            cost = pos["qty"] * c * self.c.funding_8h * bar_hours / 8
            st.cash -= cost
            pos["funding"] += cost
            pos["bars_held"] += 1
        equity = self._equity(c)
        st.peak = max(st.peak, equity)
        st.equity.append((ts.isoformat(), equity))
        st.last_ts = ts.isoformat()
        if self.r.halt_drawdown and not st.halted and equity <= st.peak * (1 - self.r.halt_drawdown):
            st.halted, st.order = True, None
            if st.position:
                st.pending_exit = "notbremse"
        if st.halted:
            return

        # 5. Strategie
        if pos and pos.get("max_bars") and pos["bars_held"] >= pos["max_bars"]:
            st.pending_exit = "zeitstop"
            return
        # Strategien mit ``wants_history`` sehen zusätzlich die bisherigen Trades
        acts = (self.s.on_bar(i, a, st.position, st.trades) if getattr(self.s, "wants_history", False)
                else self.s.on_bar(i, a, st.position))
        for act in acts or []:
            kind = act[0]
            if kind == "exit" and st.position:
                st.pending_exit = act[1]
            elif kind == "order" and not st.position:
                order = dict(act[1])
                order.setdefault("expire", 1)
                st.order = order
            elif kind == "add" and st.position:
                st.pending_add = True
            elif kind == "cancel":
                st.order = None
            elif kind == "stop" and st.position:
                # Stop nur in Gewinnrichtung nachziehen, nie lockern
                p, new = st.position, float(act[1])
                if (p["side"] == 1 and new > p["sl"]) or (p["side"] == -1 and new < p["sl"]):
                    p["sl"] = new

    # --------------------------------------------------------------------
    def _try_fill(self, od, o, h, l):
        side, typ, p = od["side"], od["type"], od.get("price")
        if typ == "market":
            return o * (1 + side * self.c.slippage), self.c.taker
        if typ == "limit":
            if side == 1 and l <= p:
                return min(o, p), self.c.maker
            if side == -1 and h >= p:
                return max(o, p), self.c.maker
        if typ == "stop":
            if side == 1 and h >= p:
                return max(o, p) * (1 + self.c.slippage), self.c.taker
            if side == -1 and l <= p:
                return min(o, p) * (1 - self.c.slippage), self.c.taker
        return None, None

    def _open(self, ts, fill, fee, od) -> bool:
        st = self.st
        side, sl = od["side"], od["sl"]
        if not _finite(fill, sl) or (side == 1 and sl >= fill) or (side == -1 and sl <= fill):
            return False  # Stop liegt schon auf der falschen Seite: kein Einstieg
        if abs(fill - sl) / fill < self.r.min_stop:
            return False
        equity = st.cash
        dist = abs(fill - sl) / fill + fee + self.c.taker + self.c.slippage
        notional = min(equity * self.r.risk_per_trade / dist, equity * self.r.max_leverage)
        if notional <= 0:
            return False
        qty = notional / fill
        entry_fee = notional * fee
        st.cash -= entry_fee
        tp = od.get("tp")
        if tp is None and od.get("rr"):
            tp = fill + side * od["rr"] * abs(fill - sl)
        st.position = {"side": side, "entry_time": ts.isoformat(), "entry_price": fill, "qty": qty,
                       "sl": sl, "tp": tp, "initial_sl": sl, "fees": entry_fee, "funding": 0.0,
                       "bars_held": 0, "max_bars": od.get("max_bars"), "tag": od.get("tag", ""),
                       "context": od.get("context", {}), "first_entry": fill,
                       "risk_amt": qty * abs(fill - sl)}
        if od.get("tp1_r"):
            # Teilgewinn: bei +tp1_r x Risiko wird tp1_frac der Position geschlossen
            st.position["tp1"] = fill + side * od["tp1_r"] * abs(fill - sl)
            st.position["tp1_frac"] = od.get("tp1_frac", 0.5)
        return True

    def _add(self, ts, o) -> None:
        """Nachkauf zur bestehenden Position (nur in Gewinnrichtung gedacht): gleicher Stop,
        Größe so, dass der Zukauf bis zum Stop ``risk_per_trade`` des Kontos riskiert."""
        st, p = self.st, self.st.position
        side = p["side"]
        fill = o * (1 + side * self.c.slippage)
        if (side == 1 and p["sl"] >= fill) or (side == -1 and p["sl"] <= fill):
            return
        equity = self._equity(o)
        dist = abs(fill - p["sl"]) / fill + 2 * self.c.taker + self.c.slippage
        if abs(fill - p["sl"]) / fill < self.r.min_stop:
            return
        room = equity * self.r.max_leverage - p["qty"] * fill
        notional = min(equity * self.r.risk_per_trade / dist, room)
        if notional <= 0:
            return
        q = notional / fill
        fee = notional * self.c.taker
        st.cash -= fee
        p["entry_price"] = (p["entry_price"] * p["qty"] + fill * q) / (p["qty"] + q)
        p["qty"] += q
        p["fees"] += fee
        p["adds"] = p.get("adds", 0) + 1
        p["last_add"] = fill

    def _close(self, ts, price, fee, reason):
        st = self.st
        p = st.position
        gross = p["qty"] * (price - p["entry_price"]) * p["side"]
        exit_fee = p["qty"] * price * fee
        st.cash += gross - exit_fee
        pnl = gross - exit_fee - p["fees"] - p["funding"]
        risk = p.get("risk_amt") or p["qty"] * abs(p["entry_price"] - p["initial_sl"])
        st.trades.append({
            "side": "long" if p["side"] == 1 else "short", "entry_time": p["entry_time"],
            "exit_time": ts.isoformat(), "entry_price": p["entry_price"], "exit_price": price,
            "qty": p["qty"], "pnl": pnl, "adds": p.get("adds", 0), "return_pct": pnl / (p["qty"] * p["entry_price"]) * 100,
            "r_multiple": pnl / risk if risk > 0 else None, "bars_held": p["bars_held"],
            "reason": reason, "initial_stop": p["initial_sl"], "tp": p["tp"],
            "fees": p["fees"] + exit_fee, "tag": p["tag"], "context": p["context"],
            # wie weit der Trade zwischendurch im Plus (mfe) bzw. Minus (mae) war, in R
            "mfe_r": self._excursion(p, best=True), "mae_r": self._excursion(p, best=False),
        })
        st.position = None

    def _partial(self, ts, price: float, frac: float) -> None:
        """Schließt einen Teil der Position (Limit, Maker) und zieht den Stop auf den Einstieg."""
        st, p = self.st, self.st.position
        q = p["qty"] * frac
        gross = q * (price - p["entry_price"]) * p["side"]
        fee = q * price * self.c.maker
        fees_part, fund_part = p["fees"] * frac, p["funding"] * frac
        st.cash += gross - fee
        pnl = gross - fee - fees_part - fund_part
        risk = q * abs(p["entry_price"] - p["initial_sl"])
        st.trades.append({
            "side": "long" if p["side"] == 1 else "short", "entry_time": p["entry_time"],
            "exit_time": ts.isoformat(), "entry_price": p["entry_price"], "exit_price": price,
            "qty": q, "pnl": pnl, "return_pct": pnl / (q * p["entry_price"]) * 100,
            "r_multiple": pnl / risk if risk > 0 else None, "bars_held": p["bars_held"],
            "reason": "teilgewinn", "initial_stop": p["initial_sl"], "tp": p["tp1"],
            "fees": fees_part + fee, "tag": p["tag"], "context": p["context"],
            "mfe_r": self._excursion(p, best=True), "mae_r": self._excursion(p, best=False),
        })
        p["qty"] -= q
        if p.get("risk_amt"):
            p["risk_amt"] *= (1 - frac)
        p["fees"] -= fees_part
        p["funding"] -= fund_part
        p["tp1"] = None
        p["sl"] = max(p["sl"], p["entry_price"]) if p["side"] == 1 else min(p["sl"], p["entry_price"])

    @staticmethod
    def _excursion(p: dict, best: bool):
        r = abs(p["entry_price"] - p["initial_sl"])
        if r <= 0 or "hi" not in p:
            return None
        if p["side"] == 1:
            return ((p["hi"] if best else p["lo"]) - p["entry_price"]) / r
        return (p["entry_price"] - (p["lo"] if best else p["hi"])) / r

    def _equity(self, price) -> float:
        p = self.st.position
        if not p:
            return self.st.cash
        upnl = p["qty"] * (price - p["entry_price"]) * p["side"]
        return self.st.cash + upnl - p["qty"] * price * self.c.taker  # inkl. Ausstiegsgebühr
