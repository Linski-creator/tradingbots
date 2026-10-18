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
    halted: bool = False
    last_ts: Optional[str] = None
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FxState":
        return FxState(**d)


def _finite(*v) -> bool:
    return all(x is not None and math.isfinite(x) for x in v)


class FxSimulator:
    def __init__(self, strategy, costs: Optional[FxCosts] = None, risk: Optional[FxRisk] = None,
                 start_cash: float = 1000.0, state: Optional[FxState] = None):
        self.s = strategy
        self.c = costs or FxCosts()
        self.r = risk or FxRisk()
        self.st = state or FxState(cash=start_cash, peak=start_cash)

    # --------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> FxState:
        st = self.st
        idx = df.index
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        bar_hours = (idx[1] - idx[0]).total_seconds() / 3600 if len(idx) > 1 else 0.25
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

        # 2. Einstiegs-Order
        opened_now = False
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
            side, sl, tp = pos["side"], pos["sl"], pos["tp"]
            hit_sl = (l <= sl) if side == 1 else (h >= sl)
            if hit_sl:
                gap = (o <= sl) if side == 1 else (o >= sl)
                price = (o if gap and not opened_now else sl) * (1 - side * self.c.slippage)
                self._close(ts, price, self.c.taker, "stop")
            elif tp is not None and not opened_now:
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
        for act in self.s.on_bar(i, a, st.position) or []:
            kind = act[0]
            if kind == "exit" and st.position:
                st.pending_exit = act[1]
            elif kind == "order" and not st.position:
                order = dict(act[1])
                order.setdefault("expire", 1)
                st.order = order
            elif kind == "cancel":
                st.order = None

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
                       "context": od.get("context", {})}
        return True

    def _close(self, ts, price, fee, reason):
        st = self.st
        p = st.position
        gross = p["qty"] * (price - p["entry_price"]) * p["side"]
        exit_fee = p["qty"] * price * fee
        st.cash += gross - exit_fee
        pnl = gross - exit_fee - p["fees"] - p["funding"]
        risk = p["qty"] * abs(p["entry_price"] - p["initial_sl"])
        st.trades.append({
            "side": "long" if p["side"] == 1 else "short", "entry_time": p["entry_time"],
            "exit_time": ts.isoformat(), "entry_price": p["entry_price"], "exit_price": price,
            "qty": p["qty"], "pnl": pnl, "return_pct": pnl / (p["qty"] * p["entry_price"]) * 100,
            "r_multiple": pnl / risk if risk > 0 else None, "bars_held": p["bars_held"],
            "reason": reason, "initial_stop": p["initial_sl"], "tp": p["tp"],
            "fees": p["fees"] + exit_fee, "tag": p["tag"], "context": p["context"],
        })
        st.position = None

    def _equity(self, price) -> float:
        p = self.st.position
        if not p:
            return self.st.cash
        upnl = p["qty"] * (price - p["entry_price"]) * p["side"]
        return self.st.cash + upnl - p["qty"] * price * self.c.taker  # inkl. Ausstiegsgebühr
