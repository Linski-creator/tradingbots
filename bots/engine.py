"""Simulator für eine Long-only-Spot-Strategie auf einem Markt.

Ablauf pro Bar i (Kerze mit Open, High, Low, Close):
  1. Zum Open von i wird ein ausstehender Auftrag (Signal vom Close i-1)
     ausgeführt. Ein Signal kann also nie zu einem Preis handeln, den es zum
     Entscheidungszeitpunkt noch nicht gab.
  2. Während der Bar werden Stop-Loss und Take-Profit geprüft. Treffen
     beide in derselben Bar, wird vorsichtig der Stop angenommen. Öffnet die
     Bar schon jenseits des Stops (Kurslücke), wird zum Open ausgeführt.
  3. Zum Close von i entscheidet die Strategie über den nächsten Auftrag.

Derselbe Code wird für Backtest und Paper-Trading benutzt: Der Paper-Bot
speichert den Zustand (``SimState``) und verarbeitet bei jedem Lauf nur
neue, abgeschlossene Bars.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Costs:
    fee: float = 0.001        # 0,1 % Gebühr pro Seite
    slippage: float = 0.0005  # 0,05 % schlechterer Ausführungspreis


@dataclass
class RiskConfig:
    enabled: bool = True
    risk_per_trade: float = 0.01   # 1 % vom Konto bis zum Stop
    target_vol: float = 0.30       # Ziel-Volatilität p. a. für Vol-Sizing
    max_exposure: float = 1.0      # nie mehr als 100 % investiert, kein Hebel
    halt_drawdown: Optional[float] = None  # Notbremse, z. B. 0.10


# --- Signale der Strategie -------------------------------------------------

@dataclass
class Enter:
    stop: Optional[float] = None
    rr: Optional[float] = None      # Take-Profit = Fill + rr * (Fill - Stop)
    vol: Optional[float] = None     # Volatilität am Signal-Bar (Vol-Sizing)


@dataclass
class Exit:
    reason: str = "signal"


@dataclass
class SetStop:
    stop: float


# --- Zustand ----------------------------------------------------------------

@dataclass
class Position:
    entry_time: str
    entry_price: float
    qty: float
    cost: float                 # bezahlt inkl. Gebühr
    stop: Optional[float]
    tp: Optional[float]
    initial_stop: Optional[float]
    highest: float
    bars_held: int = 0


@dataclass
class SimState:
    cash: float
    peak: float
    position: Optional[Position] = None
    pending: Optional[dict] = None
    halted: bool = False
    last_ts: Optional[str] = None
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)   # [(ts, equity)]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SimState":
        d = dict(d)
        pos = d.pop("position")
        st = SimState(**d)
        st.position = Position(**pos) if pos else None
        return st


@dataclass
class BarView:
    """Was die Strategie zum Close einer Bar sieht."""
    ts: pd.Timestamp
    row: pd.Series
    position: Optional[Position]


class Simulator:
    def __init__(self, strategy, costs: Costs, risk: RiskConfig, start_cash: float = 1000.0,
                 state: Optional[SimState] = None):
        self.strategy = strategy
        self.costs = costs
        self.risk = risk
        self.state = state or SimState(cash=start_cash, peak=start_cash)

    # ---- öffentliche API ----

    def run(self, df: pd.DataFrame) -> SimState:
        """Verarbeitet alle Bars aus ``df`` (vorbereitet von strategy.prepare),
        die neuer sind als ``state.last_ts``."""
        last = pd.Timestamp(self.state.last_ts) if self.state.last_ts else None
        for ts, row in df.iterrows():
            if last is not None and ts <= last:
                continue
            self._step(ts, row)
        return self.state

    # ---- interner Ablauf ----

    def _step(self, ts: pd.Timestamp, row: pd.Series) -> None:
        st = self.state
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        for v in (o, h, l, c):
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"Ungültiger Preis in Bar {ts}: {row.to_dict()}")
        if not (l <= min(o, c) and h >= max(o, c)):
            raise ValueError(f"Inkonsistente Kerze {ts}: {row.to_dict()}")

        # 1. ausstehenden Auftrag zum Open ausführen
        if st.pending is not None:
            p, st.pending = st.pending, None
            if p["type"] == "exit" and st.position is not None:
                self._close(ts, o * (1 - self.costs.slippage), p["reason"])
            elif p["type"] == "enter" and st.position is None and not st.halted:
                self._open(ts, o, p)

        # 2. Stop / Take-Profit innerhalb der Bar
        pos = st.position
        if pos is not None:
            pos.highest = max(pos.highest, h)
            if pos.stop is not None and l <= pos.stop:
                price = o if o < pos.stop else pos.stop
                self._close(ts, price * (1 - self.costs.slippage), "stop")
            elif pos.tp is not None and h >= pos.tp:
                price = o if o > pos.tp else pos.tp
                self._close(ts, price, "take_profit")

        # 3. Bewertung zum Close, Notbremse
        equity = self._equity(c)
        st.peak = max(st.peak, equity)
        st.equity.append((ts.isoformat(), equity))
        st.last_ts = ts.isoformat()
        if (self.risk.enabled and self.risk.halt_drawdown is not None and not st.halted
                and equity <= st.peak * (1 - self.risk.halt_drawdown)):
            st.halted = True
            if st.position is not None:
                st.pending = {"type": "exit", "reason": "notbremse"}
            return
        if st.halted:
            return

        # 4. Strategie entscheidet zum Close
        if st.position is not None:
            st.position.bars_held += 1
        signals = self.strategy.on_bar(BarView(ts, row, st.position)) or []
        for sig in signals:
            if isinstance(sig, Exit) and st.position is not None:
                st.pending = {"type": "exit", "reason": sig.reason}
            elif isinstance(sig, SetStop) and st.position is not None:
                if math.isfinite(sig.stop):
                    old = st.position.stop
                    # Stops werden nur nachgezogen, nie gelockert
                    st.position.stop = sig.stop if old is None else max(old, sig.stop)
            elif isinstance(sig, Enter) and st.position is None and st.pending is None:
                st.pending = {"type": "enter", "stop": sig.stop, "rr": sig.rr, "vol": sig.vol,
                              "signal_ts": ts.isoformat()}

    def _equity(self, price: float) -> float:
        st = self.state
        if st.position is None:
            return st.cash
        return st.cash + st.position.qty * price * (1 - self.costs.fee)

    def _open(self, ts: pd.Timestamp, open_price: float, p: dict) -> None:
        st = self.state
        fill = open_price * (1 + self.costs.slippage)
        stop = p.get("stop")
        if stop is not None and stop >= fill:
            return  # Kurs ist schon unter den Stop gefallen: kein Einstieg
        equity = st.cash
        cap = equity * self.risk.max_exposure
        if not self.risk.enabled or self.strategy.sizing == "all":
            notional = cap
        elif self.strategy.sizing == "risk":
            if stop is None:
                raise ValueError(f"{self.strategy.name}: Risiko-Sizing braucht einen Stop")
            dist = (fill - stop) / fill + self.costs.fee * 2 + self.costs.slippage
            notional = min(cap, equity * self.risk.risk_per_trade / dist)
        elif self.strategy.sizing == "vol":
            vol = p.get("vol")
            if vol is None or not math.isfinite(vol) or vol <= 0:
                return
            notional = min(cap, equity * self.risk.target_vol / vol)
        else:
            raise ValueError(f"Unbekanntes Sizing: {self.strategy.sizing}")
        if notional < 1e-9:
            return
        qty = notional / (fill * (1 + self.costs.fee))
        cost = qty * fill * (1 + self.costs.fee)
        st.cash -= cost
        tp = None
        if p.get("rr") is not None and stop is not None:
            tp = fill + p["rr"] * (fill - stop)
        st.position = Position(entry_time=ts.isoformat(), entry_price=fill, qty=qty, cost=cost,
                               stop=stop, tp=tp, initial_stop=stop, highest=open_price)

    def _close(self, ts: pd.Timestamp, price: float, reason: str) -> None:
        st = self.state
        pos = st.position
        proceeds = pos.qty * price * (1 - self.costs.fee)
        st.cash += proceeds
        pnl = proceeds - pos.cost
        risk_amount = None
        if pos.initial_stop is not None:
            risk_amount = pos.qty * (pos.entry_price - pos.initial_stop)
        st.trades.append({
            "entry_time": pos.entry_time,
            "exit_time": ts.isoformat(),
            "entry_price": pos.entry_price,
            "exit_price": price,
            "qty": pos.qty,
            "pnl": pnl,
            "return_pct": pnl / pos.cost * 100,
            "r_multiple": pnl / risk_amount if risk_amount else None,
            "bars_held": pos.bars_held,
            "reason": reason,
        })
        st.position = None
