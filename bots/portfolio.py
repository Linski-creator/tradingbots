"""Strategien über mehrere Märkte: Dual Momentum und Funding-Arbitrage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .engine import Costs, RiskConfig
from . import indicators as ind


@dataclass
class RotationState:
    cash: float
    peak: float
    holding: Optional[str] = None
    qty: float = 0.0
    cost: float = 0.0
    entry_time: Optional[str] = None
    entry_context: Optional[dict] = None
    pending: Optional[dict] = None
    halted: bool = False
    last_ts: Optional[str] = None
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)


class DualMomentum:
    """Hält jede Woche den Markt mit der höchsten Rendite der letzten
    ``lookback`` Tage, aber nur wenn diese Rendite positiv ist. Sonst Cash.

    Entscheidung am Close des letzten Tages der Woche (Sonntag),
    Umschichtung zum Open des nächsten Tages.
    """
    name, label, timeframe = "dualmom", "Dual Momentum", "1d"

    def __init__(self, lookback: int = 90, vol_n: int = 30, rebalance_weekday: int = 6):
        self.lookback, self.vol_n, self.rebalance_weekday = lookback, vol_n, rebalance_weekday
        self.warmup = max(lookback, vol_n) + 1

    def prepare(self, closes: dict[str, pd.DataFrame], bars_per_year: float) -> dict[str, pd.DataFrame]:
        out = {}
        for sym, df in closes.items():
            d = df.copy()
            d["mom"] = d["close"] / d["close"].shift(self.lookback) - 1
            d["vol"] = ind.realized_vol(d["close"], self.vol_n, bars_per_year)
            out[sym] = d
        return out

    def target(self, rows: dict[str, pd.Series]) -> Optional[str]:
        best, best_mom = None, 0.0
        for sym, r in rows.items():
            m = r["mom"]
            if m is not None and math.isfinite(m) and m > best_mom:
                best, best_mom = sym, m
        return best


def run_rotation(strategy: DualMomentum, data: dict[str, pd.DataFrame], costs: Costs,
                 risk: RiskConfig, start_cash: float = 1000.0,
                 state: Optional[RotationState] = None) -> RotationState:
    """Simuliert die Rotation. Nur Tage, an denen ALLE Märkte eine Kerze
    haben, werden verarbeitet (gemeinsamer Index)."""
    st = state or RotationState(cash=start_cash, peak=start_cash)
    index = None
    for df in data.values():
        index = df.index if index is None else index.intersection(df.index)
    last = pd.Timestamp(st.last_ts) if st.last_ts else None

    for ts in index:
        if last is not None and ts <= last:
            continue
        rows = {s: data[s].loc[ts] for s in data}

        # 1. Umschichtung zum Open
        if st.pending is not None:
            p, st.pending = st.pending, None
            target = p["target"]
            if st.holding is not None and st.holding != target:
                price = float(rows[st.holding]["open"]) * (1 - costs.slippage)
                proceeds = st.qty * price * (1 - costs.fee)
                st.trades.append({"symbol": st.holding, "entry_time": st.entry_time,
                                  "exit_time": ts.isoformat(), "pnl": proceeds - st.cost,
                                  "return_pct": (proceeds / st.cost - 1) * 100, "reason": p["reason"],
                                  "entry_price": st.cost / st.qty, "exit_price": price,
                                  "context": st.entry_context or {}})
                st.cash += proceeds
                st.holding, st.qty, st.cost = None, 0.0, 0.0
            if target is not None and st.holding is None and not st.halted:
                fill = float(rows[target]["open"]) * (1 + costs.slippage)
                cap = st.cash * risk.max_exposure
                if risk.enabled:
                    vol = p.get("vol")
                    if vol is not None and math.isfinite(vol) and vol > 0:
                        notional = min(cap, st.cash * risk.target_vol / vol)
                    else:
                        notional = 0.0
                else:
                    notional = cap
                if notional > 1e-9:
                    st.qty = notional / (fill * (1 + costs.fee))
                    st.cost = st.qty * fill * (1 + costs.fee)
                    st.cash -= st.cost
                    st.holding, st.entry_time = target, ts.isoformat()
                    st.entry_context = p.get("context")

        # 2. Bewertung
        equity = st.cash
        if st.holding is not None:
            equity += st.qty * float(rows[st.holding]["close"]) * (1 - costs.fee)
        st.peak = max(st.peak, equity)
        st.equity.append((ts.isoformat(), equity))
        st.last_ts = ts.isoformat()

        if (risk.enabled and risk.halt_drawdown is not None and not st.halted
                and equity <= st.peak * (1 - risk.halt_drawdown)):
            st.halted = True
            if st.holding is not None:
                st.pending = {"target": None, "reason": "notbremse"}
            continue
        if st.halted:
            continue

        # 3. Wöchentliche Entscheidung
        if ts.weekday() == strategy.rebalance_weekday:
            target = strategy.target(rows)
            if target != st.holding:
                vol = float(rows[target]["vol"]) if target else None
                ctx = {f"mom_{s}": float(r["mom"]) for s, r in rows.items() if math.isfinite(r["mom"])}
                st.pending = {"target": target, "vol": vol, "reason": "rotation", "context": ctx}
    return st


# --- Funding-Arbitrage -------------------------------------------------------

@dataclass
class FundingConfig:
    enter_rate: float = 0.00005   # Ø-Funding pro 8 h für Einstieg (~5,5 % p. a.)
    exit_rate: float = 0.0        # Ausstieg, wenn Ø-Funding <= 0
    avg_n: int = 9                # Durchschnitt über 9 Zahlungen = 3 Tage
    hedge_fraction: float = 0.5   # 50 % Spot, 50 % als Sicherheit für den Short
    spot_fee: float = 0.001
    perp_fee: float = 0.0005
    slippage: float = 0.0005


@dataclass
class FundingState:
    cash: float
    peak: float
    active: bool = False
    notional: float = 0.0
    last_ts: Optional[str] = None
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)
    current: Optional[dict] = None


class FundingArbitrage:
    name, label, timeframe = "funding", "Funding-Arbitrage", "8h"

    def __init__(self, cfg: Optional[FundingConfig] = None):
        self.cfg = cfg or FundingConfig()


def run_funding(strategy: FundingArbitrage, funding: pd.Series, start_cash: float = 1000.0,
                state: Optional[FundingState] = None) -> FundingState:
    """Simuliert Spot-Long + Perp-Short auf demselben Coin.

    ``funding``: Funding-Rate pro Zahlung (z. B. alle 8 h), Index = Zeitpunkt.
    Positive Rate: Longs zahlen an Shorts, wir kassieren rate * Nominal.

    Vereinfachungen (im Bericht genannt): Die Absicherung gilt als perfekt
    (Kursbewegung gleicht sich aus), Basis-Schwankungen und Börsenrisiko
    sind nicht enthalten. Ein- und Ausstieg kosten 4 Gebühren (je 2 Seiten
    Spot und Perp) plus Slippage.
    """
    cfg = strategy.cfg
    st = state or FundingState(cash=start_cash, peak=start_cash)
    last = pd.Timestamp(st.last_ts) if st.last_ts else None
    avg = funding.rolling(cfg.avg_n, min_periods=cfg.avg_n).mean()
    leg_cost = cfg.spot_fee + cfg.perp_fee + 2 * cfg.slippage

    for ts, rate in funding.items():
        if last is not None and ts <= last:
            continue
        if not math.isfinite(rate):
            raise ValueError(f"Ungültige Funding-Rate bei {ts}")
        # Zahlung für die laufende Position (Entscheidung fiel vorher)
        if st.active:
            income = st.notional * rate
            st.cash += income
            st.current["funding"] += income
        a = avg.loc[ts]
        if math.isfinite(a):
            if not st.active and a > cfg.enter_rate:
                st.notional = st.cash * cfg.hedge_fraction
                fee = st.notional * leg_cost
                st.cash -= fee
                st.active = True
                st.current = {"entry_time": ts.isoformat(), "funding": 0.0, "fees": fee,
                              "context": {"avg_rate": float(a)}}
            elif st.active and a <= cfg.exit_rate:
                fee = st.notional * leg_cost
                st.cash -= fee
                c = st.current
                c["fees"] += fee
                st.trades.append({"entry_time": c["entry_time"], "exit_time": ts.isoformat(),
                                  "pnl": c["funding"] - c["fees"], "funding": c["funding"],
                                  "fees": c["fees"], "reason": "funding_negativ",
                                  "context": {**c.get("context", {}), "exit_avg_rate": float(a)}})
                st.active, st.notional, st.current = False, 0.0, None
        # Bewertung inkl. der Kosten, die ein Ausstieg jetzt verursachen würde
        equity = st.cash - (st.notional * leg_cost if st.active else 0.0)
        st.peak = max(st.peak, equity)
        st.equity.append((ts.isoformat(), equity))
        st.last_ts = ts.isoformat()
    return st
