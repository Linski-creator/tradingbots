"""Backtest aller Strategien auf allen Märkten.

Aufruf:  python -m bots.backtest            (lädt Daten, schreibt results/)

Für jede Kombination aus Strategie und Markt wird zweimal gerechnet:
mit Risikoregeln (1 % Risiko / Vol-Sizing) und ohne (immer 100 % investiert).
Zusätzlich werden die Kennzahlen getrennt für die ersten 70 % ("bekannt")
und die letzten 30 % ("unbekannt") der Daten ausgewiesen. Da keine Parameter
optimiert werden, zeigt das, ob eine Strategie über die Zeit stabil ist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import data as D
from .engine import Costs, RiskConfig, Simulator
from .metrics import summarize
from .portfolio import DualMomentum, FundingArbitrage, run_funding, run_rotation
from .strategies import SINGLE_MARKET

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parent.parent / "results"

MARKETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "PAXGUSDT"]
BPY_CRYPTO = {"1h": 8760, "4h": 2190, "1d": 365}
BPY_GOLD_FUTURES = 252
SPLIT = 0.7


def load_all() -> dict:
    """{(symbol, interval): DataFrame}. Gold zusätzlich als lange Tagesreihe."""
    out = {}
    for sym in MARKETS:
        for itv in ("1h", "4h", "1d"):
            out[(sym, itv)] = D.binance_vision_klines(sym, itv)
            log.info("%s %s: %d Kerzen ab %s", sym, itv, len(out[(sym, itv)]), out[(sym, itv)].index[0])
    try:
        out[("GOLD", "1d")] = D.yahoo_daily("GC=F")
    except Exception as e:  # noqa: BLE001
        log.warning("Gold-Historie (Yahoo) nicht verfügbar: %s", e)
    return out


def _split_metrics(state, start_cash: float) -> dict:
    eq = state.equity
    if len(eq) < 10:
        return {}
    cut = eq[int(len(eq) * SPLIT)][0]
    eq_a = [e for e in eq if e[0] <= cut]
    eq_b = [e for e in eq if e[0] > cut]
    tr_a = [t for t in state.trades if t["exit_time"] <= cut]
    tr_b = [t for t in state.trades if t["exit_time"] > cut]
    return {
        "all": summarize(eq, state.trades, start_cash),
        "known": summarize(eq_a, tr_a, start_cash),
        "unknown": summarize(eq_b, tr_b, eq_a[-1][1]),
        "split_date": cut,
    }


def run_single(name: str, symbol: str, df: pd.DataFrame, bpy: float, risk_on: bool) -> tuple:
    strat = SINGLE_MARKET[name]()
    prepared = strat.prepare(df, bpy)
    risk = RiskConfig(enabled=risk_on)
    st = Simulator(strat, Costs(), risk).run(prepared)
    return st, _split_metrics(st, 1000.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    datasets = load_all()
    results, curves = [], {}

    for name, cls in SINGLE_MARKET.items():
        itv = cls.timeframe
        markets = [s for s in MARKETS] + (["GOLD"] if itv == "1d" and ("GOLD", "1d") in datasets else [])
        for sym in markets:
            df = datasets[(sym, itv)]
            bpy = BPY_GOLD_FUTURES if sym == "GOLD" else BPY_CRYPTO[itv]
            for risk_on in (True, False):
                st, m = run_single(name, sym, df, bpy, risk_on)
                key = f"{name}|{sym}|{'regeln' if risk_on else 'ohne'}"
                results.append({"strategy": name, "label": cls.label, "symbol": sym, "timeframe": itv,
                                "risk_rules": risk_on, **m})
                curves[key] = st.equity[:: max(1, len(st.equity) // 1500)]
                log.info("%-28s CAGR %6.1f %%  MaxDD %6.1f %%", key,
                         m["all"].get("cagr_pct", 0), m["all"].get("max_drawdown_pct", 0))

    # Dual Momentum: BTC / ETH / Gold-Token / Cash
    dm = DualMomentum()
    universe = {s: datasets[(s, "1d")] for s in ("BTCUSDT", "ETHUSDT", "PAXGUSDT")}
    prepared = dm.prepare(universe, BPY_CRYPTO["1d"])
    for risk_on in (True, False):
        st = run_rotation(dm, prepared, Costs(), RiskConfig(enabled=risk_on))
        m = _split_metrics(st, 1000.0)
        results.append({"strategy": dm.name, "label": dm.label, "symbol": "BTC/ETH/PAXG",
                        "timeframe": "1d", "risk_rules": risk_on, **m})
        curves[f"dualmom|ROT|{'regeln' if risk_on else 'ohne'}"] = st.equity[:: max(1, len(st.equity) // 1500)]

    # Funding-Arbitrage
    fa = FundingArbitrage()
    for sym in ("BTCUSDT", "ETHUSDT"):
        rates = D.binance_funding_history(sym)
        st = run_funding(fa, rates)
        m = _split_metrics(st, 1000.0)
        results.append({"strategy": fa.name, "label": fa.label, "symbol": sym, "timeframe": "8h",
                        "risk_rules": True, **m})
        curves[f"funding|{sym}|regeln"] = st.equity[:: max(1, len(st.equity) // 1500)]

    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (OUT / "curves.json").write_text(json.dumps(curves))
    log.info("Fertig: %d Ergebnisse in %s", len(results), OUT)


if __name__ == "__main__":
    main()
