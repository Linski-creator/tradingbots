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
from .metrics import equity_series, summarize
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


RISK_LEVELS = {
    "vorsichtig": RiskConfig(risk_per_trade=0.01, target_vol=0.15),
    "standard": RiskConfig(risk_per_trade=0.01, target_vol=0.30),
    "mutig": RiskConfig(risk_per_trade=0.02, target_vol=0.40),
    "ohne": RiskConfig(enabled=False),
}
# Diese beiden lagen im ersten Backtest klar im Minus: nur zur Dokumentation
WEAK = {"rsi2", "pullback"}

# Portfolio-Kandidaten: (Strategie, Markt)
PORTFOLIO = [("donchian", "BTCUSDT"), ("donchian", "BNBUSDT"), ("chandelier", "BTCUSDT"),
             ("tsmom", "BTCUSDT"), ("sma200", "PAXGUSDT"), ("dualmom", "ROT"), ("funding", "BTCUSDT")]


def run_single(name: str, df: pd.DataFrame, bpy: float, risk: RiskConfig):
    strat = SINGLE_MARKET[name]()
    st = Simulator(strat, Costs(), risk).run(strat.prepare(df, bpy))
    return st, _split_metrics(st, 1000.0)


def daily(equity: list) -> pd.Series:
    return equity_series(equity).resample("1D").last().ffill()


def portfolio(curves: dict[str, pd.Series], members: list[str], start_cash: float = 1000.0) -> list:
    """Gleichgewichtet, täglich neu ausbalanciert, ab dem Tag, an dem alle
    Mitglieder Daten haben."""
    df = pd.concat({m: curves[m] for m in members}, axis=1).dropna()
    rets = df.pct_change().dropna().mean(axis=1)
    eq = start_cash * (1 + rets).cumprod()
    return [(ts.isoformat(), float(v)) for ts, v in eq.items()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    datasets = load_all()
    results, curves, dailies = [], {}, {}

    def record(strategy, label, symbol, itv, level, st, m):
        key = f"{strategy}|{symbol}|{level}"
        results.append({"strategy": strategy, "label": label, "symbol": symbol, "timeframe": itv,
                        "level": level, "risk_rules": level != "ohne", **m})
        curves[key] = st.equity[:: max(1, len(st.equity) // 1500)]
        dailies[key] = daily(st.equity)
        log.info("%-32s CAGR %6.1f %%  MaxDD %6.1f %%", key,
                 m["all"].get("cagr_pct", 0), m["all"].get("max_drawdown_pct", 0))

    for name, cls in SINGLE_MARKET.items():
        itv = cls.timeframe
        markets = list(MARKETS) + (["GOLD"] if itv == "1d" and ("GOLD", "1d") in datasets else [])
        levels = ["standard", "ohne"] if name in WEAK else list(RISK_LEVELS)
        if name == "emacross":  # Video-Strategie: nur wie im Video, auf den großen Coins
            levels = ["ohne"]
            markets = [m for m in ("BTCUSDT", "ETHUSDT", "SOLUSDT") if m in MARKETS]
        for sym in markets:
            df = datasets[(sym, itv)]
            bpy = BPY_GOLD_FUTURES if sym == "GOLD" else BPY_CRYPTO[itv]
            for level in levels:
                st, m = run_single(name, df, bpy, RISK_LEVELS[level])
                record(name, cls.label, sym, itv, level, st, m)

    # Dual Momentum: BTC / ETH / Gold-Token / Cash
    dm = DualMomentum()
    universe = {s: datasets[(s, "1d")] for s in ("BTCUSDT", "ETHUSDT", "PAXGUSDT")}
    prepared = dm.prepare(universe, BPY_CRYPTO["1d"])
    for level, risk in RISK_LEVELS.items():
        st = run_rotation(dm, prepared, Costs(), risk)
        record(dm.name, dm.label, "ROT", "1d", level, st, _split_metrics(st, 1000.0))

    # Funding-Arbitrage (marktneutral, Risikostufe ohne Einfluss)
    fa = FundingArbitrage()
    for sym in ("BTCUSDT", "ETHUSDT"):
        st = run_funding(fa, D.binance_funding_history(sym))
        m = _split_metrics(st, 1000.0)
        for level in ("vorsichtig", "standard", "mutig"):
            record(fa.name, fa.label, sym, "8h", level, st, m)

    # Portfolio aus den Kandidaten, je Risikostufe
    for level in ("vorsichtig", "standard", "mutig"):
        members = [f"{s}|{m}|{level}" for s, m in PORTFOLIO]
        eq = portfolio(dailies, members)
        results.append({"strategy": "portfolio", "label": "Portfolio (7 Bots)", "symbol": "MIX",
                        "timeframe": "1d", "level": level, "risk_rules": True,
                        "members": members, **_split_metrics_eq(eq)})
        curves[f"portfolio|MIX|{level}"] = eq[:: max(1, len(eq) // 1500)]
    corr = pd.concat({f"{s}|{m}": dailies[f"{s}|{m}|standard"] for s, m in PORTFOLIO}, axis=1) \
        .dropna().pct_change().dropna().corr().round(2)

    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (OUT / "curves.json").write_text(json.dumps(curves))
    (OUT / "correlation.json").write_text(corr.to_json())
    log.info("Fertig: %d Ergebnisse in %s", len(results), OUT)


def _split_metrics_eq(eq: list) -> dict:
    class _S:  # Equity ohne Einzeltrades
        equity, trades = eq, []
    return _split_metrics(_S, 1000.0)


if __name__ == "__main__":
    main()
