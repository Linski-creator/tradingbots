"""Backtest der Hardcore-Strategien auf Binance-Futures-Daten.

Aufruf:  python -m bots.hardcore.backtest   (schreibt results/hardcore.json)

Zwei Kostenszenarien je Strategie und Markt:
  futures - Maker 0,02 %, Taker 0,05 %, Slippage 0,02 %, Haltekosten 0,01 %/8 h
  stress  - alles als Taker 0,05 % plus 0,05 % Slippage (prüft, ob ein Ergebnis
            nur an günstigen Annahmen hängt)
Dazu die Aufteilung in bekannte (70 %) und unbekannte (30 %) Daten.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .. import data as D
from ..metrics import summarize
from .engine import FxCosts, FxRisk, FxSimulator
from .strategies import HARDCORE

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[2] / "results"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
START = {"15m": "2022-01", "5m": "2024-01"}
SPLIT = 0.7
COSTS = {
    "futures": FxCosts(),
    "stress": FxCosts(maker=0.0005, taker=0.0005, slippage=0.0005, funding_8h=0.0001),
}


def load(symbol: str, interval: str):
    return D.binance_vision_klines(symbol, interval, start=START[interval], market="futures")


def split_metrics(st) -> dict:
    eq = st.equity
    cut = eq[int(len(eq) * SPLIT)][0]
    a = [e for e in eq if e[0] <= cut]
    b = [e for e in eq if e[0] > cut]
    m = {
        "all": summarize(eq, st.trades, 1000.0),
        "known": summarize(a, [t for t in st.trades if t["exit_time"] <= cut], 1000.0),
        "unknown": summarize(b, [t for t in st.trades if t["exit_time"] > cut], a[-1][1]),
        "split_date": cut,
    }
    days = max(m["all"]["years"] * 365.25, 1e-9)
    m["all"]["trades_per_day"] = len(st.trades) / days
    m["all"]["fees_total"] = sum(t["fees"] for t in st.trades)
    m["all"]["long_short"] = [sum(t["side"] == "long" for t in st.trades),
                              sum(t["side"] == "short" for t in st.trades)]
    return m


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    cache = {}
    results, curves = [], {}
    for name, cls in HARDCORE.items():
        strat = cls()
        for sym in SYMBOLS:
            key = (sym, strat.timeframe)
            if key not in cache:
                cache[key] = load(*key)
                log.info("%s %s: %d Kerzen ab %s", sym, strat.timeframe, len(cache[key]), cache[key].index[0])
            prepared = strat.prepare(cache[key])
            for cname, costs in COSTS.items():
                st = FxSimulator(strat, costs, FxRisk()).run(prepared)
                m = split_metrics(st)
                results.append({"strategy": name, "label": cls.label, "symbol": sym,
                                "timeframe": strat.timeframe, "costs": cname, **m})
                curves[f"{name}|{sym}|{cname}"] = st.equity[:: max(1, len(st.equity) // 1500)]
                a = m["all"]
                log.info("%-14s %-8s %-7s CAGR %7.1f %%  MaxDD %6.1f %%  Trades/Tag %5.2f  Treffer %3.0f %%",
                         name, sym, cname, a["cagr_pct"], a["max_drawdown_pct"], a["trades_per_day"],
                         a["win_rate_pct"])
    (OUT / "hardcore.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    (OUT / "hardcore_curves.json").write_text(json.dumps(curves), encoding="utf-8")
    log.info("Fertig: %d Ergebnisse", len(results))


if __name__ == "__main__":
    main()
