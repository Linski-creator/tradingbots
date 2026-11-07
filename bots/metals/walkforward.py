"""Walk-Forward-Optimierung der Rücksetzer-Strategie.

Statt "so lange optimieren, bis der Backtest perfekt aussieht" (Überanpassung):
  1. Optimieren auf einem Trainingsfenster (beste Kombination nach Profitfaktor)
  2. Diese Kombination auf dem DARAUF folgenden, ungesehenen Testfenster laufen lassen
  3. Fenster weiterrücken, wiederholen
Gezählt wird nur die aneinandergereihte Kurve der Testfenster ("out of sample").
Zum Vergleich: die festen Standardwerte (EMA 150, RSI 35, Stop 2, Ziel 1) über
dieselben Testfenster, und wie oft jede Kombination in den Testfenstern Gewinn macht.

Aufruf: python -m bots.metals.walkforward   (schreibt results/walkforward.json)
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import pandas as pd

from .. import data as D
from ..hardcore.engine import FxRisk, FxSimulator
from ..metrics import equity_series
from .backtest import COSTS, TrendDip, stress

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[2] / "results"

GRID = {"ema": [100, 150, 200], "level": [30, 35, 40], "sl": [1.0, 1.5, 2.0, 3.0], "tp": [1.0, 1.5, 2.0, 3.0]}
DEFAULT = (150, 35, 2.0, 1.0)
MIN_TRADES = 10          # weniger Trades im Training = Kombination wird nicht gewählt
# (Symbol, Trainingslänge, Testlänge) - PAXG hat 6 Jahre, Yahoo-Futures nur 2 Jahre
MARKETS = [("PAXGUSDT", "730D", "182D"), ("GC=F", "365D", "91D"), ("SI=F", "365D", "91D"),
           ("PL=F", "365D", "91D"), ("JPY=X", "365D", "91D")]


def combos():
    return list(itertools.product(GRID["ema"], GRID["level"], GRID["sl"], GRID["tp"]))


def run_combo(df: pd.DataFrame, sym: str, combo: tuple, costs=None) -> tuple[pd.Series, list]:
    ema, level, sl, tp = combo
    strat = TrendDip(sl, tp, ema_n=ema, level=level)
    st = FxSimulator(strat, costs or COSTS[sym], FxRisk(risk_per_trade=0.01, max_leverage=5,
                                                        min_stop=0.0005)).run(strat.prepare(df))
    return equity_series(st.equity), st.trades


def window_stats(eq: pd.Series, trades: list, a: pd.Timestamp, b: pd.Timestamp) -> dict:
    seg = eq[(eq.index >= a) & (eq.index < b)]
    if len(seg) < 2:
        return {"ret": 0.0, "pf": 0.0, "trades": 0}
    tr = [t for t in trades if a.isoformat() <= t["entry_time"] < b.isoformat()]
    wins = sum(t["pnl"] for t in tr if t["pnl"] > 0)
    loss = -sum(t["pnl"] for t in tr if t["pnl"] <= 0)
    pf = wins / loss if loss > 0 else (9.99 if wins > 0 else 0.0)
    return {"ret": float(seg.iloc[-1] / seg.iloc[0] - 1), "pf": float(pf), "trades": len(tr)}


def windows(index: pd.DatetimeIndex, train: str, test: str) -> list[tuple]:
    out, start = [], index[0]
    while True:
        tr_end = start + pd.Timedelta(train)
        te_end = tr_end + pd.Timedelta(test)
        if tr_end >= index[-1]:
            break
        out.append((start, tr_end, min(te_end, index[-1] + pd.Timedelta("1h"))))
        start = start + pd.Timedelta(test)
    return out


def walk_forward(df: pd.DataFrame, sym: str, train: str, test: str) -> dict:
    runs = {c: run_combo(df, sym, c) for c in combos()}
    stress_runs = {}
    wins = windows(df.index, train, test)
    oos, oos_stress, base, chosen = 1.0, 1.0, 1.0, []
    profitable = {c: 0 for c in runs}
    for a, b, c_end in wins:
        best, best_pf = None, -1.0
        for c, (eq, tr) in runs.items():
            s = window_stats(eq, tr, a, b)
            if s["trades"] >= MIN_TRADES and s["pf"] > best_pf:
                best, best_pf = c, s["pf"]
        for c, (eq, tr) in runs.items():
            profitable[c] += window_stats(eq, tr, b, c_end)["ret"] > 0
        d = window_stats(*runs[DEFAULT], b, c_end)
        base *= 1 + d["ret"]
        if best is None:  # zu wenige Trades: Testfenster aussetzen (Cash)
            chosen.append({"train": [a.isoformat()[:10], b.isoformat()[:10]], "combo": None,
                           "test_ret": 0.0, "default_ret": d["ret"]})
            continue
        if best not in stress_runs:
            stress_runs[best] = run_combo(df, sym, best, stress(COSTS[sym]))
        t = window_stats(*runs[best], b, c_end)
        ts = window_stats(*stress_runs[best], b, c_end)
        oos *= 1 + t["ret"]
        oos_stress *= 1 + ts["ret"]
        chosen.append({"train": [a.isoformat()[:10], b.isoformat()[:10]], "test_end": c_end.isoformat()[:10],
                       "combo": dict(zip(GRID, best)), "train_pf": round(best_pf, 2),
                       "test_ret": round(t["ret"] * 100, 2), "test_ret_stress": round(ts["ret"] * 100, 2),
                       "test_trades": t["trades"], "default_ret": round(d["ret"] * 100, 2)})
    n = len(wins)
    robust = sorted(((profitable[c] / n if n else 0, c) for c in runs), reverse=True)[:10]
    full_default = window_stats(*runs[DEFAULT], df.index[0], df.index[-1] + pd.Timedelta("1h"))
    return {
        "symbol": sym, "windows": n, "data_from": df.index[0].isoformat()[:10],
        "oos_return_pct": round((oos - 1) * 100, 2),
        "oos_return_stress_pct": round((oos_stress - 1) * 100, 2),
        "default_oos_return_pct": round((base - 1) * 100, 2),
        "default_full_period": {k: round(v, 3) for k, v in full_default.items()},
        "most_robust": [{"share_profitable_windows": round(s, 2), "combo": dict(zip(GRID, c))} for s, c in robust],
        "windows_detail": chosen,
    }


def load(sym: str) -> pd.DataFrame:
    if sym == "PAXGUSDT":
        return D.binance_vision_klines(sym, "1h", start="2020-01")
    return D.yahoo_hourly(sym)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    results = []
    for sym, train, test in MARKETS:
        df = load(sym)
        log.info("%s: %d Kerzen ab %s", sym, len(df), df.index[0])
        r = walk_forward(df, sym, train, test)
        results.append(r)
        log.info("%s: %d Fenster | optimiert (ungesehen) %+.1f %% | Stress %+.1f %% | Standard %+.1f %%",
                 sym, r["windows"], r["oos_return_pct"], r["oos_return_stress_pct"], r["default_oos_return_pct"])
    (OUT / "walkforward.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
