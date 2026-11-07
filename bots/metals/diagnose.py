"""Diagnose: Woran scheitern die Trades? Und was bringen gezielte Reparaturen?

1. Vermessung der Trades der Basis-Version (v3):
   - Stop-Verluste, die vorher schon >= 1R im Plus waren (-> Teilgewinn würde helfen)
   - Stop-Verluste, nach denen der Kurs binnen 20 Kerzen 2R in die alte Richtung lief
     (-> Stop saß zu knapp, "ausgestoppt und dann lief es")
   - Wochenend-Ausstiege: wie viel lief der Trend danach noch weiter?
2. Varianten mit jeweils einer Reparatur (vorab festgelegt, keine Optimierung).

Aufruf: python -m bots.metals.diagnose   (schreibt results/diagnose.json)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..hardcore.engine import FxRisk, FxSimulator
from ..hardcore.search import split
from .backtest import COSTS, stress
from .trend3 import TrendFollowerV3, load

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[2] / "results"

VARIANTS = {
    "basis": {},
    "stop_mit_luft": dict(wide_stop=True),
    "montag_wieder_rein": dict(reentry=True),
    "einstieg_an_marken": dict(levels=True),
    "luft_plus_montag": dict(wide_stop=True, reentry=True),
    "alles": dict(wide_stop=True, reentry=True, levels=True),
    "luft_montag_teilgewinn": dict(wide_stop=True, reentry=True, partial=True),
    "luft_teilgewinn": dict(wide_stop=True, partial=True),
}
JOBS = [("PAXGUSDT", "4h", "2024-01-01"), ("PAXGUSDT", "1h", "2024-01-01")] + \
       [(s, tf, None) for s in ("GC=F", "SI=F", "PL=F", "JPY=X") for tf in ("4h", "1h")]


def anatomy(df: pd.DataFrame, trades: list) -> dict:
    """Wie enden die Trades, und was passierte danach?"""
    idx = df.index
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    stops = [t for t in trades if t["reason"] == "stop" and (t["r_multiple"] or 0) < 0]
    weekend = [t for t in trades if t["reason"] == "wochenende"]
    were_up = sum(1 for t in stops if (t.get("mfe_r") or 0) >= 1)
    ran_after, weekend_more = 0, []
    for t in stops + weekend:
        r = abs(t["entry_price"] - t["initial_stop"])
        j = int(idx.searchsorted(pd.Timestamp(t["exit_time"])))
        seg = slice(j + 1, min(j + 21, len(idx)))
        if j + 1 >= len(idx) or r <= 0:
            continue
        if t["side"] == "long":
            move = (hi[seg].max() - t["entry_price"]) / r
        else:
            move = (t["entry_price"] - lo[seg].min()) / r
        if t in stops and move >= 2:
            ran_after += 1
        if t in weekend:
            weekend_more.append(round(float(move - (t["r_multiple"] or 0)), 2))
    n = len(trades)
    return {
        "trades": n,
        "exits": {k: sum(t["reason"] == k for t in trades) for k in {t["reason"] for t in trades}},
        "stop_losses": len(stops),
        "stop_losses_vorher_1R_im_plus": were_up,
        "stop_losses_danach_2R_in_richtung": ran_after,
        "wochenende_danach_noch_moeglich_R_median": float(np.median(weekend_more)) if weekend_more else None,
        "avg_mfe_r_verlierer": round(float(np.mean([t["mfe_r"] for t in trades if t["pnl"] <= 0
                                                    and t.get("mfe_r") is not None] or [0])), 2),
        "avg_mfe_r_gewinner": round(float(np.mean([t["mfe_r"] for t in trades if t["pnl"] > 0
                                                   and t.get("mfe_r") is not None] or [0])), 2),
        "win_rate": round(100 * sum(t["pnl"] > 0 for t in trades) / n, 1) if n else 0,
        "avg_r": round(float(np.mean([t["r_multiple"] for t in trades if t["r_multiple"] is not None] or [0])), 3),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    out = []
    for sym, tf, cut in JOBS:
        df = load(sym, tf)
        cut = cut or (df.index[-1] - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        for vname, flags in VARIANTS.items():
            strat = TrendFollowerV3(tf, **flags)
            p = strat.prepare(df)
            risk = FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.0005, halt_drawdown=None)
            st = FxSimulator(strat, COSTS[sym], risk).run(p)
            st_s = FxSimulator(strat, stress(COSTS[sym]), risk).run(p)
            m, ms = split(st, cut), split(st_s, cut)
            row = {"symbol": sym, "timeframe": tf, "variant": vname, "split": cut,
                   "anatomy": anatomy(p, st.trades),
                   "selection": m["selection"], "validation": m["validation"],
                   "validation_stress": ms["validation"]}
            out.append(row)
            a = row["anatomy"]
            log.info("%-8s %-3s %-20s Treffer %4.1f %% Ø %+.2fR | Suche %+6.1f %% Prüfung %+6.1f %% (Stress %+6.1f %%)",
                     sym, tf, vname, a["win_rate"], a["avg_r"], m["selection"].get("total_return_pct", 0),
                     m["validation"].get("total_return_pct", 0), ms["validation"].get("total_return_pct", 0))
    (OUT / "diagnose.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
