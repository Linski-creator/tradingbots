"""Idee E: engerer Stop von Beginn an.

1. Messung (nur 2000-2015, Gold, Standard = Basis + B): Wie tief fallen Trades nach dem
   Einstieg, bevor sie laufen (MAE)? Besonders die Traum-Trades (beste 10 %).
2. Durchrechnung: Start-Stop auf 2/3, 1/2, 1/3 des bisherigen Abstands (der nachgezogene
   Stop bleibt gleich). Bei 1 % Risiko wird die Position entsprechend größer.
   Bewertet in Suche (bis 2015) und Prüfung (ab 2016), Gold und Silber, normal und Stress.

Aufruf: python -m bots.metals.stop_study   (schreibt results/stop_study.json)
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pandas as pd

from .. import data as D
from ..hardcore.engine import FxRisk, FxSimulator
from ..hardcore.search import split
from .backtest import COSTS, stress
from .daily import OUT, tidy
from .variants import SETUPS, Variant

log = logging.getLogger(__name__)
CUT = "2016-01-01"
SCALES = (1.0, 2 / 3, 1 / 2, 1 / 3)


class TightStart(Variant):
    def __init__(self, fast, slow, k, lag, scale: float):
        super().__init__(fast, slow, k, lag, "B")
        self.scale = scale

    def on_bar(self, i, a, pos):
        acts = super().on_bar(i, a, pos)
        out = []
        for act in acts:
            if act[0] == "order":
                od = dict(act[1])
                c = a["close"][i]
                od["sl"] = float(c - self.scale * (c - od["sl"]))
                act = ("order", od)
            out.append(act)
        return out


def pct(x, q):
    return round(float(np.percentile(x, q)), 2) if len(x) else None


def mae_profile(trades: list) -> dict:
    t = [x for x in trades if x["exit_time"] < CUT and x["mae_r"] is not None]
    win = [x for x in t if x["pnl"] > 0]
    lose = [x for x in t if x["pnl"] <= 0]
    top = sorted(win, key=lambda x: x["r_multiple"], reverse=True)[:max(len(t) // 10, 1)]
    def depth(group):
        m = np.array([-x["mae_r"] for x in group])   # positiv = so tief im Minus (in R)
        return {"anzahl": len(group), "median_tiefe_r": pct(m, 50), "p75_tiefe_r": pct(m, 75),
                "p90_tiefe_r": pct(m, 90),
                "tiefer_als_0.33R_pct": round(float((m > 1 / 3).mean() * 100), 1) if len(m) else None,
                "tiefer_als_0.5R_pct": round(float((m > 0.5).mean() * 100), 1) if len(m) else None,
                "tiefer_als_0.67R_pct": round(float((m > 2 / 3).mean() * 100), 1) if len(m) else None}
    direct = [x for x in lose if (x["mfe_r"] or 0) < 0.2]
    return {"gewinner": depth(win), "traum_trades_top10pct": depth(top), "verlierer": depth(lose),
            "verlierer_direkt_runter_pct": round(len(direct) / len(lose) * 100, 1) if lose else None,
            "traum_trades_r": [round(x["r_multiple"], 1) for x in top]}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = {"profil": {}, "durchrechnung": []}
    keep = ("cagr_pct", "max_drawdown_pct", "profit_factor", "win_rate_pct", "trades", "avg_r")
    for sym in ("GC=F", "SI=F"):
        df = tidy(D.yahoo_daily(sym, start="2000-01-01"))
        for setup, (fast, slow, k, lag) in SETUPS.items():
            for scale in SCALES:
                strat = TightStart(fast, slow, k, lag, scale)
                p = strat.prepare(df)
                risk = FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.002)
                st = FxSimulator(strat, COSTS[sym], risk).run(p)
                st_s = FxSimulator(strat, stress(COSTS[sym]), risk).run(p)
                if scale == 1.0 and sym == "GC=F":
                    out["profil"][setup] = mae_profile(st.trades)
                    log.info("%s Profil: %s", setup, out["profil"][setup])
                m, ms = split(st, CUT), split(st_s, CUT)
                row = {"symbol": sym, "setup": setup, "stop_anteil": round(scale, 2),
                       "suche": {x: m["selection"].get(x) for x in keep},
                       "pruefung": {x: m["validation"].get(x) for x in keep},
                       "suche_stress_cagr": ms["selection"].get("cagr_pct"),
                       "pruefung_stress_cagr": ms["validation"].get("cagr_pct"),
                       "schnitt_verlust_r": round(float(np.mean([t["r_multiple"] for t in st.trades
                                                                 if t["pnl"] <= 0 and t["r_multiple"] is not None]
                                                                or [0])), 2),
                       "groesster_gewinn_r": round(max((t["r_multiple"] or 0) for t in st.trades), 1)}
                out["durchrechnung"].append(row)
                s, v = row["suche"], row["pruefung"]
                log.info("%s %-10s Stop x%.2f | Suche CAGR %+5.2f DD %6.1f Tr %4.1f PF %.2f n=%d | "
                         "Prüfung CAGR %+5.2f DD %6.1f Tr %4.1f PF %.2f n=%d", sym, setup, scale,
                         s["cagr_pct"], s["max_drawdown_pct"], s["win_rate_pct"], s["profit_factor"], s["trades"],
                         v["cagr_pct"], v["max_drawdown_pct"], v["win_rate_pct"], v["profit_factor"], v["trades"])
    (OUT / "stop_study.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
