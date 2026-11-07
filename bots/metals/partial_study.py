"""Teilgewinn (T): bei +1 R die Hälfte verkaufen, Rest mit Stop auf Einstieg weiterlaufen lassen.

Ziel des Nutzers: auch kleinere Bewegungen mitnehmen und weniger volle Verlust-Trades (-1 R).
Grundlage: Standard Basis + B, Start-Stop 2 ATR (E), Crash-Schutz Position max. 1x Konto.
Verglichen: ohne / mit Teilgewinn, Tageschart und 4h-Nachbau, Gold und Silber,
Suche bis 2015 und Prüfung ab 2016, normal und Stress-Kosten.
Zusätzlich gezählt: volle Verlust-Trades (schlechter als -0,8 R) pro Jahr.

Aufruf: python -m bots.metals.partial_study   (schreibt results/partial_study.json)
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from .. import data as D
from ..hardcore.engine import FxRisk, FxSimulator
from ..hardcore.search import split
from .backtest import COSTS, stress
from .daily import OUT, tidy
from .stop_study import TightStart
from .variants import SETUPS

log = logging.getLogger(__name__)
CUT = "2016-01-01"


class WithPartial(TightStart):
    def __init__(self, fast, slow, k, lag, scale: float, partial: bool):
        super().__init__(fast, slow, k, lag, scale)
        self.partial = partial

    def on_bar(self, i, a, pos):
        acts = super().on_bar(i, a, pos)
        if not self.partial:
            return acts
        out = []
        for act in acts:
            if act[0] == "order":
                act = ("order", {**act[1], "tp1_r": 1.0, "tp1_frac": 0.5})
            out.append(act)
        return out


def per_year(trades, start, end, cond):
    t = [x for x in trades if start <= x["exit_time"] < end and cond(x)]
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    return round(len(t) / years, 1)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keep = ("cagr_pct", "max_drawdown_pct", "profit_factor", "win_rate_pct", "trades")
    rows = []
    for sym in ("GC=F", "SI=F"):
        df = tidy(D.yahoo_daily(sym, start="2000-01-01"))
        end = (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for setup, (fast, slow, k, lag) in SETUPS.items():
            for partial in (False, True):
                strat = WithPartial(fast, slow, k, lag, 2 / 3, partial)
                p = strat.prepare(df)
                risk = FxRisk(risk_per_trade=0.01, max_leverage=1, min_stop=0.002)
                st = FxSimulator(strat, COSTS[sym], risk).run(p)
                st_s = FxSimulator(strat, stress(COSTS[sym]), risk).run(p)
                m, ms = split(st, CUT), split(st_s, CUT)
                full_loss = lambda x: x["r_multiple"] is not None and x["r_multiple"] < -0.8
                row = {"symbol": sym, "setup": setup, "teilgewinn": partial,
                       "suche": {x: m["selection"].get(x) for x in keep},
                       "pruefung": {x: m["validation"].get(x) for x in keep},
                       "suche_stress_cagr": ms["selection"].get("cagr_pct"),
                       "pruefung_stress_cagr": ms["validation"].get("cagr_pct"),
                       "volle_verluste_pro_jahr": {"suche": per_year(st.trades, "2000-01-01", CUT, full_loss),
                                                   "pruefung": per_year(st.trades, CUT, end, full_loss)},
                       "gewinn_trades_pro_jahr": {"suche": per_year(st.trades, "2000-01-01", CUT, lambda x: x["pnl"] > 0),
                                                  "pruefung": per_year(st.trades, CUT, end, lambda x: x["pnl"] > 0)},
                       "groesster_r": round(max((t["r_multiple"] or 0) for t in st.trades), 1)}
                rows.append(row)
                s, v = row["suche"], row["pruefung"]
                log.info("%s %-10s Teilgewinn %-5s | Suche CAGR %+5.2f DD %6.1f Tr %4.1f PF %.2f | Prüfung CAGR %+5.2f "
                         "DD %6.1f Tr %4.1f PF %.2f | volle Verluste/J %s", sym, setup, partial, s["cagr_pct"],
                         s["max_drawdown_pct"], s["win_rate_pct"], s["profit_factor"], v["cagr_pct"],
                         v["max_drawdown_pct"], v["win_rate_pct"], v["profit_factor"], row["volle_verluste_pro_jahr"])
    (OUT / "partial_study.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
