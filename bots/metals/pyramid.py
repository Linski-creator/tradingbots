"""Gewinner ausbauen (Pyramide): im Gewinn nachkaufen, nie im Verlust.

Regel (vorab festgelegt): Ein Nachkauf ist erlaubt, wenn
  - der Stop der GESAMTEN Position schon auf oder über dem Durchschnitts-Einstieg liegt
    (alles, was man schon hat, ist risikofrei), und
  - der Kurs mindestens 1 R über dem letzten Kauf schließt (R = Abstand erster Einstieg - erster Stop).
Der Nachkauf riskiert bis zum gemeinsamen Stop wieder höchstens 1 % des Kontos.
Damit liegt das offene Risiko nie über 1 %. Höchstens 0 / 1 / 2 / 3 Nachkäufe.
Grundlage: Standard Basis + B, Start-Stop 3 ATR bzw. 2 ATR (Idee E).

Aufruf: python -m bots.metals.pyramid   (schreibt results/pyramid.json)
"""

from __future__ import annotations

import json
import logging

import numpy as np

from .. import data as D
from ..hardcore.engine import FxRisk, FxSimulator
from ..hardcore.search import split
from .backtest import COSTS, stress
from .daily import OUT, tidy
from .stop_study import TightStart
from .variants import SETUPS

log = logging.getLogger(__name__)
CUT = "2016-01-01"


class Pyramid(TightStart):
    def __init__(self, fast, slow, k, lag, scale: float, max_adds: int):
        super().__init__(fast, slow, k, lag, scale)
        self.max_adds = max_adds

    def on_bar(self, i, a, pos):
        acts = super().on_bar(i, a, pos)
        if pos is None or pos.get("adds", 0) >= self.max_adds or any(x[0] == "exit" for x in acts):
            return acts
        r0 = pos["first_entry"] - pos["initial_sl"]
        last = pos.get("last_add") or pos["first_entry"]
        if pos["sl"] >= pos["entry_price"] and a["close"][i] >= last + r0:
            acts = list(acts) + [("add", None)]
        return acts


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keep = ("cagr_pct", "max_drawdown_pct", "profit_factor", "win_rate_pct", "trades", "avg_r")
    rows = []
    for sym in ("GC=F", "SI=F"):
        df = tidy(D.yahoo_daily(sym, start="2000-01-01"))
        for setup, (fast, slow, k, lag) in SETUPS.items():
            for scale in (1.0, 2 / 3):
                for adds in (0, 1, 2, 3):
                    strat = Pyramid(fast, slow, k, lag, scale, adds)
                    p = strat.prepare(df)
                    risk = FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.002)
                    st = FxSimulator(strat, COSTS[sym], risk).run(p)
                    st_s = FxSimulator(strat, stress(COSTS[sym]), risk).run(p)
                    m, ms = split(st, CUT), split(st_s, CUT)
                    rs = sorted((t["r_multiple"] or 0) for t in st.trades)
                    row = {"symbol": sym, "setup": setup, "stop_anteil": round(scale, 2), "nachkaeufe_max": adds,
                           "suche": {x: m["selection"].get(x) for x in keep},
                           "pruefung": {x: m["validation"].get(x) for x in keep},
                           "suche_stress_cagr": ms["selection"].get("cagr_pct"),
                           "pruefung_stress_cagr": ms["validation"].get("cagr_pct"),
                           "trades_mit_nachkauf": sum(1 for t in st.trades if t.get("adds")),
                           "top5_r": [round(x, 1) for x in rs[-5:]],
                           "schlimmster_r": round(rs[0], 2) if rs else None}
                    rows.append(row)
                    s, v = row["suche"], row["pruefung"]
                    log.info("%s %-10s Stop x%.2f Nachkauf %d | Suche CAGR %+5.2f DD %6.1f PF %.2f | "
                             "Prüfung CAGR %+5.2f DD %6.1f PF %.2f | top %s", sym, setup, scale, adds,
                             s["cagr_pct"], s["max_drawdown_pct"], s["profit_factor"],
                             v["cagr_pct"], v["max_drawdown_pct"], v["profit_factor"], row["top5_r"])
    (OUT / "pyramid.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
