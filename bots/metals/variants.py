"""Woche 2: Reparaturen A-D (und Wochenende flach) einzeln und kombiniert, Gold ab 2000.

Gleiche Schalter wie im TradingView-Skript tradingview/gold_tagestrend.pine (v2):
  A  Rücksetzer abwarten: Tief der letzten 5 Kerzen <= schnelle Linie und Schluss > Vorkerzen-Hoch
  B  langsame Linie höher als vor 20 Kerzen (4h-Nachbau: 3 Tage ≈ 20 × 4h)
  C  Stop auf Einstieg, sobald ein Hoch +1 R erreicht
  D  Stop folgt dem höchsten Hoch statt dem höchsten Schluss
  W  Wochenende flach (Donnerstag-Schluss -> Freitag-Open raus, nur Tageschart sinnvoll)
Ausgewertet: Suche 2000-2015 und Prüfung 2016+ getrennt, normal und Stress-Kosten.

Aufruf: python -m bots.metals.variants   (schreibt results/variants.json)
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pandas as pd

from .. import data as D
from .. import indicators as ind
from ..hardcore.engine import FxRisk, FxSimulator
from ..hardcore.search import split
from .backtest import COSTS, stress
from .daily import OUT, DailyTrend, tidy

log = logging.getLogger(__name__)
CUT = "2016-01-01"
SETUPS = {"tageschart": (50, 200, 3.0, 20), "4h_nachbau": (8, 33, 3 / math.sqrt(6), 3)}
COMBOS = ["", "A", "B", "C", "D", "W", "AB", "CD", "ABC", "ABD", "ABCD", "BW", "ABW"]


class Variant(DailyTrend):
    def __init__(self, fast: int, slow: int, k: float, lag: int, flags: str = ""):
        super().__init__("sma", shorts=False, flat_weekend="W" in flags)
        self.fast, self.slow, self.k, self.lag, self.flags = fast, slow, k, lag, flags

    def prepare(self, df):
        out = super().prepare(df)
        c, l, h = out["close"], out["low"], out["high"]
        f, s = ind.sma(c, self.fast), ind.sma(c, self.slow)
        out["s200"] = s
        ok = (c > s) & (f > s)
        if "A" in self.flags:
            ok &= (l.rolling(5).min() <= f) & (c > h.shift(1))
        if "B" in self.flags:
            ok &= s > s.shift(self.lag)
        out["long"] = ok.fillna(False).astype(bool)
        out["exit_long"] = (c < s).fillna(False).astype(bool)
        return out

    def on_bar(self, i, a, pos):
        c, atr = a["close"][i], a["atr"][i]
        if not (math.isfinite(atr) and atr > 0):
            return []
        thursday = a["dow"][i] == 3
        if pos is not None:
            if self.flat and thursday:
                return [("exit", "wochenende")]
            if a["exit_long"][i]:
                return [("exit", "trendbruch")]
            e = int(np.searchsorted(a["ts_ns"], pd.Timestamp(pos["entry_time"]).value))
            best = float(np.max(a["high" if "D" in self.flags else "close"][e:i + 1]))
            stop = best - self.k * atr
            if "C" in self.flags:
                entry, risk = pos["entry_price"], abs(pos["entry_price"] - pos["initial_sl"])
                if float(np.max(a["high"][e:i + 1])) - entry >= risk:
                    stop = max(stop, entry)
            return [("stop", stop)]
        if (self.flat and thursday) or not a["long"][i]:
            return []
        return [("order", {"side": 1, "type": "market", "sl": float(c - self.k * atr), "tag": "variante",
                           "context": {"close": float(c), "sma200": float(a["s200"][i]), "atr": float(atr)}})]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = tidy(D.yahoo_daily("GC=F", start="2000-01-01"))
    rows = []
    for setup, (fast, slow, k, lag) in SETUPS.items():
        for flags in COMBOS:
            if "W" in flags and setup != "tageschart":
                continue
            strat = Variant(fast, slow, k, lag, flags)
            p = strat.prepare(df)
            risk = FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.002)
            st = FxSimulator(strat, COSTS["GC=F"], risk).run(p)
            st_s = FxSimulator(strat, stress(COSTS["GC=F"]), risk).run(p)
            m, ms = split(st, CUT), split(st_s, CUT)
            keep = ("cagr_pct", "max_drawdown_pct", "profit_factor", "win_rate_pct", "trades", "avg_r")
            row = {"setup": setup, "variante": flags or "basis",
                   "suche": {k2: m["selection"].get(k2) for k2 in keep},
                   "pruefung": {k2: m["validation"].get(k2) for k2 in keep},
                   "suche_stress_cagr": ms["selection"].get("cagr_pct"),
                   "pruefung_stress_cagr": ms["validation"].get("cagr_pct")}
            rows.append(row)
            s, v = row["suche"], row["pruefung"]
            log.info("%-10s %-5s | Suche CAGR %+5.2f DD %6.1f Treffer %4.1f PF %.2f n=%d | Prüfung CAGR %+5.2f DD %6.1f "
                     "Treffer %4.1f PF %.2f n=%d", setup, row["variante"], s["cagr_pct"], s["max_drawdown_pct"],
                     s["win_rate_pct"], s["profit_factor"], s["trades"], v["cagr_pct"], v["max_drawdown_pct"],
                     v["win_rate_pct"], v["profit_factor"], v["trades"])
    (OUT / "variants.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
