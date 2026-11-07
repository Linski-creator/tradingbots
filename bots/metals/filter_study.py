"""Filter gegen Sorte-1-Trades (aus results/sorte1.json abgeleitet, vorab festgelegt):

  F  nur kaufen, wenn der Kurs über der schnellen Linie (50er) schließt
  H  nur Signale aus Kerzen, die zwischen 12 und 24 Uhr UTC beginnen (US-Handelszeit + Abend);
     keine Signale aus der Asien- und Europa-Vormittagszeit (0-12 Uhr UTC)

Grundlage: aktueller Stand (Basis + B, Start-Stop 2 ATR, Nachzieh-Stop 3 ATR, Position max. 1x Konto).
Geprüft auf:
  1. Tageschart und 4h-Nachbau ab 2000 (nur F), Gold und Silber, Suche/Prüfung, Stress
  2. echte Gold-Kerzen 3h/4h (Yahoo, ca. 2 Jahre): F, H, F+H
  3. UNABHÄNGIG: Gold-Token PAXG 3h/4h, 2020 bis Mitte 2024 (vor den Yahoo-Daten): F, H, F+H

Aufruf: python -m bots.metals.filter_study   (schreibt results/filter_study.json)
"""

from __future__ import annotations

import json
import logging
import math

import pandas as pd

from .. import data as D
from .. import indicators as ind
from ..hardcore.engine import FxRisk, FxSimulator
from ..hardcore.search import split
from ..metrics import summarize
from .backtest import COSTS, stress
from .daily import OUT, tidy
from .sorte1 import resample
from .stop_study import TightStart
from .variants import SETUPS

log = logging.getLogger(__name__)
CUT = "2016-01-01"
KEEP = ("cagr_pct", "max_drawdown_pct", "profit_factor", "win_rate_pct", "trades")


class Filtered(TightStart):
    def __init__(self, fast, slow, k, lag, scale, flags: str):
        super().__init__(fast, slow, k, lag, scale)
        self.extra = flags

    def prepare(self, df):
        out = super().prepare(df)
        ok = out["long"]
        if "F" in self.extra:
            ok &= out["close"] > ind.sma(out["close"], self.fast)
        if "H" in self.extra:
            ok &= pd.Series(out.index.hour >= 12, index=out.index)
        out["long"] = ok.fillna(False).astype(bool)
        return out


def full_losses_per_year(trades, eq):
    if not eq:
        return 0
    years = max((pd.Timestamp(eq[-1][0]) - pd.Timestamp(eq[0][0])).days / 365.25, 0.1)
    return round(sum(1 for t in trades if (t["r_multiple"] or 0) < -0.8) / years, 1)


def run(df, strat, costs):
    p = strat.prepare(df)
    return FxSimulator(strat, costs, FxRisk(risk_per_trade=0.01, max_leverage=1, min_stop=0.0005)).run(p)


def short(m):
    return {k: (round(m.get(k), 2) if isinstance(m.get(k), float) else m.get(k)) for k in KEEP}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = {"lang": [], "echt": [], "paxg": []}
    for sym in ("GC=F", "SI=F"):
        df = tidy(D.yahoo_daily(sym, start="2000-01-01"))
        for setup, (fast, slow, k, lag) in SETUPS.items():
            for flags in ("", "F"):
                strat = Filtered(fast, slow, k, lag, 2 / 3, flags)
                st, st_s = run(df, strat, COSTS[sym]), run(df, strat, stress(COSTS[sym]))
                m, ms = split(st, CUT), split(st_s, CUT)
                row = {"symbol": sym, "setup": setup, "filter": flags or "ohne",
                       "suche": short(m["selection"]), "pruefung": short(m["validation"]),
                       "stress_cagr": [ms["selection"].get("cagr_pct"), ms["validation"].get("cagr_pct")],
                       "volle_verluste_pro_jahr": full_losses_per_year(st.trades, st.equity)}
                out["lang"].append(row)
                log.info("LANG %s %-10s %-4s | S %s | P %s | Stress %s | volle Verl./J %s", sym, setup, row["filter"],
                         row["suche"], row["pruefung"], row["stress_cagr"], row["volle_verluste_pro_jahr"])
    sets = {"echt": tidy(D.yahoo_hourly("GC=F")),
            "paxg": tidy(D.binance_vision_klines("PAXGUSDT", "1h", start="2020-01"))}
    sets["paxg"] = sets["paxg"][sets["paxg"].index < sets["echt"].index[0]]
    for name, h in sets.items():
        costs = COSTS["GC=F"] if name == "echt" else COSTS["PAXGUSDT"]
        for rule in ("3h", "4h"):
            df = tidy(resample(h, rule))
            for flags in ("", "F", "H", "FH"):
                st = run(df, Filtered(50, 200, 3.0, 20, 2 / 3, flags), costs)
                st_s = run(df, Filtered(50, 200, 3.0, 20, 2 / 3, flags), stress(costs))
                m = summarize(st.equity, st.trades, 1000.0)
                row = {"daten": name, "kerzen": rule, "filter": flags or "ohne", "zeitraum": [str(df.index[0].date()), str(df.index[-1].date())],
                       "ergebnis": short(m), "stress_cagr": summarize(st_s.equity, st_s.trades, 1000.0).get("cagr_pct"),
                       "volle_verluste_pro_jahr": full_losses_per_year(st.trades, st.equity)}
                out[name].append(row)
                log.info("%s %s %-4s | %s | Stress %.2f | volle Verl./J %s", name.upper(), rule, row["filter"], row["ergebnis"],
                         row["stress_cagr"] or 0, row["volle_verluste_pro_jahr"])
    (OUT / "filter_study.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
