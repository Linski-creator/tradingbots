"""Wie hätte die 4h-Version des Gold-Tagestrends in schlechten Gold-Jahren abgeschnitten?

TradingView (Basic) zeigt 4h nur ab 2023 (+42 % bis Nov 2026, Gold-Rallye).
Nachbau auf Tageskerzen ab 2000: 6 Kerzen à 4h = 1 Handelstag (rund 23 h), also
  SMA 200 (4h) ≈ SMA 33 (Tag), SMA 50 (4h) ≈ SMA 8 (Tag),
  3 × ATR(4h) ≈ 3 / √6 × ATR(Tag) ≈ 1,22 × ATR(Tag).
Eine einzige feste Übersetzung, keine Optimierung. Kontrolle: 2023-2026 sollte grob
zum TradingView-Ergebnis passen. Zum Vergleich der Tagestrend (SMA 50/200, 3 ATR).

Aufruf: python -m bots.metals.fast_proxy   (schreibt results/fast_proxy.json)
"""

from __future__ import annotations

import json
import logging
import math

import pandas as pd

from .. import data as D
from .. import indicators as ind
from ..hardcore.engine import FxRisk, FxSimulator
from .backtest import COSTS, stress, yearly
from .daily import OUT, DailyTrend, tidy

log = logging.getLogger(__name__)


class ScaledTrend(DailyTrend):
    def __init__(self, fast: int, slow: int, k: float):
        super().__init__("sma", shorts=False, flat_weekend=False)
        self.fast, self.slow, self.k = fast, slow, k

    def prepare(self, df):
        out = super().prepare(df)
        c = out["close"]
        f, s = ind.sma(c, self.fast), ind.sma(c, self.slow)
        out["s200"] = s
        out["long"] = ((c > s) & (f > s)).fillna(False).astype(bool)
        out["exit_long"] = (c < s).fillna(False).astype(bool)
        return out

    def on_bar(self, i, a, pos):
        acts = super().on_bar(i, a, pos)
        atr = a["atr"][i]
        out = []
        for act in acts:
            if act[0] == "stop":   # 3 ATR -> k ATR nachziehen
                e = int(pd.Index(a["ts_ns"]).searchsorted(pd.Timestamp(pos["entry_time"]).value))
                best = float(max(a["close"][e:i + 1]))
                act = ("stop", best - self.k * atr)
            elif act[0] == "order":
                od = dict(act[1])
                od["sl"] = float(a["close"][i] - self.k * atr)
                act = ("order", od)
            out.append(act)
        return out


def run(df, strat, costs):
    p = strat.prepare(df)
    return FxSimulator(strat, costs, FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.002)).run(p)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = tidy(D.yahoo_daily("GC=F", start="2000-01-01"))
    out = {}
    for name, strat in (("4h_nachbau", ScaledTrend(8, 33, 3 / math.sqrt(6))),
                        ("tagestrend", ScaledTrend(50, 200, 3.0))):
        for cname, costs in (("normal", COSTS["GC=F"]), ("stress", stress(COSTS["GC=F"]))):
            st = run(df, strat, costs)
            y = yearly(st.equity)
            eq = pd.Series([e[1] for e in st.equity], index=pd.to_datetime([e[0] for e in st.equity]))
            dd = float((eq / eq.cummax() - 1).min() * 100)
            def span(a, b):
                s = eq[(eq.index >= a) & (eq.index < b)]
                return round((s.iloc[-1] / s.iloc[0] - 1) * 100, 1) if len(s) > 1 else None
            out[f"{name}_{cname}"] = {
                "yearly": y, "max_dd_pct": round(dd, 1), "trades": len(st.trades),
                "2000-2011": span("2000-01-01", "2011-09-01"), "2011-2019 (Gold fällt/seitwärts)": span("2011-09-01", "2019-06-01"),
                "2019-2023": span("2019-06-01", "2023-01-01"), "2023-2026 (TV-Kontrolle)": span("2023-01-01", "2027-01-01"),
                "gesamt": round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1)}
            log.info("%s %s %s", name, cname, {k: v for k, v in out[f'{name}_{cname}'].items() if k != 'yearly'})
    c = df["close"]
    out["gold_halten"] = {"2011-2019": round((c[c.index >= "2019-06-01"].iloc[0] / c[c.index >= "2011-09-01"].iloc[0] - 1) * 100, 1),
                          "2023-2026": round((c.iloc[-1] / c[c.index >= "2023-01-01"].iloc[0] - 1) * 100, 1)}
    (OUT / "fast_proxy.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
