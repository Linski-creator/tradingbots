"""Woche 1: Warum sind die roten Trades rot?

Nur Gold, nur 2000-2015 (Suchzeitraum). Silber und 2016+ bleiben unberührt.
Für jeden Trade werden Merkmale zum Zeitpunkt des Signals gemessen und dann
gezeigt, in welchen Gruppen sich die Verlierer häufen:
  - Wiedereinstieg: wie viele Tage seit dem letzten Ausstieg, war der ein Stop?
  - Abstand zur Trendlinie (in ATR), Steigung der Trendlinie
  - Trendalter (seit wann gilt der Aufwärtstrend)
  - neues Hoch? (Schluss = höchster Schluss der letzten 20 Kerzen)
  - ADX (Trendstärke), Schwankung im Vergleich zum Jahr
  - wie weit war der Trade im Plus (MFE), bevor er rot wurde?

Aufruf: python -m bots.metals.anatomy   (schreibt results/anatomy.json)
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pandas as pd

from .. import data as D
from .. import indicators as ind
from .backtest import COSTS
from .daily import OUT, tidy
from .fast_proxy import ScaledTrend, run

log = logging.getLogger(__name__)
END = "2016-01-01"
SETUPS = {"tageschart": (50, 200, 3.0), "4h_nachbau": (8, 33, 3 / math.sqrt(6))}


def features(p: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    c = p["close"]
    s = ind.sma(c, slow)
    atr = p["atr"]
    f = pd.DataFrame(index=p.index)
    f["abstand_atr"] = (c - s) / atr
    k = max(slow // 10, 3)
    f["steigung_atr"] = (s - s.shift(k)) / atr
    up = (c > s) & (ind.sma(c, fast) > s)
    run_id = (~up).cumsum()
    f["trendalter"] = up.groupby(run_id).cumsum()
    f["neues_hoch"] = (c >= c.rolling(20).max()).astype(int)
    f["adx"] = ind.adx(p, 14)
    vol = atr / c
    f["schwankung_rang"] = vol.rolling(250, min_periods=100).rank(pct=True)
    return f


def buckets(df: pd.DataFrame, col: str, q: int = 3) -> list[dict]:
    d = df.dropna(subset=[col])
    if d[col].nunique() <= 4:
        groups = d.groupby(col)
    else:
        groups = d.groupby(pd.qcut(d[col], q, duplicates="drop"), observed=True)
    out = []
    for k, g in groups:
        out.append({"gruppe": str(k), "trades": len(g), "treffer_pct": round(float((g.r > 0).mean() * 100), 1),
                    "summe_r": round(float(g.r.sum()), 1), "schnitt_r": round(float(g.r.mean()), 2)})
    return out


def analyse(df: pd.DataFrame, name: str, fast: int, slow: int, k: float) -> dict:
    strat = ScaledTrend(fast, slow, k)
    p = strat.prepare(df)
    st = run(df, strat, COSTS["GC=F"])
    f = features(p, fast, slow)
    idx = p.index
    rows, last_exit, last_reason = [], None, None
    for t in st.trades:
        e = pd.Timestamp(t["entry_time"])
        if e >= pd.Timestamp(END, tz=e.tz):
            break
        j = int(idx.searchsorted(e)) - 1          # Signal-Kerze = Kerze vor dem Einstieg
        row = f.iloc[j].to_dict()
        row["r"] = t["r_multiple"]
        row["mfe_r"] = t["mfe_r"]
        row["grund"] = t["reason"]
        row["tage_seit_ausstieg"] = (e - last_exit).days if last_exit is not None else 999
        row["nach_stop"] = int(last_reason == "stop")
        row["wiedereinstieg_5t"] = int(row["tage_seit_ausstieg"] <= 5)
        row["tage_gehalten"] = (pd.Timestamp(t["exit_time"]) - e).days
        rows.append(row)
        last_exit, last_reason = pd.Timestamp(t["exit_time"]), t["reason"]
    d = pd.DataFrame(rows)
    losers = d[d.r <= 0]
    out = {
        "trades": len(d), "treffer_pct": round(float((d.r > 0).mean() * 100), 1),
        "summe_r": round(float(d.r.sum()), 1),
        "groesster_gewinn_r": round(float(d.r.max()), 1),
        "anteil_gewinn_aus_top10pct": round(float(d.r.nlargest(max(len(d) // 10, 1)).sum() / d.r[d.r > 0].sum() * 100), 1),
        "verlierer": {
            "anzahl": len(losers), "schnitt_r": round(float(losers.r.mean()), 2),
            "vorher_mind_0_5R_im_plus_pct": round(float((losers.mfe_r >= 0.5).mean() * 100), 1),
            "vorher_mind_1R_im_plus_pct": round(float((losers.mfe_r >= 1).mean() * 100), 1),
            "median_tage_gehalten": float(losers.tage_gehalten.median()),
            "ausstieg": losers.grund.value_counts().to_dict(),
        },
        "gewinner_median_tage": float(d[d.r > 0].tage_gehalten.median()),
        "nach_merkmal": {c: buckets(d, c) for c in ("wiedereinstieg_5t", "nach_stop", "abstand_atr", "steigung_atr",
                                                    "trendalter", "neues_hoch", "adx", "schwankung_rang")},
    }
    log.info("%s: %d Trades, %.0f %% Treffer, Summe %.1f R", name, out["trades"], out["treffer_pct"], out["summe_r"])
    for c, b in out["nach_merkmal"].items():
        log.info("  %-18s %s", c, " | ".join(f"{x['gruppe']}: n={x['trades']} {x['treffer_pct']}% {x['summe_r']:+}R" for x in b))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = tidy(D.yahoo_daily("GC=F", start="2000-01-01"))
    res = {n: analyse(df, n, *cfg) for n, cfg in SETUPS.items()}
    (OUT / "anatomy.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
