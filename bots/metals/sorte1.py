"""Sorte-1-Trades: Was haben die Trades gemeinsam, die direkt nach dem Kauf nach unten laufen?

Sorte 1 = Verlust-Trade, der nie nennenswert im Plus war (MFE < 0,3 R).
Sorte 2 = war mind. 1 R im Plus, endete aber ohne großen Gewinn (<= +0,5 R).
Sorte 3 = Gewinn-Trade über +0,5 R.

Regeln = aktueller Stand (Basis + B, Start-Stop 2 ATR, Nachzieh-Stop 3 ATR, nur Long).
Daten:
  1. echte Gold-Kerzen (Yahoo 1h, ca. 2 Jahre) zu 3h und 4h zusammengefasst, Regeln 1:1 wie TradingView
  2. 4h-Nachbau auf Tageskerzen, NUR 2000-2015 (Suchzeitraum; 2016+ bleibt für die Prüfung)
Für jedes Merkmal zum Zeitpunkt des Signals: Anteil Sorte 1 und durchschnittliches R je Gruppe.
Ein Filter kommt nur in Frage, wenn er in BEIDEN Datensätzen in dieselbe Richtung zeigt.

Aufruf: python -m bots.metals.sorte1   (schreibt results/sorte1.json)
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
from .backtest import COSTS
from .daily import OUT, tidy
from .stop_study import TightStart

log = logging.getLogger(__name__)
END = "2016-01-01"


def resample(h: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = h.resample(rule, label="left", closed="left", origin="epoch")
    return g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def sorte(t: dict) -> int:
    mfe = t.get("mfe_r") or 0
    r = t["r_multiple"] or 0
    if r < 0 and mfe < 0.3:
        return 1
    if mfe >= 1 and r <= 0.5:
        return 2
    return 3 if r > 0.5 else 0


def features(p: pd.DataFrame, slow: int, fast: int) -> pd.DataFrame:
    c, h, l, o = p["close"], p["high"], p["low"], p["open"]
    atr = p["atr"]
    s, f = ind.sma(c, slow), ind.sma(c, fast)
    out = pd.DataFrame(index=p.index)
    out["abstand_200er_atr"] = (c - s) / atr
    out["abstand_50er_atr"] = (c - f) / atr
    out["steigung_200er_atr"] = (s - s.shift(20)) / atr
    out["rsi"] = ind.rsi(c, 14)
    out["adx"] = ind.adx(p, 14)
    out["schwankung_rang"] = (atr / c).rolling(250, min_periods=100).rank(pct=True)
    out["kerze_groesse_atr"] = (h - l) / atr
    out["schluss_in_kerze"] = ((c - l) / (h - l)).where(h > l)
    up = c > c.shift(1)
    out["gruene_kerzen_folge"] = up.groupby((~up).cumsum()).cumsum()
    out["abstand_20er_hoch_atr"] = (h.rolling(20).max() - c) / atr
    out["wochentag"] = p.index.dayofweek
    out["stunde_utc"] = p.index.hour
    return out


def analyse(df: pd.DataFrame, fast: int, slow: int, k: float, lag: int, scale: float, until: str | None) -> dict:
    strat = TightStart(fast, slow, k, lag, scale)
    p = strat.prepare(df)
    st = FxSimulator(strat, COSTS["GC=F"], FxRisk(risk_per_trade=0.01, max_leverage=1, min_stop=0.0005)).run(p)
    f = features(p, slow, fast)
    rows, prev = [], None
    for t in st.trades:
        if until and t["entry_time"] >= until:
            break
        j = int(p.index.searchsorted(pd.Timestamp(t["entry_time"]))) - 1
        row = f.iloc[j].to_dict()
        row["vorher_verlust"] = int(prev is not None and prev < 0)
        row["sorte"] = sorte(t)
        row["r"] = t["r_multiple"] or 0
        rows.append(row)
        prev = row["r"]
    d = pd.DataFrame(rows)
    res = {"trades": len(d), "anteile": {f"sorte_{s}": round(float((d.sorte == s).mean() * 100), 1) for s in (0, 1, 2, 3)},
           "summe_r": round(float(d.r.sum()), 1), "merkmale": {}}
    for col in [c for c in d.columns if c not in ("sorte", "r")]:
        x = d.dropna(subset=[col])
        if x.empty:
            continue
        if col in ("wochentag", "vorher_verlust"):
            groups = x.groupby(col)
        elif col == "stunde_utc":
            if x[col].nunique() <= 1:
                continue
            groups = x.groupby(pd.cut(x[col], [-1, 5, 11, 17, 23], labels=["0-5", "6-11", "12-17", "18-23"]),
                               observed=True)
        else:
            groups = x.groupby(pd.qcut(x[col], 3, duplicates="drop"), observed=True)
        res["merkmale"][col] = [{"gruppe": str(g), "n": len(v), "sorte1_pct": round(float((v.sorte == 1).mean() * 100), 1),
                                 "schnitt_r": round(float(v.r.mean()), 2), "summe_r": round(float(v.r.sum()), 1)}
                                for g, v in groups]
    return res


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = {}
    h = D.yahoo_hourly("GC=F")
    for rule in ("3h", "4h"):
        df = tidy(resample(h, rule))
        out[f"echt_{rule}"] = analyse(df, 50, 200, 3.0, 20, 2 / 3, None)
        out[f"echt_{rule}"]["zeitraum"] = [str(df.index[0]), str(df.index[-1])]
    d = tidy(D.yahoo_daily("GC=F", start="2000-01-01"))
    out["nachbau_4h_2000_2015"] = analyse(d, 8, 33, 3 / math.sqrt(6), 3, 2 / 3, END)
    for name, r in out.items():
        log.info("== %s: %d Trades, Anteile %s, Summe %s R", name, r["trades"], r["anteile"], r["summe_r"])
        for col, b in r["merkmale"].items():
            log.info("  %-22s %s", col, " | ".join(f"{x['gruppe']}: n={x['n']} S1 {x['sorte1_pct']}% {x['schnitt_r']:+}R"
                                                  for x in b))
    (OUT / "sorte1.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
