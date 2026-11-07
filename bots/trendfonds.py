"""Trendfolge über viele Märkte gleichzeitig (Ansatz der großen Trendfonds).

Daten: börsengehandelte Fonds (ETF) mit dividendenbereinigten Kursen (Yahoo "adjclose"),
damit keine künstlichen Sprünge durch Kontraktwechsel entstehen wie bei Endlos-Futures.
Märkte: Aktien (USA, Europa/Japan, Schwellenländer), Anleihen, Gold, Silber, Öl,
Rohstoffe, Agrar, Dollar, Euro, Yen, Immobilien.

Regeln (VORAB festgelegt nach der Fachliteratur, z. B. Hurst/Ooi/Pedersen
"A Century of Evidence on Trend-Following Investing" - keine Optimierung):
  Signal je Markt: Durchschnitt aus Vorzeichen der Rendite über 1, 3 und 12 Monate (-1 bis +1).
  Größe:  gleiches Risiko je Markt; Schwankung (60 Tage) wird auf 10 % p. a. für das
          Gesamtdepot skaliert (Diversifikationsfaktor fest 2,5), Hebel höchstens 4.
  Handel: wöchentlich (Freitag-Schluss berechnet, ab dem nächsten Handelstag gültig).
  Kosten: 0,05 % je umgeschichtetem Betrag (Stress: 0,15 %), dazu 2 % p. a. Finanzierung
          auf den Teil über 100 % Investitionsgrad.
Varianten: long+short und nur long.
Suche bis Ende 2015, Prüfung ab 2016.

Aufruf: python -m bots.trendfonds   (schreibt results/trendfonds.json)
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[1] / "results"
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "SLV", "USO", "DBC",
            "DBA", "UUP", "FXE", "FXY", "VNQ"]
CUT = "2016-01-01"
LOOKBACKS = (21, 63, 252)
TARGET_VOL, DIVERSIFICATION, MAX_GROSS = 0.10, 2.5, 4.0
COST, COST_STRESS, FINANCING = 0.0005, 0.0015, 0.02


def adj_close(ticker: str, start: str = "2000-01-01") -> pd.Series:
    path = CACHE / f"adj_{ticker}.csv"
    if path.exists() and time.time() - path.stat().st_mtime < 12 * 3600:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return s
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    r = D._get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
               {"period1": p1, "period2": int(time.time()), "interval": "1d"})
    res = r.json()["chart"]["result"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    s = pd.Series(adj, index=pd.to_datetime(res["timestamp"], unit="s").normalize(), name=ticker,
                  dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")]
    CACHE.mkdir(parents=True, exist_ok=True)
    s.to_csv(path)
    return s


def signals(prices: pd.DataFrame, long_only: bool = False) -> pd.DataFrame:
    sig = sum(np.sign(prices / prices.shift(n) - 1) for n in LOOKBACKS) / len(LOOKBACKS)
    sig = sig.where(prices.shift(max(LOOKBACKS)).notna())
    return sig.clip(lower=0) if long_only else sig


def weights(prices: pd.DataFrame, long_only: bool = False) -> pd.DataFrame:
    rets = prices.pct_change()
    vol = rets.rolling(60, min_periods=40).std() * math.sqrt(252)
    sig = signals(prices, long_only)
    n = sig.notna().sum(axis=1).replace(0, np.nan)
    w = sig.mul(TARGET_VOL * DIVERSIFICATION, axis=0).div(vol).div(n, axis=0)
    gross = w.abs().sum(axis=1)
    w = w.mul((MAX_GROSS / gross).clip(upper=1), axis=0).fillna(0.0)
    # nur freitags neu festlegen, dazwischen halten
    fri = w.index.dayofweek == 4
    w = w.where(pd.Series(fri, index=w.index), np.nan).ffill().fillna(0.0)
    return w


def backtest(prices: pd.DataFrame, long_only: bool = False, cost: float = COST) -> pd.Series:
    w = weights(prices, long_only)
    rets = prices.pct_change().fillna(0.0)
    held = w.shift(1).fillna(0.0)            # Gewicht von gestern bestimmt heutige Rendite
    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    days = pd.Series(np.r_[1, np.diff(held.index.values).astype("timedelta64[D]").astype(int)], index=held.index)
    lev = held.abs().sum(axis=1)
    fin = (lev - 1).clip(lower=0) * FINANCING * days / 365
    net = gross - turnover * cost - fin
    return net


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 50:
        return {}
    eq = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * math.sqrt(252)
    dd = float((eq / eq.cummax() - 1).min())
    monthly = (1 + r).resample("ME").prod() - 1
    return {"cagr_pct": round(cagr * 100, 2), "vol_pct": round(vol * 100, 2),
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else 0,
            "max_drawdown_pct": round(dd * 100, 2), "calmar": round(cagr / abs(dd), 2) if dd < 0 else None,
            "positive_months_pct": round(float((monthly > 0).mean() * 100), 1),
            "worst_month_pct": round(float(monthly.min() * 100), 2), "years": round(years, 1)}


def yearly(r: pd.Series) -> dict:
    return {str(k.year): round(float(v) * 100, 1) for k, v in ((1 + r).resample("YE").prod() - 1).items()}


def periods(r: pd.Series) -> dict:
    return {"gesamt": stats(r), "suche": stats(r[r.index < CUT]), "pruefung": stats(r[r.index >= CUT])}


def load_universe() -> pd.DataFrame:
    cols = {}
    for t in UNIVERSE:
        try:
            cols[t] = adj_close(t)
            log.info("%s: %d Tage ab %s", t, len(cols[t]), cols[t].index[0].date())
        except Exception as e:  # noqa: BLE001
            log.warning("%s: keine Daten (%s)", t, e)
    px = pd.DataFrame(cols).sort_index()
    return px.ffill(limit=5)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    px = load_universe()
    px = px[px.index >= "2004-01-01"]
    out = {"universe": list(px.columns), "runs": {}}
    for name, lo in (("long_short", False), ("nur_long", True)):
        r = backtest(px, lo)
        rs = backtest(px, lo, COST_STRESS)
        out["runs"][name] = {"normal": periods(r), "stress": periods(rs), "yearly": yearly(r)}
        g = out["runs"][name]["normal"]
        log.info("%-10s Suche %s | Prüfung %s | Stress-Prüfung %s", name, g["suche"], g["pruefung"],
                 out["runs"][name]["stress"]["pruefung"])
    # Vergleich: gleich gewichtet kaufen und halten, und nur Aktien (SPY)
    bh = px.pct_change().mean(axis=1).fillna(0.0)
    out["vergleich"] = {"gleichgewicht_halten": periods(bh), "spy": periods(px["SPY"].pct_change().fillna(0.0)),
                        "gld": periods(px["GLD"].pct_change().fillna(0.0))}
    # Beitrag je Markt (long+short, normal)
    w = weights(px).shift(1).fillna(0.0)
    contrib = (w * px.pct_change().fillna(0.0)).sum()
    out["beitrag_je_markt_pct"] = {k: round(float(v) * 100, 1) for k, v in contrib.sort_values().items()}
    (OUT / "trendfonds.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
