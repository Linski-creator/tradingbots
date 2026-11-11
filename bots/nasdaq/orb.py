"""Nasdaq: Eröffnungs-Kerze / Opening Range Breakout (ORB) auf Stundenkerzen.

Daten: QQQ (Nasdaq-100-ETF), Yahoo 60-Minuten-Kerzen der regulären Handelszeit (ca. 2 Jahre).
Die erste Kerze des Tages ist 9:30-10:30 New York (15:30-16:30 deutsche Zeit).

Varianten (vorab festgelegt, keine Optimierung):
  ausbruch      Hoch/Tief der ersten Stunde = Spanne. Ausbruch über das Hoch -> Long (Stop-Order),
                unter das Tief -> Short. Stop: andere Seite der Spanne. Raus zum Tagesschluss.
                Nur der erste Ausbruch pro Tag.
  ausbruch_long wie oben, nur Long.
  kerze         erste Stunde grün -> Kauf zum Open der zweiten Stunde, rot -> Short.
                Stop: anderes Ende der ersten Kerze. Raus zum Tagesschluss.
Vorsichtige Annahmen: Liegt Einstieg und Stop in derselben Stundenkerze, zählt es als Verlust.
Kurslücke über ein Niveau -> Ausführung zum Open.
Risiko: 1 % des Kontos je Trade, Position höchstens 4x Konto. Kosten je Seite 0,02 %
(CFD-Spread + Ausführung), Stress: 0,06 %.
Auswertung: gesamter Zeitraum und beide Hälften getrennt.

Aufruf: python -m bots.nasdaq.orb   (schreibt results/orb.json)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[2] / "results"
COST, COST_STRESS = 0.0002, 0.0006


def hourly_rth(ticker: str) -> pd.DataFrame:
    r = D._get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
               {"range": "730d", "interval": "60m", "includePrePost": "false"})
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                      index=pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York"))
    df = df.dropna()
    df = df[(df.index.hour * 60 + df.index.minute >= 9 * 60 + 30) & (df.index.hour < 16)]
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


def day_trades(day: pd.DataFrame, variant: str) -> dict | None:
    if len(day) < 3 or not (day.index[0].hour == 9 and day.index[0].minute == 30):
        return None
    first, rest = day.iloc[0], day.iloc[1:]
    hi, lo = first["high"], first["low"]
    if hi <= lo:
        return None
    exit_px = rest["close"].iloc[-1]
    if variant == "kerze":
        side = 1 if first["close"] > first["open"] else -1 if first["close"] < first["open"] else 0
        if side == 0:
            return None
        entry, stop = rest["open"].iloc[0], (lo if side == 1 else hi)
        if (side == 1 and entry <= stop) or (side == -1 and entry >= stop):
            return None
        for _, b in rest.iterrows():
            if (side == 1 and b["low"] <= stop) or (side == -1 and b["high"] >= stop):
                px = min(b["open"], stop) if side == 1 else max(b["open"], stop)
                return {"side": side, "entry": entry, "stop": stop, "exit": px, "reason": "stop"}
        return {"side": side, "entry": entry, "stop": stop, "exit": exit_px, "reason": "schluss"}
    pos = None
    for _, b in rest.iterrows():
        if pos is None:
            up, dn = b["high"] > hi, b["low"] < lo
            if up and (not dn or variant == "ausbruch_long"):
                pos = {"side": 1, "entry": max(b["open"], hi), "stop": lo}
            elif dn and not up and variant == "ausbruch":
                pos = {"side": -1, "entry": min(b["open"], lo), "stop": hi}
            elif up and dn:   # beide Seiten in einer Kerze: unklar -> vorsichtig als Verlust
                return {"side": 1, "entry": hi, "stop": lo, "exit": lo, "reason": "unklar"}
            if pos is not None:
                # Einstieg und Stop in derselben Kerze -> Verlust
                if (pos["side"] == 1 and b["low"] <= pos["stop"]) or (pos["side"] == -1 and b["high"] >= pos["stop"]):
                    return {**pos, "exit": pos["stop"], "reason": "stop"}
            continue
        s = pos["side"]
        if (s == 1 and b["low"] <= pos["stop"]) or (s == -1 and b["high"] >= pos["stop"]):
            px = min(b["open"], pos["stop"]) if s == 1 else max(b["open"], pos["stop"])
            return {**pos, "exit": px, "reason": "stop"}
    if pos is None:
        return None
    return {**pos, "exit": exit_px, "reason": "schluss"}


def simulate(df: pd.DataFrame, variant: str, cost: float) -> tuple[list, pd.Series]:
    eq, trades, curve = 1000.0, [], {}
    for d, day in df.groupby(df.index.date):
        t = day_trades(day, variant)
        if t:
            dist = abs(t["entry"] - t["stop"]) / t["entry"]
            if dist > 0:
                notional = min(eq * 0.01 / (dist + 2 * cost), eq * 4)
                gross = notional * t["side"] * (t["exit"] / t["entry"] - 1)
                pnl = gross - notional * 2 * cost
                eq += pnl
                trades.append({**t, "date": str(d), "pnl": pnl, "r": pnl / (eq - pnl) / 0.01})
        curve[pd.Timestamp(d)] = eq
    return trades, pd.Series(curve)


def stats(trades: list, curve: pd.Series) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = np.array([t["pnl"] for t in trades])
    r = np.array([t["r"] for t in trades])
    dd = float((curve / curve.cummax() - 1).min() * 100)
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 0.1)
    start = curve.iloc[0] - (trades[0]["pnl"] if trades[0]["date"] == str(curve.index[0].date()) else 0)
    total = (curve.iloc[-1] / start - 1) * 100
    return {"trades": len(trades), "treffer_pct": round(float((pnl > 0).mean() * 100), 1),
            "schnitt_r": round(float(r.mean()), 3),
            "profit_factor": round(float(pnl[pnl > 0].sum() / -pnl[pnl < 0].sum()), 2) if (pnl < 0).any() else None,
            "gewinn_pct": round(float(total), 1), "pro_jahr_pct": round(float(((1 + total / 100) ** (1 / years) - 1) * 100), 1),
            "max_dd_pct": round(dd, 1), "volle_verluste": int((r < -0.8).sum()),
            "groesster_gewinn_r": round(float(r.max()), 1)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    df = hourly_rth("QQQ")
    days = sorted(set(df.index.date))
    mid = days[len(days) // 2]
    out = {"daten": "QQQ 60m reguläre Handelszeit", "zeitraum": [str(days[0]), str(days[-1])], "tage": len(days),
           "ergebnisse": []}
    for variant in ("ausbruch", "ausbruch_long", "kerze"):
        for cname, cost in (("normal", COST), ("stress", COST_STRESS)):
            tr, cv = simulate(df, variant, cost)
            row = {"variante": variant, "kosten": cname, "gesamt": stats(tr, cv),
                   "haelfte_1": stats([t for t in tr if t["date"] < str(mid)], cv[cv.index < pd.Timestamp(mid)]),
                   "haelfte_2": stats([t for t in tr if t["date"] >= str(mid)], cv[cv.index >= pd.Timestamp(mid)]),
                   "long_short_pnl": [round(sum(t["pnl"] for t in tr if t["side"] == s), 1) for s in (1, -1)]}
            out["ergebnisse"].append(row)
            log.info("%-14s %-6s gesamt %s | H1 %s | H2 %s", variant, cname, row["gesamt"], row["haelfte_1"], row["haelfte_2"])
    (OUT / "orb.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
