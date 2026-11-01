"""Historische Kurse von Dukascopy (kostenlos, Minutenkerzen ab 2003).

Pro Handelstag gibt es eine LZMA-gepackte Datei mit 1-Minuten-Kerzen
(Geldkurs/BID, Zeit in UTC). Monate in der URL beginnen bei 00.
Jeder Datensatz: Sekunden seit Tagesbeginn, Open, Close, Low, High
(ganze Zahlen in "Punkten"), Volumen (float), big-endian, 24 Bytes.
"""

from __future__ import annotations

import logging
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "dukascopy"
URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
RECORD = struct.Struct(">5if")

# Preis = Ganzzahl / Teiler; Plausibilitätsbereich gegen falsche Teiler
INSTRUMENTS = {
    "XAUUSD": {"divisor": 1000, "range": (900, 8000)},
    "XAGUSD": {"divisor": 1000, "range": (8, 200)},
    "XPTUSD": {"divisor": 1000, "range": (400, 4000)},
    "USDJPY": {"divisor": 1000, "range": (60, 250)},
}


def decode(raw: bytes, day: pd.Timestamp, divisor: float) -> pd.DataFrame:
    """Entpackt eine Tagesdatei zu 1-Minuten-Kerzen (UTC-Index)."""
    cols = ["open", "high", "low", "close", "volume"]
    if not raw:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC"))
    data = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
    n = len(data) // RECORD.size
    rows = [RECORD.unpack_from(data, i * RECORD.size) for i in range(n)]
    df = pd.DataFrame(rows, columns=["sec", "open", "close", "low", "high", "volume"])
    df.index = day + pd.to_timedelta(df.pop("sec"), unit="s")
    for c in ("open", "close", "low", "high"):
        df[c] = df[c] / divisor
    df = df[df["volume"] > 0]  # Wochenende/Feiertag: flache Kerzen ohne Umsatz
    return df[cols]


def _fetch(sym: str, day: pd.Timestamp) -> bytes:
    url = URL.format(sym=sym, y=day.year, m=day.month - 1, d=day.day)
    last = None
    for k in range(4):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "krypto-bots/1.0"})
            if r.status_code == 404:
                return b""
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** k)
    raise RuntimeError(f"Dukascopy nicht erreichbar: {url} ({last})")


def resample(df1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df1m.resample(rule, label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["open", "close"])


def month_5m(sym: str, month: pd.Period) -> pd.DataFrame:
    """5-Minuten-Kerzen eines Monats, mit Cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{sym}_{month}.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
        return df
    divisor = INSTRUMENTS[sym]["divisor"]
    days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D", tz="UTC")
    days = [d for d in days if d.weekday() != 5]  # samstags kein Handel
    with ThreadPoolExecutor(8) as ex:
        raws = list(ex.map(lambda d: _fetch(sym, d), days))
    parts = [decode(r, d, divisor) for r, d in zip(raws, days)]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = resample(pd.concat(parts).sort_index(), "5min")
    lo, hi = INSTRUMENTS[sym]["range"]
    med = df["close"].median()
    if not lo <= med <= hi:
        raise ValueError(f"{sym} {month}: Median {med} außerhalb {lo}-{hi}, falscher Teiler?")
    if month < pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M"):
        df.to_csv(path)  # laufenden Monat nicht cachen
    return df


def history(sym: str, start: str = "2012-01", interval: str = "15min") -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
    frames = []
    for m in pd.period_range(start, end, freq="M"):
        frames.append(month_5m(sym, m))
        log.info("%s %s geladen", sym, m)
    df = pd.concat([f for f in frames if not f.empty]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df if interval == "5min" else resample(df, interval)
