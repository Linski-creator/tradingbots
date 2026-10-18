"""Kursdaten laden: historisch (Backtest) und aktuell (Paper-Trading).

Quellen:
  - Binance Vision (data.binance.vision): komplette Historie als Monatsdateien
  - Binance Market-Data-API (data-api.binance.vision): aktuelle Kerzen
  - Coinbase Exchange: Ersatzquelle für aktuelle Kerzen
  - Yahoo Finance (GC=F): lange Gold-Historie für Tagesstrategien
  - Funding-Raten: Binance Vision (Historie), Binance/Kraken-Futures (aktuell)

Alle Kerzen: UTC-Index = Startzeit der Kerze, Spalten open/high/low/close/volume,
NUR abgeschlossene Kerzen.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
UA = {"User-Agent": "krypto-bots/1.0"}
INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
COINBASE_GRAN = {"1h": 3600, "1d": 86400}
COINBASE_SYMBOL = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
                   "XRPUSDT": "XRP-USD", "PAXGUSDT": "PAXG-USD"}


def _get(url: str, params: Optional[dict] = None, tries: int = 4, timeout: int = 30):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code in (418, 429) or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.RequestException, requests.HTTPError) as e:
            last = e
            time.sleep(2 ** (k + 1))
    raise RuntimeError(f"Abruf fehlgeschlagen: {url} ({last})")


def _to_ts(values: pd.Series) -> pd.DatetimeIndex:
    v = pd.to_numeric(values)
    # Binance Vision liefert seit 2025 Mikrosekunden statt Millisekunden
    unit = "us" if v.max() > 10 ** 14 else "ms"
    return pd.to_datetime(v, unit=unit, utc=True)


def clean(df: pd.DataFrame, interval_ms: Optional[int] = None) -> pd.DataFrame:
    """Sortiert, entfernt Duplikate und kaputte Kerzen. Prüft Konsistenz."""
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | df.isna().any(axis=1)
    if bad.any():
        log.warning("%d ungültige Kerzen entfernt", int(bad.sum()))
        df = df[~bad]
    # kleine Rundungsfehler in High/Low korrigieren, grobe Fehler verwerfen
    hi = df[["open", "close"]].max(axis=1)
    lo = df[["open", "close"]].min(axis=1)
    gross = (df["high"] < hi * 0.999) | (df["low"] > lo * 1.001)
    if gross.any():
        log.warning("%d inkonsistente Kerzen entfernt", int(gross.sum()))
        df = df[~gross]
        hi, lo = hi[~gross], lo[~gross]
    df["high"] = df["high"].where(df["high"] >= hi, hi)
    df["low"] = df["low"].where(df["low"] <= lo, lo)
    return df


def align(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Entfernt Kerzen, deren Startzeit nicht auf dem Raster liegt
    (Binance hatte z. B. 2018 nach Wartungen Kerzen mit krummen Zeiten)."""
    step = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    ok = (df.index - pd.Timestamp("1970-01-01", tz="UTC")) % step == pd.Timedelta(0)
    if (~ok).any():
        log.warning("%d Kerzen außerhalb des %s-Rasters entfernt", int((~ok).sum()), interval)
    return df[ok]


def drop_unclosed(df: pd.DataFrame, interval: str, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    end = df.index + pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    return df[end <= now]


def resample_4h(df1h: pd.DataFrame) -> pd.DataFrame:
    """4H-Kerzen aus 1H-Kerzen, nur vollständige Blöcke (4 Stunden vorhanden)."""
    g = df1h.resample("4h", label="left", closed="left", origin="epoch")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    count = g["close"].count()
    return out[count == 4].dropna()


# --- Binance --------------------------------------------------------------

def _parse_klines(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    df.index = _to_ts(df.pop("t"))
    return df


def binance_vision_klines(symbol: str, interval: str, start: str = "2017-08",
                          market: str = "spot") -> pd.DataFrame:
    """Komplette Historie aus den Monatsdateien (mit lokalem Cache).
    ``market="futures"``: USDT-Perpetuals; dort ohne den laufenden Monat,
    weil die Futures-API aus den USA (GitHub-Server) gesperrt ist."""
    prefix = {"spot": "spot", "futures": "futures/um"}[market]
    cache = CACHE / (f"bv_{symbol}_{interval}.csv" if market == "spot" else f"bvf_{symbol}_{interval}.csv")
    CACHE.mkdir(parents=True, exist_ok=True)
    have = pd.read_csv(cache, index_col=0, parse_dates=True) if cache.exists() else None
    months = pd.period_range(start, pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M") - 1, freq="M")
    frames = [have] if have is not None else []
    done = set(have.index.to_period("M").astype(str)) if have is not None else set()
    for m in months:
        if str(m) in done:
            continue
        url = (f"https://data.binance.vision/data/{prefix}/monthly/klines/{symbol}/{interval}/"
               f"{symbol}-{interval}-{m}.zip")
        r = _get(url)
        if r is None:
            continue  # Coin existierte in diesem Monat noch nicht
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(raw.iloc[0, 0]).isdigit():  # Kopfzeile vorhanden
            raw = raw.iloc[1:]
        frames.append(_parse_klines(raw.values.tolist()))
    if not frames:
        raise RuntimeError(f"Keine Daten für {symbol} {interval}")
    df = align(clean(pd.concat(frames)), interval)
    df.to_csv(cache)
    if market != "spot":
        return df
    # aktueller Monat über die API ergänzen
    recent = binance_api_klines(symbol, interval, since=df.index[-1])
    return align(clean(pd.concat([df, recent])), interval)


def binance_api_klines(symbol: str, interval: str, since: Optional[pd.Timestamp] = None,
                       limit_bars: int = 1000) -> pd.DataFrame:
    url = "https://data-api.binance.vision/api/v3/klines"
    step = INTERVAL_MS[interval]
    now_ms = int(time.time() * 1000)
    start = int(since.timestamp() * 1000) if since is not None else now_ms - limit_bars * step
    rows = []
    while start < now_ms:
        r = _get(url, {"symbol": symbol, "interval": interval, "startTime": start, "limit": 1000})
        data = r.json() if r is not None else []
        if not data:
            break
        rows.extend(data)
        start = data[-1][0] + step
        if len(data) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], tz="UTC"))
    return drop_unclosed(align(clean(_parse_klines(rows)), interval), interval)


# --- Coinbase (Ersatz) ------------------------------------------------------

def coinbase_klines(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    product = COINBASE_SYMBOL[symbol]
    gran = COINBASE_GRAN[interval]
    end = int(time.time()) // gran * gran
    start = end - bars * gran
    rows = []
    cur = start
    while cur < end:
        stop = min(cur + 300 * gran, end)
        r = _get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                 {"granularity": gran, "start": datetime.fromtimestamp(cur, timezone.utc).isoformat(),
                  "end": datetime.fromtimestamp(stop, timezone.utc).isoformat()})
        rows.extend(r.json() if r is not None else [])
        cur = stop
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df.pop("t"), unit="s", utc=True)
    return drop_unclosed(align(clean(df), interval), interval)


def recent_klines(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    """Die letzten ``bars`` abgeschlossenen Kerzen, Binance zuerst, sonst Coinbase."""
    try:
        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(milliseconds=INTERVAL_MS[interval] * (bars + 1))
        df = binance_api_klines(symbol, interval, since=since)
        if len(df) < bars * 0.95:
            raise RuntimeError("zu wenige Kerzen von Binance")
    except Exception as e:  # noqa: BLE001 - Ersatzquelle versuchen
        log.warning("Binance nicht erreichbar (%s), nutze Coinbase", e)
        if interval == "4h":  # Coinbase kennt kein 4H: aus 1H zusammensetzen
            df = resample_4h(coinbase_klines(symbol, "1h", bars * 4 + 8))
        else:
            df = coinbase_klines(symbol, interval, bars)
    return df.iloc[-bars:]


# --- Gold -----------------------------------------------------------------

def yahoo_daily(ticker: str = "GC=F", start: str = "2000-01-01") -> pd.DataFrame:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(time.time())
    r = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
             {"period1": p1, "period2": p2, "interval": "1d"})
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(res["timestamp"], unit="s", utc=True).normalize())
    df["volume"] = df["volume"].fillna(0)
    return drop_unclosed(clean(df.dropna(subset=["open", "high", "low", "close"])), "1d")


# --- Funding-Raten --------------------------------------------------------

def binance_funding_history(symbol: str = "BTCUSDT", start: str = "2020-01") -> pd.Series:
    months = pd.period_range(start, pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M") - 1, freq="M")
    parts = []
    for m in months:
        url = (f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/"
               f"{symbol}-fundingRate-{m}.zip")
        r = _get(url)
        if r is None:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = pd.read_csv(z.open(z.namelist()[0]))
        # Spalten: calc_time, funding_interval_hours, last_funding_rate
        ts = _to_ts(raw.iloc[:, 0])
        parts.append(pd.Series(raw.iloc[:, -1].astype(float).values, index=ts))
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    # Zeitstempel auf die volle Stunde runden (Berechnungszeit liegt minimal daneben)
    s.index = s.index.round("h")
    return s[~s.index.duplicated(keep="last")]


def recent_funding(symbol: str = "BTCUSDT", n: int = 30) -> pd.Series:
    """Letzte Funding-Zahlungen: Binance Futures, sonst Kraken Futures."""
    try:
        r = _get("https://fapi.binance.com/fapi/v1/fundingRate", {"symbol": symbol, "limit": n}, tries=2)
        data = r.json()
        s = pd.Series([float(d["fundingRate"]) for d in data],
                      index=pd.to_datetime([d["fundingTime"] for d in data], unit="ms", utc=True))
    except Exception as e:  # noqa: BLE001
        log.warning("Binance Futures nicht erreichbar (%s), nutze Kraken Futures", e)
        kraken = {"BTCUSDT": "PF_XBTUSD", "ETHUSDT": "PF_ETHUSD"}[symbol]
        r = _get("https://futures.kraken.com/derivatives/api/v4/historicalfundingrates",
                 {"symbol": kraken})
        rates = r.json()["rates"][-n * 8:]
        # Kraken zahlt stündlich, relativ zum Preis: auf 8-h-Summen umrechnen
        s = pd.Series([float(x["relativeFundingRate"]) for x in rates],
                      index=pd.to_datetime([x["timestamp"] for x in rates], utc=True))
        s = s.resample("8h", origin="epoch").sum(min_count=8).dropna()
    s.index = s.index.round("h")
    return s[s.index <= pd.Timestamp.now(tz="UTC")].sort_index()
