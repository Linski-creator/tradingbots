"""Aktive Intraday-Strategien für Futures (Long und Short).

Alle Parameter sind Lehrbuch-Standardwerte und werden NICHT optimiert.
``prepare`` berechnet alle Spalten kausal; ``on_bar(i, a, pos)`` bekommt die
Spalten als Arrays und entscheidet zum Close von Bar i.
Aktionen: ("order", {...}), ("exit", grund), ("cancel",)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import indicators as ind


def _ok(*v) -> bool:
    return all(x is not None and math.isfinite(x) for x in v)


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman: Netto-Bewegung / Summe der Einzelbewegungen (0 = Seitwärts, 1 = Trend)."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return (net / path).where(path > 0, 0.0)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP, der jeden Tag um 00:00 UTC neu beginnt."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.normalize()
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()
    return (pv / vol).where(vol > 0)


class HardcoreStrategy:
    name = label = ""
    timeframe = "15m"
    warmup = 0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def on_bar(self, i, a, pos):
        raise NotImplementedError


class AsiaBreakout(HardcoreStrategy):
    """Spanne 00:00-08:00 UTC. Erster 15m-Schluss außerhalb (08-20 Uhr):
    Einstieg in Ausbruchsrichtung, Stop Spannenmitte, Ziel 2R,
    spätestens um Mitternacht raus. Ein Trade pro Tag."""
    name, label, timeframe, warmup = "asia_breakout", "Asia-Range-Ausbruch", "15m", 40

    def __init__(self, end_hour=8, last_entry_hour=20, rr=2.0, min_w=0.003, max_w=0.04):
        self.end_hour, self.last_entry_hour, self.rr = end_hour, last_entry_hour, rr
        self.min_w, self.max_w = min_w, max_w

    def prepare(self, df):
        out = df.copy()
        day = out.index.normalize()
        hour = out.index.hour
        asia = hour < self.end_hour
        per_bar = (out.index[1] - out.index[0]) if len(out) > 1 else pd.Timedelta(minutes=15)
        expected = int(pd.Timedelta(hours=self.end_hour) / per_bar)
        hi = out["high"].where(asia).groupby(day).transform("max")
        lo = out["low"].where(asia).groupby(day).transform("min")
        cnt = out["close"].where(asia).groupby(day).transform("count")
        valid = (~asia) & (cnt == expected)
        out["r_hi"] = hi.where(valid)
        out["r_lo"] = lo.where(valid)
        brk = valid & ((out["close"] > out["r_hi"]) | (out["close"] < out["r_lo"]))
        out["first_break"] = brk & (brk.astype(int).groupby(day).cumsum() == 1)
        out["hour"] = hour + out.index.minute / 60
        return out

    def on_bar(self, i, a, pos):
        hr = a["hour"][i]
        if pos is not None:
            # letzte Bar des Tages: zum nächsten Open (Mitternacht) schließen
            return [("exit", "tagesende")] if hr >= 23.75 else []
        if not a["first_break"][i] or hr >= self.last_entry_hour:
            return []
        hi, lo, c = a["r_hi"][i], a["r_lo"][i], a["close"][i]
        if not _ok(hi, lo) or not (self.min_w <= (hi - lo) / c <= self.max_w):
            return []
        side = 1 if c > hi else -1
        return [("order", {"side": side, "type": "market", "sl": (hi + lo) / 2, "rr": self.rr,
                           "tag": "ausbruch_oben" if side == 1 else "ausbruch_unten",
                           "context": {"range_hi": hi, "range_lo": lo, "close": c}})]


class BollingerRevert(HardcoreStrategy):
    """Schluss außerhalb des Bollinger-Bands (20, 2), nur in Richtung des
    Trends (EMA 200 steigt/fällt): Limit-Order zum Schlusskurs, Stop 1,5 ATR,
    Ziel Mittellinie, max. 8 Stunden."""
    name, label, timeframe, warmup = "bb_revert", "Bollinger-Rückkehr 15m", "15m", 230

    def prepare(self, df):
        out = df.copy()
        mid = ind.sma(out["close"], 20)
        sd = out["close"].rolling(20, min_periods=20).std(ddof=0)
        out["mid"], out["up"], out["dn"] = mid, mid + 2 * sd, mid - 2 * sd
        out["ema"] = ind.ema(out["close"], 200)
        out["ema_slope"] = out["ema"] - out["ema"].shift(20)
        out["atr"] = ind.atr(out, 14)
        return out

    def on_bar(self, i, a, pos):
        if pos is not None:
            return []
        c, e, s, atr = a["close"][i], a["ema"][i], a["ema_slope"][i], a["atr"][i]
        if not _ok(c, e, s, atr, a["dn"][i], a["up"][i]):
            return []
        if c < a["dn"][i] and c > e and s > 0:
            return [("order", {"side": 1, "type": "limit", "price": c, "sl": c - 1.5 * atr,
                               "tp": a["mid"][i], "expire": 2, "max_bars": 32, "tag": "unter_band"})]
        if c > a["up"][i] and c < e and s < 0:
            return [("order", {"side": -1, "type": "limit", "price": c, "sl": c + 1.5 * atr,
                               "tp": a["mid"][i], "expire": 2, "max_bars": 32, "tag": "über_band"})]
        return []


class VWAPRevert(HardcoreStrategy):
    """5m: Kurs mehr als 2 Standardabweichungen vom Tages-VWAP entfernt und
    RSI(14) überdehnt: Limit-Order zurück Richtung VWAP. Stop 1,5 Abweichungen,
    Ziel VWAP, max. 4 Stunden. Erste 2 Stunden des Tages ausgelassen."""
    name, label, timeframe, warmup = "vwap_revert", "VWAP-Rückkehr 5m", "5m", 300

    def prepare(self, df):
        out = df.copy()
        out["vwap"] = session_vwap(out)
        out["dev"] = (out["close"] - out["vwap"]).rolling(288, min_periods=288).std(ddof=0)
        out["rsi"] = ind.rsi(out["close"], 14)
        out["hour"] = out.index.hour.astype(float)
        return out

    def on_bar(self, i, a, pos):
        if pos is not None or a["hour"][i] < 2:
            return []
        c, v, d, r = a["close"][i], a["vwap"][i], a["dev"][i], a["rsi"][i]
        if not _ok(c, v, d, r) or d <= 0:
            return []
        if c < v - 2 * d and r < 30:
            return [("order", {"side": 1, "type": "limit", "price": c, "sl": c - 1.5 * d, "tp": v,
                               "expire": 2, "max_bars": 48, "tag": "unter_vwap"})]
        if c > v + 2 * d and r > 70:
            return [("order", {"side": -1, "type": "limit", "price": c, "sl": c + 1.5 * d, "tp": v,
                               "expire": 2, "max_bars": 48, "tag": "über_vwap"})]
        return []


class RSI2Intraday(HardcoreStrategy):
    """15m: RSI(2) < 5 über EMA 200 -> Long, RSI(2) > 95 unter EMA 200 -> Short.
    Ausstieg RSI zurück über 60 / unter 40, Stop 2 ATR, max. 6 Stunden."""
    name, label, timeframe, warmup = "rsi2_15m", "RSI(2) 15m mit Trendfilter", "15m", 210

    def prepare(self, df):
        out = df.copy()
        out["rsi"] = ind.rsi(out["close"], 2)
        out["ema"] = ind.ema(out["close"], 200)
        out["atr"] = ind.atr(out, 14)
        return out

    def on_bar(self, i, a, pos):
        c, r, e, atr = a["close"][i], a["rsi"][i], a["ema"][i], a["atr"][i]
        if not _ok(c, r, e, atr):
            return []
        if pos is not None:
            if pos["side"] == 1 and r > 60:
                return [("exit", "rsi_zurück")]
            if pos["side"] == -1 and r < 40:
                return [("exit", "rsi_zurück")]
            return []
        if r < 5 and c > e:
            return [("order", {"side": 1, "type": "market", "sl": c - 2 * atr, "max_bars": 24, "tag": "rsi_tief"})]
        if r > 95 and c < e:
            return [("order", {"side": -1, "type": "market", "sl": c + 2 * atr, "max_bars": 24, "tag": "rsi_hoch"})]
        return []


class VolumeBreakout(HardcoreStrategy):
    """5m: Volumen > 3x Durchschnitt, Schluss über/unter dem 20-Bar-Extrem,
    kräftige Kerze (Körper > 60 % der Spanne). Stop am anderen Kerzenende,
    Ziel 2R, max. 3 Stunden."""
    name, label, timeframe, warmup = "vol_breakout", "Volumen-Ausbruch 5m", "5m", 60

    def prepare(self, df):
        out = df.copy()
        out["vol_avg"] = out["volume"].rolling(50, min_periods=50).mean().shift(1)
        out["hh"] = ind.highest(out["high"], 20).shift(1)
        out["ll"] = ind.lowest(out["low"], 20).shift(1)
        rng = (out["high"] - out["low"]).replace(0, np.nan)
        out["body"] = (out["close"] - out["open"]).abs() / rng
        return out

    def on_bar(self, i, a, pos):
        if pos is not None:
            return []
        c, hi, lo = a["close"][i], a["high"][i], a["low"][i]
        va, body = a["vol_avg"][i], a["body"][i]
        if not _ok(c, va, body, a["hh"][i], a["ll"][i]) or va <= 0:
            return []
        if a["volume"][i] < 3 * va or body < 0.6:
            return []
        if c > a["hh"][i] and (c - lo) / c >= 0.0015:
            return [("order", {"side": 1, "type": "market", "sl": lo, "rr": 2.0, "max_bars": 36, "tag": "vol_hoch"})]
        if c < a["ll"][i] and (hi - c) / c >= 0.0015:
            return [("order", {"side": -1, "type": "market", "sl": hi, "rr": 2.0, "max_bars": 36, "tag": "vol_runter"})]
        return []


class RangeGrid(HardcoreStrategy):
    """15m-"Grid mit Stop": nur im Seitwärtsmarkt (Efficiency Ratio < 0,2).
    Nahe am unteren Rand der 12-Stunden-Spanne kaufen, nahe am oberen verkaufen
    (Limit), Ziel Mitte, Stop 0,5 ATR hinter dem Rand. Raus, wenn ein Trend
    entsteht (ER > 0,4)."""
    name, label, timeframe, warmup = "range_grid", "Range-Grid mit Stop", "15m", 60

    def prepare(self, df):
        out = df.copy()
        out["er"] = efficiency_ratio(out["close"], 48)
        out["top"] = ind.highest(out["high"], 48).shift(1)
        out["bot"] = ind.lowest(out["low"], 48).shift(1)
        out["atr"] = ind.atr(out, 14)
        return out

    def on_bar(self, i, a, pos):
        er, top, bot, atr, c = a["er"][i], a["top"][i], a["bot"][i], a["atr"][i], a["close"][i]
        if not _ok(er, top, bot, atr, c):
            return []
        if pos is not None:
            return [("exit", "trend_beginnt")] if er > 0.4 else []
        w = top - bot
        if er >= 0.2 or w <= 0 or w / c < 0.004:
            return []
        mid = (top + bot) / 2
        if c < bot + 0.25 * w:
            return [("order", {"side": 1, "type": "limit", "price": c, "sl": bot - 0.5 * atr, "tp": mid,
                               "expire": 2, "max_bars": 64, "tag": "unten_kaufen"})]
        if c > top - 0.25 * w:
            return [("order", {"side": -1, "type": "limit", "price": c, "sl": top + 0.5 * atr, "tp": mid,
                               "expire": 2, "max_bars": 64, "tag": "oben_verkaufen"})]
        return []


HARDCORE = {s.name: s for s in (AsiaBreakout, BollingerRevert, VWAPRevert, RSI2Intraday, VolumeBreakout,
                                 RangeGrid)}
