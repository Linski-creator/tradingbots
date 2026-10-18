"""Die Einzelmarkt-Strategien.

Jede Strategie hat:
  - ``prepare(df, bars_per_year)``: berechnet alle Indikatoren kausal
  - ``on_bar(view)``: entscheidet zum Close einer Bar, gibt Signale zurück
  - ``warmup``: Bars, die die Indikatoren mindestens brauchen

Alle Regeln sind Long-only (Kaufen oder Cash), ohne Hebel.
"""

from __future__ import annotations

import math

import pandas as pd

from . import indicators as ind
from .engine import BarView, Enter, Exit, SetStop


def _ok(*vals) -> bool:
    return all(v is not None and isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def _with_vol(df: pd.DataFrame, bars_per_year: float, n: int) -> pd.DataFrame:
    out = df.copy()
    out["vol"] = ind.realized_vol(out["close"], n, bars_per_year)
    return out


class Strategy:
    name = "base"
    label = ""
    timeframe = "1d"
    sizing = "risk"
    warmup = 0

    def prepare(self, df: pd.DataFrame, bars_per_year: float) -> pd.DataFrame:
        raise NotImplementedError

    def on_bar(self, v: BarView) -> list:
        raise NotImplementedError


class SMA200Filter(Strategy):
    """Über dem 200-Tage-Durchschnitt investiert, darunter Cash."""
    name, label, timeframe, sizing = "sma200", "200-Tage-Linie", "1d", "vol"

    def __init__(self, n: int = 200, vol_n: int = 30):
        self.n, self.vol_n = n, vol_n
        self.warmup = n

    def prepare(self, df, bars_per_year):
        out = _with_vol(df, bars_per_year, self.vol_n)
        out["sma"] = ind.sma(out["close"], self.n)
        return out

    def on_bar(self, v):
        c, s = v.row["close"], v.row["sma"]
        if not _ok(s):
            return []
        if v.position is None and c > s:
            return [Enter(vol=v.row["vol"])]
        if v.position is not None and c < s:
            return [Exit("unter_sma")]
        return []


class TimeSeriesMomentum(Strategy):
    """Investiert, solange die Rendite der letzten ``lookback`` Tage positiv ist."""
    name, label, timeframe, sizing = "tsmom", "Zeitreihen-Momentum", "1d", "vol"

    def __init__(self, lookback: int = 90, vol_n: int = 30):
        self.lookback, self.vol_n = lookback, vol_n
        self.warmup = max(lookback, vol_n) + 1

    def prepare(self, df, bars_per_year):
        out = _with_vol(df, bars_per_year, self.vol_n)
        out["mom"] = out["close"] / out["close"].shift(self.lookback) - 1
        return out

    def on_bar(self, v):
        m = v.row["mom"]
        if not _ok(m):
            return []
        if v.position is None and m > 0:
            return [Enter(vol=v.row["vol"])]
        if v.position is not None and m <= 0:
            return [Exit("momentum_negativ")]
        return []


class DonchianBreakout(Strategy):
    """Turtle-Ausbruch: neues 20-Tage-Hoch kaufen, Stop 2 ATR,
    Ausstieg bei neuem 10-Tage-Tief."""
    name, label, timeframe, sizing = "donchian", "Donchian-Breakout", "1d", "risk"

    def __init__(self, entry_n: int = 20, exit_n: int = 10, atr_n: int = 20, atr_mult: float = 2.0):
        self.entry_n, self.exit_n, self.atr_n, self.atr_mult = entry_n, exit_n, atr_n, atr_mult
        self.warmup = max(entry_n, exit_n, atr_n) + 1

    def prepare(self, df, bars_per_year):
        out = df.copy()
        # Kanal aus den VORHERIGEN Bars, sonst wäre ein Ausbruch unmöglich
        out["upper"] = ind.highest(out["high"], self.entry_n).shift(1)
        out["lower"] = ind.lowest(out["low"], self.exit_n).shift(1)
        out["atr"] = ind.atr(out, self.atr_n)
        return out

    def on_bar(self, v):
        r = v.row
        if not _ok(r["upper"], r["lower"], r["atr"]):
            return []
        if v.position is None and r["close"] > r["upper"]:
            return [Enter(stop=r["close"] - self.atr_mult * r["atr"])]
        if v.position is not None and r["close"] < r["lower"]:
            return [Exit("kanal_tief")]
        return []


class ChandelierTrend(Strategy):
    """EMA 20 kreuzt EMA 50 nach oben (über EMA 200): Einstieg.
    Ausstieg über nachgezogenen Stop: höchstes Hoch seit Einstieg - 3 ATR."""
    name, label, timeframe, sizing = "chandelier", "Chandelier-Trend", "4h", "risk"

    def __init__(self, fast: int = 20, slow: int = 50, trend: int = 200, atr_n: int = 22,
                 atr_mult: float = 3.0):
        self.fast, self.slow, self.trend, self.atr_n, self.atr_mult = fast, slow, trend, atr_n, atr_mult
        self.warmup = trend + 1

    def prepare(self, df, bars_per_year):
        out = df.copy()
        out["ema_f"] = ind.ema(out["close"], self.fast)
        out["ema_s"] = ind.ema(out["close"], self.slow)
        out["ema_t"] = ind.ema(out["close"], self.trend)
        out["atr"] = ind.atr(out, self.atr_n)
        above = out["ema_f"] > out["ema_s"]
        valid = out["ema_f"].notna() & out["ema_s"].notna()
        prev_valid = valid.shift(1, fill_value=False)
        # Kreuzung nur, wenn beide EMAs auch in der Vorbar schon gültig waren
        out["cross_up"] = above & ~above.shift(1, fill_value=True) & valid & prev_valid
        return out

    def on_bar(self, v):
        r = v.row
        if not _ok(r["ema_f"], r["ema_s"], r["ema_t"], r["atr"]):
            return []
        if v.position is None:
            if bool(r["cross_up"]) and r["close"] > r["ema_t"]:
                return [Enter(stop=r["close"] - self.atr_mult * r["atr"])]
            return []
        return [SetStop(v.position.highest - self.atr_mult * r["atr"])]


class RSI2MeanReversion(Strategy):
    """Connors RSI(2): im Aufwärtstrend (über SMA 200) bei RSI(2) < 10 kaufen,
    bei RSI(2) > 70 oder nach 10 Tagen verkaufen. Notfall-Stop 2,5 ATR."""
    name, label, timeframe, sizing = "rsi2", "RSI(2) mit Trendfilter", "1d", "risk"

    def __init__(self, rsi_n: int = 2, entry: float = 10, exit: float = 70, trend: int = 200,
                 max_bars: int = 10, atr_n: int = 14, atr_mult: float = 2.5):
        self.rsi_n, self.entry, self.exit, self.trend = rsi_n, entry, exit, trend
        self.max_bars, self.atr_n, self.atr_mult = max_bars, atr_n, atr_mult
        self.warmup = trend

    def prepare(self, df, bars_per_year):
        out = df.copy()
        out["rsi"] = ind.rsi(out["close"], self.rsi_n)
        out["sma"] = ind.sma(out["close"], self.trend)
        out["atr"] = ind.atr(out, self.atr_n)
        return out

    def on_bar(self, v):
        r = v.row
        if not _ok(r["rsi"], r["sma"], r["atr"]):
            return []
        if v.position is None:
            if r["close"] > r["sma"] and r["rsi"] < self.entry:
                return [Enter(stop=r["close"] - self.atr_mult * r["atr"])]
            return []
        if r["rsi"] > self.exit:
            return [Exit("rsi_hoch")]
        if v.position.bars_held >= self.max_bars:
            return [Exit("zeitstop")]
        return []


class EMAPullback(Strategy):
    """Aufwärtstrend (EMA 50 > EMA 200, Kurs > EMA 200). Rücksetzer berührt
    die EMA 20 und schließt grün darüber: Einstieg. Stop unter dem Tief der
    letzten 5 Bars, Ziel 3:1."""
    name, label, timeframe, sizing = "pullback", "Rücksetzer auf EMA 20 (3:1)", "1h", "risk"

    def __init__(self, fast: int = 20, mid: int = 50, trend: int = 200, swing_n: int = 5,
                 atr_n: int = 14, rr: float = 3.0, min_atr: float = 0.5, max_atr: float = 3.0):
        self.fast, self.mid, self.trend, self.swing_n = fast, mid, trend, swing_n
        self.atr_n, self.rr, self.min_atr, self.max_atr = atr_n, rr, min_atr, max_atr
        self.warmup = trend + 1

    def prepare(self, df, bars_per_year):
        out = df.copy()
        out["ema_f"] = ind.ema(out["close"], self.fast)
        out["ema_m"] = ind.ema(out["close"], self.mid)
        out["ema_t"] = ind.ema(out["close"], self.trend)
        out["atr"] = ind.atr(out, self.atr_n)
        out["swing_low"] = ind.lowest(out["low"], self.swing_n)
        return out

    def on_bar(self, v):
        r = v.row
        if v.position is not None:
            return []  # Stop und Ziel erledigen den Ausstieg
        if not _ok(r["ema_f"], r["ema_m"], r["ema_t"], r["atr"], r["swing_low"]):
            return []
        trend = r["ema_m"] > r["ema_t"] and r["close"] > r["ema_t"]
        touch = r["low"] <= r["ema_f"] < r["close"] and r["close"] > r["open"]
        if not (trend and touch):
            return []
        stop = r["swing_low"] - 0.1 * r["atr"]
        dist = r["close"] - stop
        if not (self.min_atr * r["atr"] <= dist <= self.max_atr * r["atr"]):
            return []
        return [Enter(stop=stop, rr=self.rr)]


SINGLE_MARKET = {
    s.name: s for s in (SMA200Filter, TimeSeriesMomentum, DonchianBreakout, ChandelierTrend,
                        RSI2MeanReversion, EMAPullback)
}
