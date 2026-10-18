"""Technische Indikatoren.

Alle Funktionen sind kausal: Der Wert an Position i hängt nur von Daten
bis einschließlich i ab. Vor ausreichend Historie ist der Wert NaN.
Glättungen (EMA, Wilder) werden wie bei TradingView mit dem einfachen
Durchschnitt der ersten n Werte gestartet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _seeded_recursive(values: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Rekursive Glättung y[t] = alpha*x[t] + (1-alpha)*y[t-1].

    Startwert ist der Mittelwert der ersten n gültigen Werte. Führende NaNs
    werden übersprungen; ein NaN nach dem Start ist ein Datenfehler.
    """
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < n:
        return out
    start = valid[0]
    if np.isnan(values[start:]).any():
        raise ValueError("Lücke (NaN) mitten in der Zeitreihe")
    seed_end = start + n - 1
    out[seed_end] = values[start:seed_end + 1].mean()
    for t in range(seed_end + 1, len(values)):
        out[t] = alpha * values[t] + (1 - alpha) * out[t - 1]
    return out


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    vals = _seeded_recursive(s.to_numpy(dtype=float), n, 2.0 / (n + 1))
    return pd.Series(vals, index=s.index)


def wilder(s: pd.Series, n: int) -> pd.Series:
    vals = _seeded_recursive(s.to_numpy(dtype=float), n, 1.0 / n)
    return pd.Series(vals, index=s.index)


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder((-delta).clip(lower=0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
        out = 100 - 100 / (1 + rs)
    # Nur Gewinne (loss == 0): RSI = 100; weder Gewinn noch Verlust: 50
    out = out.where(loss != 0, 100.0)
    out = out.where(~((gain == 0) & (loss == 0)), 50.0)
    return out.where(gain.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return wilder(true_range(df), n)


def highest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).max()


def lowest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).min()


def realized_vol(close: pd.Series, n: int, bars_per_year: float) -> pd.Series:
    """Annualisierte Volatilität der Log-Renditen der letzten n Bars."""
    r = np.log(close).diff()
    return r.rolling(n, min_periods=n).std(ddof=1) * np.sqrt(bars_per_year)
