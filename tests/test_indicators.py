"""Indikatoren gegen unabhängige, bewusst simpel geschriebene Referenzen
und gegen von Hand nachgerechnete Werte."""

import math

import numpy as np
import pandas as pd
import pytest

from bots import indicators as ind
from tests.conftest import make_ohlc


def ref_ema(xs, n):
    out = [math.nan] * len(xs)
    k = 2 / (n + 1)
    prev = sum(xs[:n]) / n
    out[n - 1] = prev
    for i in range(n, len(xs)):
        prev = xs[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def ref_rsi(xs, n):
    out = [math.nan] * len(xs)
    gains = [max(xs[i] - xs[i - 1], 0) for i in range(1, len(xs))]
    losses = [max(xs[i - 1] - xs[i], 0) for i in range(1, len(xs))]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n

    def val(g, l):
        if l == 0:
            return 50.0 if g == 0 else 100.0
        return 100 - 100 / (1 + g / l)

    out[n] = val(ag, al)
    for i in range(n, len(gains)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
        out[i + 1] = val(ag, al)
    return out


def ref_atr(h, l, c, n):
    trs = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
                           for i in range(1, len(c))]
    out = [math.nan] * len(c)
    prev = sum(trs[:n]) / n
    out[n - 1] = prev
    for i in range(n, len(c)):
        prev = (prev * (n - 1) + trs[i]) / n
        out[i] = prev
    return out


def assert_series_close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    np.testing.assert_allclose(a[m], b[m], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("n", [2, 5, 20, 50])
def test_ema_matches_reference(n):
    df = make_ohlc(300, seed=n)
    assert_series_close(ind.ema(df["close"], n), ref_ema(df["close"].tolist(), n))


@pytest.mark.parametrize("n", [2, 14])
def test_rsi_matches_reference(n):
    df = make_ohlc(300, seed=n + 7)
    assert_series_close(ind.rsi(df["close"], n), ref_rsi(df["close"].tolist(), n))


@pytest.mark.parametrize("n", [14, 20, 22])
def test_atr_matches_reference(n):
    df = make_ohlc(300, seed=n + 3)
    assert_series_close(ind.atr(df, n),
                        ref_atr(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), n))


def test_hand_computed_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # EMA(3): Start = (1+2+3)/3 = 2, dann 0.5*4+0.5*2 = 3, 0.5*5+0.5*3 = 4
    assert ind.ema(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert ind.sma(s, 2).tolist()[1:] == [1.5, 2.5, 3.5, 4.5]
    # RSI(2): nur Gewinne -> 100
    assert ind.rsi(s, 2).iloc[-1] == 100.0
    # RSI(2) auf 10, 11, 10: Gewinne 1, 0 -> ag=0.5, Verluste 0, 1 -> al=0.5 -> 50
    assert ind.rsi(pd.Series([10.0, 11.0, 10.0]), 2).iloc[-1] == pytest.approx(50.0)
    # konstanter Kurs -> RSI 50
    assert ind.rsi(pd.Series([5.0] * 6), 2).iloc[-1] == 50.0


def test_rsi_bounds():
    df = make_ohlc(1000, seed=11)
    r = ind.rsi(df["close"], 2).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_highest_lowest_and_vol():
    s = pd.Series([3.0, 1.0, 4.0, 1.0, 5.0])
    assert ind.highest(s, 3).tolist()[2:] == [4.0, 4.0, 5.0]
    assert ind.lowest(s, 3).tolist()[2:] == [1.0, 1.0, 1.0]
    flat = pd.Series([100.0] * 40)
    assert ind.realized_vol(flat, 30, 365).iloc[-1] == 0.0


def test_gap_in_series_raises():
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    with pytest.raises(ValueError):
        ind.ema(s, 2)
