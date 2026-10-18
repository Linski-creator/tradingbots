import numpy as np
import pandas as pd
import pytest


def make_ohlc(n=600, seed=0, freq="1D", drift=0.0005, vol=0.03, start=100.0):
    """Künstliche Kerzen nur für Tests der Rechenlogik (keine Marktdaten)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start], close[:-1]]) * np.exp(rng.normal(0, vol / 4, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": 1.0}, index=idx)


def bars(rows, freq="1D"):
    """Kerzen aus Liste von (open, high, low, close)."""
    idx = pd.date_range("2021-01-01", periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


@pytest.fixture
def ohlc():
    return make_ohlc()
