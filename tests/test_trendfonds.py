import numpy as np
import pandas as pd

from bots import trendfonds as T


def prices(n=1500, seed=0):
    idx = pd.bdate_range("2010-01-01", periods=n)
    rng = np.random.default_rng(seed)
    cols = {}
    for k, d in enumerate((0.0008, -0.0008, 0.0, 0.0004)):
        cols[f"A{k}"] = 100 * np.exp(np.cumsum(rng.normal(d, 0.01, n)))
    return pd.DataFrame(cols, index=idx)


def test_weights_weekly_capped_and_no_lookahead():
    px = prices()
    w = T.weights(px)
    assert (w.abs().sum(axis=1) <= T.MAX_GROSS + 1e-9).all()
    changes = w.diff().abs().sum(axis=1) > 0
    assert set(w.index[changes].dayofweek) <= {4}
    # Zukunft verändern darf frühere Gewichte nicht ändern
    px2 = px.copy()
    px2.iloc[1000:] *= 1.5
    pd.testing.assert_frame_equal(T.weights(px).iloc[:1000], T.weights(px2).iloc[:1000])


def test_trend_portfolio_profits_on_trends_and_long_only():
    px = prices()
    r = T.backtest(px)
    assert T.stats(r)["cagr_pct"] > 0
    assert (T.weights(px, long_only=True) >= 0).all().all()
    assert T.stats(T.backtest(px, cost=0.01))["cagr_pct"] < T.stats(r)["cagr_pct"]
