import numpy as np
import pandas as pd

from bots.hardcore.engine import FxRisk, FxSimulator
from bots.metals.daily import DailyTrend, gap_damage, tidy, weekend_gaps


def make(n=900, drift=0.001, seed=1):
    idx = pd.bdate_range("2010-01-01", periods=n, tz="UTC")
    r = np.random.default_rng(seed).normal(drift, 0.01, n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]] * (1 + np.random.default_rng(seed + 1).normal(0, 0.002, n))
    h = np.maximum(o, c) * 1.004
    l = np.minimum(o, c) * 0.996
    return tidy(pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx))


def run(df, **kw):
    s = DailyTrend(**kw)
    return FxSimulator(s, risk=FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.002)).run(s.prepare(df))


def test_all_rules_run_and_uptrend_profits():
    df = make()
    for rule in ("donchian", "sma", "pullback"):
        for flat in (False, True):
            st = run(df, rule=rule, shorts=True, flat_weekend=flat)
            assert st.equity
    st = run(df, rule="sma", shorts=False)
    assert st.trades and st.equity[-1][1] > 1000


def test_flat_weekend_never_holds_over_weekend():
    st = run(make(), rule="sma", shorts=False, flat_weekend=True)
    assert st.trades
    for t in st.trades:
        e, x = pd.Timestamp(t["entry_time"]), pd.Timestamp(t["exit_time"])
        assert (x - e).days <= 4 and e.dayofweek <= x.dayofweek


def test_gap_stats():
    g = weekend_gaps(make())
    assert g["anzahl_wochenenden"] > 100 and g["median_pct"] >= 0
    assert gap_damage([{"reason": "stop", "r_multiple": -2.5}])["schlechter_als_-2R"] == 1
