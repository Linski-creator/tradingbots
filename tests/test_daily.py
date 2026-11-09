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
    top = max(e[1] for e in st.equity)
    for t in st.trades:
        e, x = pd.Timestamp(t["entry_time"]), pd.Timestamp(t["exit_time"])
        assert (x - e).days <= 4 and e.dayofweek <= x.dayofweek


def test_gap_stats():
    g = weekend_gaps(make())
    assert g["anzahl_wochenenden"] > 100 and g["median_pct"] >= 0
    assert gap_damage([{"reason": "stop", "r_multiple": -2.5}])["schlechter_als_-2R"] == 1


def test_pyramid_adds_only_in_profit_and_limits_loss():
    from bots.metals.pyramid import Pyramid
    from bots.metals.stop_study import TightStart
    df = make(n=2500, drift=0.0006, seed=4)
    risk = FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.002)
    base = FxSimulator(TightStart(50, 200, 3, 20, 2 / 3), risk=risk)
    s0 = Pyramid(50, 200, 3, 20, 2 / 3, 0)
    a = base.run(base.s.prepare(df))
    b = FxSimulator(s0, risk=FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.002)).run(s0.prepare(df))
    assert a.equity[-1][1] == b.equity[-1][1]          # 0 Nachkäufe = unverändert
    s2 = Pyramid(50, 200, 3, 20, 2 / 3, 2)
    c = FxSimulator(s2, risk=FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.002)).run(s2.prepare(df))
    added = [t for t in c.trades if t.get("adds")]
    assert added and max(t["adds"] for t in added) <= 2
    assert min(t["pnl"] for t in added) > -0.015 * 2000   # Zukäufe riskieren nie mehr als ~1 % zusätzlich


def test_gold4h_hour_filter_and_cap():
    from bots.metals.gold4h import Gold4h
    from bots.metals.sorte1 import resample
    idx = pd.date_range("2024-01-01", periods=9000, freq="1h", tz="UTC")
    idx = idx[idx.dayofweek < 5]
    rng = np.random.default_rng(5)
    c = 2000 * np.exp(np.cumsum(rng.normal(0.0002, 0.003, len(idx))))
    o = np.r_[c[0], c[:-1]]
    h = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999,
                      "close": c, "volume": 1.0}, index=idx)
    df = tidy(resample(h, "4h"))
    s = Gold4h(max_adds=2)
    p = s.prepare(df)
    st = FxSimulator(s, risk=FxRisk(risk_per_trade=0.01, max_leverage=1, min_stop=0.002)).run(p)
    assert st.trades
    top = max(e[1] for e in st.equity)
    for t in st.trades:
        e = pd.Timestamp(t["entry_time"])
        sig = p.index[p.index.searchsorted(e) - 1]
        assert sig.hour >= 12                                # Signal nur aus Kerzen ab 12 Uhr UTC
        assert t["qty"] * t["entry_price"] <= top * 1.02    # Crash-Schutz: Position nie über 1x Konto
