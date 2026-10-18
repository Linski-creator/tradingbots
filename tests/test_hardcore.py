"""Futures-Simulator und Hardcore-Strategien."""

import json

import numpy as np
import pandas as pd
import pytest

from bots.hardcore.engine import FxCosts, FxRisk, FxSimulator, FxState
from bots.hardcore.strategies import HARDCORE, AsiaBreakout, efficiency_ratio, session_vwap
from tests.conftest import bars, make_ohlc

ZERO = FxCosts(maker=0, taker=0, slippage=0, funding_8h=0)


class Script:
    def __init__(self, script):
        self.script = script

    def on_bar(self, i, a, pos):
        return self.script.get(i, [])


def run(rows, script, costs=ZERO, risk=None, freq="15min"):
    df = bars(rows, freq=freq)
    return FxSimulator(Script(script), costs, risk or FxRisk(risk_per_trade=0.01, max_leverage=100)).run(df)


FLAT = (100, 100, 100, 100)


def test_short_profit_and_market_fill():
    st = run([FLAT, (100, 100, 95, 95), (95, 95, 90, 90), (90, 91, 89, 90)],
             {0: [("order", {"side": -1, "type": "market", "sl": 110})], 2: [("exit", "x")]})
    t = st.trades[0]
    assert t["side"] == "short" and t["entry_price"] == 100 and t["exit_price"] == 90
    # Risiko 1 % = 10 € bei 10 Punkten Stop -> 1 Stück -> 10 € Gewinn
    assert t["pnl"] == pytest.approx(10.0) and t["r_multiple"] == pytest.approx(1.0)


def test_limit_fill_maker_fee_and_gap_better_price():
    costs = FxCosts(maker=0.001, taker=0.002, slippage=0.0, funding_8h=0)
    st = run([FLAT, (99, 99, 97, 98), FLAT],
             {0: [("order", {"side": 1, "type": "limit", "price": 99.5, "sl": 90})], 1: [("exit", "x")]},
             costs=costs)
    t = st.trades[0]
    assert t["entry_price"] == 99  # Bar öffnet unter dem Limit -> Open
    entry_fee = t["qty"] * 99 * 0.001
    exit_fee = t["qty"] * 100 * 0.002
    assert t["fees"] == pytest.approx(entry_fee + exit_fee)
    assert t["pnl"] == pytest.approx(t["qty"] * 1 - entry_fee - exit_fee)


def test_limit_not_filled_expires():
    st = run([FLAT, (100, 101, 99.9, 100), (100, 101, 99.9, 100), (100, 100, 90, 95)],
             {0: [("order", {"side": 1, "type": "limit", "price": 99, "sl": 95, "expire": 2})]})
    assert not st.trades and st.position is None and st.order is None


def test_stop_entry_and_slippage():
    costs = FxCosts(maker=0, taker=0, slippage=0.001, funding_8h=0)
    st = run([FLAT, (100, 106, 99, 105), FLAT],
             {0: [("order", {"side": 1, "type": "stop", "price": 105, "sl": 95})], 1: [("exit", "x")]},
             costs=costs)
    assert st.trades[0]["entry_price"] == pytest.approx(105 * 1.001)


def test_stop_beats_target_and_no_target_on_entry_bar():
    # Einstiegs-Bar berührt Ziel -> zählt nicht
    st = run([FLAT, (100, 120, 99, 100), (100, 100, 100, 100)],
             {0: [("order", {"side": 1, "type": "market", "sl": 95, "tp": 110})]})
    assert st.position is not None and not st.trades
    # danach Stop und Ziel in derselben Bar -> Stop
    st = run([FLAT, FLAT, (100, 120, 90, 100)],
             {0: [("order", {"side": 1, "type": "market", "sl": 95, "tp": 110})]})
    assert st.trades[0]["reason"] == "stop" and st.trades[0]["exit_price"] == 95


def test_stop_on_entry_bar_and_gap():
    st = run([FLAT, (100, 100, 94, 96)], {0: [("order", {"side": 1, "type": "market", "sl": 95})]})
    assert st.trades[0]["reason"] == "stop" and st.trades[0]["exit_price"] == 95
    st = run([FLAT, FLAT, (90, 91, 88, 89)], {0: [("order", {"side": 1, "type": "market", "sl": 95})]})
    assert st.trades[0]["exit_price"] == 90  # Lücke unter den Stop: Open
    st = run([FLAT, FLAT, (110, 112, 108, 111)], {0: [("order", {"side": -1, "type": "market", "sl": 105})]})
    assert st.trades[0]["exit_price"] == 110


def test_take_profit_short_and_rr():
    st = run([FLAT, FLAT, (99, 99, 79, 85)], {0: [("order", {"side": -1, "type": "market", "sl": 110, "rr": 2})]})
    t = st.trades[0]
    assert t["reason"] == "take_profit" and t["exit_price"] == 80 and t["r_multiple"] == pytest.approx(2.0)


def test_leverage_cap_and_risk_amount():
    risk = FxRisk(risk_per_trade=0.01, max_leverage=3)
    st = run([FLAT, FLAT], {0: [("order", {"side": 1, "type": "market", "sl": 99.8})]}, risk=risk)
    assert st.position["qty"] * 100 == pytest.approx(3000)  # 3x gedeckelt
    st = run([FLAT, FLAT], {0: [("order", {"side": 1, "type": "market", "sl": 99.99})]}, risk=risk)
    assert st.position is None  # Stop näher als 0,15 %: kein Einstieg
    st = run([FLAT, FLAT, (100, 100, 89, 90)], {0: [("order", {"side": 1, "type": "market", "sl": 90})]},
             costs=FxCosts(), risk=FxRisk())
    assert 9.0 < 1000 - st.cash <= 10.0 + 1e-9  # Verlust bis Stop höchstens 1 % inkl. Kosten


def test_funding_cost_and_time_stop():
    costs = FxCosts(maker=0, taker=0, slippage=0, funding_8h=0.001)
    rows = [FLAT] * 40
    st = run(rows, {0: [("order", {"side": 1, "type": "market", "sl": 90, "max_bars": 32})]}, costs=costs,
             freq="15min")
    t = st.trades[0]
    assert t["reason"] == "zeitstop" and t["bars_held"] == 32
    # 32 Bars à 15 min = 8 h -> 0,1 % vom Nominal
    assert t["pnl"] == pytest.approx(-t["qty"] * 100 * 0.001)


def test_halt_closes_and_blocks():
    st = FxSimulator(Script({0: [("order", {"side": 1, "type": "market", "sl": 50})],
                             3: [("order", {"side": 1, "type": "market", "sl": 50})]}),
                     ZERO, FxRisk(risk_per_trade=0.5, max_leverage=1, halt_drawdown=0.1)).run(
        bars([FLAT, FLAT, (100, 100, 80, 85), (85, 86, 84, 85), FLAT, FLAT]))
    assert st.halted and st.trades[0]["reason"] == "notbremse" and len(st.trades) == 1


def test_invalid_stop_side_skips_entry():
    st = run([FLAT, FLAT], {0: [("order", {"side": 1, "type": "market", "sl": 105})]})
    assert st.position is None and not st.trades


def test_helpers():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    assert efficiency_ratio(s, 3).iloc[-1] == pytest.approx(1.0)
    z = pd.Series([1.0, 2, 1, 2, 1, 2, 1])
    assert efficiency_ratio(z, 4).iloc[-1] == pytest.approx(0.0)
    idx = pd.date_range("2024-01-01 23:00", periods=4, freq="30min", tz="UTC")
    df = pd.DataFrame({"high": [10, 10, 20, 20.0], "low": [10, 10, 20, 20.0], "close": [10, 10, 20, 20.0],
                       "volume": [1, 1, 1, 3.0]}, index=idx)
    v = session_vwap(df)
    assert v.iloc[1] == 10 and v.iloc[2] == 20  # neuer Tag startet neu


def test_asia_breakout_range_and_single_trade():
    df = make_ohlc(96 * 20, seed=2, freq="15min", vol=0.003)
    p = AsiaBreakout().prepare(df)
    asia = p.index.hour < 8
    assert p.loc[asia, "r_hi"].isna().all()  # in der Asia-Zeit gibt es noch keine Spanne
    day = p.index[40].normalize()
    sub = df[(df.index.normalize() == day) & (df.index.hour < 8)]
    assert p.loc[p.index[40], "r_hi"] == sub["high"].max()
    assert (p["first_break"].groupby(p.index.normalize()).sum() <= 1).all()
    st = FxSimulator(AsiaBreakout(), FxCosts(), FxRisk()).run(p)
    per_day = pd.Series([t["entry_time"][:10] for t in st.trades]).value_counts()
    assert st.trades and per_day.max() == 1
    for t in st.trades:  # spätestens zum Tagesende raus
        assert t["exit_time"][:10] == t["entry_time"][:10] or t["exit_time"][11:16] == "00:00"


FREQ = {"15m": "15min", "5m": "5min"}


@pytest.mark.parametrize("name", list(HARDCORE))
def test_no_lookahead_in_indicators(name):
    s = HARDCORE[name]()
    df = make_ohlc(1500, seed=4, freq=FREQ[s.timeframe], vol=0.004)
    df["volume"] = np.random.default_rng(1).lognormal(0, 1, len(df))
    full = s.prepare(df)
    for k in (400, 977, 1400):
        pd.testing.assert_frame_equal(s.prepare(df.iloc[:k]), full.iloc[:k])


@pytest.mark.parametrize("name", list(HARDCORE))
def test_future_changes_do_not_alter_past_and_invariants(name):
    s = HARDCORE[name]()
    rng = np.random.default_rng(3)
    df = make_ohlc(3000, seed=6, freq=FREQ[s.timeframe], vol=0.004)
    df["volume"] = rng.lognormal(0, 1, len(df))
    other = make_ohlc(3000, seed=60, freq=FREQ[s.timeframe], vol=0.004)
    other["volume"] = rng.lognormal(0, 1, len(df))
    k = 2000
    scale = df["close"].iloc[k - 1] / other["open"].iloc[k]
    alt = pd.concat([df.iloc[:k], other.iloc[k:].assign(**{c: other[c].iloc[k:] * scale
                                                            for c in ("open", "high", "low", "close")})])
    a = FxSimulator(s, FxCosts(), FxRisk()).run(s.prepare(df))
    b = FxSimulator(s, FxCosts(), FxRisk()).run(s.prepare(alt))
    cut = df.index[k - 1].isoformat()
    assert [e for e in a.equity if e[0] <= cut] == [e for e in b.equity if e[0] <= cut]
    assert [t for t in a.trades if t["exit_time"] <= cut] == [t for t in b.trades if t["exit_time"] <= cut]
    eq = np.array([e[1] for e in a.equity])
    assert (eq > 0).all()
    for t in a.trades:
        if t["reason"] == "stop":
            assert t["r_multiple"] > -3  # Stop-Verlust bleibt im Rahmen (Lücken möglich)


@pytest.mark.parametrize("name", list(HARDCORE))
def test_chunked_run_with_json_state_equals_full_run(name):
    s = HARDCORE[name]()
    df = make_ohlc(2500, seed=9, freq=FREQ[s.timeframe], vol=0.004)
    df["volume"] = np.random.default_rng(2).lognormal(0, 1, len(df))
    p = s.prepare(df)
    full = FxSimulator(s, FxCosts(), FxRisk()).run(p)
    st = None
    for cut in (700, 1300, 1301, 2500):
        st = FxState.from_dict(json.loads(json.dumps(st.to_dict()))) if st else None
        st = FxSimulator(s, FxCosts(), FxRisk(), state=st).run(p.iloc[:cut])
    assert len(st.trades) == len(full.trades)
    assert st.cash == pytest.approx(full.cash, rel=1e-12)
    assert [tuple(e) for e in st.equity] == [tuple(e) for e in full.equity]


def test_hardcore_backtest_main_runs(tmp_path, monkeypatch):
    from bots.hardcore import backtest as HB
    rng = np.random.default_rng(5)

    def fake(symbol, interval):
        df = make_ohlc(4000, seed=hash(symbol) % 100, freq=FREQ[interval], vol=0.004)
        df["volume"] = rng.lognormal(0, 1, len(df))
        return df

    monkeypatch.setattr(HB, "load", fake)
    monkeypatch.setattr(HB, "OUT", tmp_path)
    HB.main()
    res = json.loads((tmp_path / "hardcore.json").read_text())
    assert len(res) == len(HARDCORE) * len(HB.SYMBOLS) * len(HB.COSTS)
    traded = {r["strategy"] for r in res if r["all"]["trades"] > 0}
    assert traded == set(HARDCORE), traded  # jede Strategie handelt wirklich
