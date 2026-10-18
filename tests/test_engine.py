"""Simulator: Ausführung, Gebühren, Stops, Sizing, Notbremse, Zustand."""

import json

import pandas as pd
import pytest

from bots.engine import Costs, Enter, Exit, RiskConfig, SetStop, SimState, Simulator
from tests.conftest import bars

NO_COST = Costs(fee=0.0, slippage=0.0)


class Scripted:
    """Gibt an vorgegebenen Bar-Nummern vorgegebene Signale."""
    name, sizing = "scripted", "risk"

    def __init__(self, script, sizing="risk"):
        self.script, self.sizing, self.i = script, sizing, -1

    def on_bar(self, v):
        self.i += 1
        return self.script.get(self.i, [])


def run(df, script, costs=NO_COST, risk=None, sizing="risk"):
    risk = risk or RiskConfig(enabled=False)
    sim = Simulator(Scripted(script, sizing), costs, risk, start_cash=1000.0)
    return sim.run(df)


def test_entry_at_next_open_and_exit_at_next_open():
    df = bars([(100, 101, 99, 100), (110, 112, 108, 111), (120, 121, 119, 120), (90, 91, 89, 90)])
    st = run(df, {0: [Enter()], 1: [Exit()]})
    t = st.trades[0]
    assert t["entry_price"] == 110  # Open der Bar nach dem Signal
    assert t["exit_price"] == 120
    assert st.cash == pytest.approx(1000 * 120 / 110)


def test_fees_and_slippage_exact():
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)])
    costs = Costs(fee=0.001, slippage=0.0005)
    st = run(df, {0: [Enter()], 1: [Exit()]}, costs=costs)
    buy = 100 * 1.0005
    qty = 1000 / (buy * 1.001)
    proceeds = qty * 100 * 0.9995 * 0.999
    assert st.cash == pytest.approx(proceeds, rel=1e-12)
    assert st.cash < 1000  # flacher Markt kostet nur Gebühren


def test_stop_intrabar_and_gap():
    # Stop bei 95 wird innerhalb der Bar getroffen
    df = bars([(100, 100, 100, 100), (100, 101, 94, 96)])
    st = run(df, {0: [Enter(stop=95)]})
    assert st.trades[0]["exit_price"] == 95 and st.trades[0]["reason"] == "stop"
    # Kurslücke: Bar öffnet bei 90 unter dem Stop -> Ausführung zum Open
    df = bars([(100, 100, 100, 100), (100, 101, 99, 100), (90, 91, 88, 89)])
    st = run(df, {0: [Enter(stop=95)]})
    assert st.trades[0]["exit_price"] == 90


def test_stop_wins_when_stop_and_target_in_same_bar():
    df = bars([(100, 100, 100, 100), (100, 120, 90, 100)])
    st = run(df, {0: [Enter(stop=95, rr=2)]})  # Ziel 110, Stop 95
    assert st.trades[0]["reason"] == "stop"


def test_take_profit():
    df = bars([(100, 100, 100, 100), (100, 102, 99, 101), (101, 111, 100, 105)])
    st = run(df, {0: [Enter(stop=95, rr=2)]})
    t = st.trades[0]
    assert t["reason"] == "take_profit" and t["exit_price"] == 110
    assert t["r_multiple"] == pytest.approx(2.0)


def test_no_entry_if_open_below_stop():
    df = bars([(100, 100, 100, 100), (90, 91, 89, 90)])
    st = run(df, {0: [Enter(stop=95)]})
    assert st.position is None and not st.trades and st.cash == 1000


def test_risk_sizing_loses_about_one_percent():
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 89, 90)])
    risk = RiskConfig(enabled=True, risk_per_trade=0.01)
    st = run(df, {0: [Enter(stop=90)]}, costs=Costs(), risk=risk)
    loss = 1000 - st.cash
    assert 9.0 < loss <= 10.0 + 1e-9  # höchstens 1 % inkl. Gebühren


def test_never_leveraged():
    # Sehr enger Stop würde > 100 % Einsatz verlangen -> auf 100 % gedeckelt
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100)])
    risk = RiskConfig(enabled=True, risk_per_trade=0.01)
    st = run(df, {0: [Enter(stop=99.99)]}, risk=risk)
    assert st.position.cost <= 1000 + 1e-9 and st.cash >= -1e-9


def test_vol_sizing():
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100)])
    risk = RiskConfig(enabled=True, target_vol=0.3)
    st = run(df, {0: [Enter(vol=0.6)]}, risk=risk, sizing="vol")
    assert st.position.cost == pytest.approx(500)


def test_stop_only_tightens():
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 93, 95)])
    st = run(df, {0: [Enter(stop=90)], 1: [SetStop(94)], 2: [SetStop(80)]})
    assert st.trades[0]["exit_price"] == 94


def test_halt_drawdown_closes_and_stops_trading():
    df = bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 80, 85),
               (84, 85, 83, 84), (84, 85, 83, 84), (84, 85, 83, 84)])
    risk = RiskConfig(enabled=True, max_exposure=1.0, halt_drawdown=0.10, target_vol=10)
    st = run(df, {0: [Enter(vol=1.0)], 4: [Enter(vol=1.0)]}, risk=risk, sizing="vol")
    assert st.halted
    assert st.trades[0]["reason"] == "notbremse"
    assert st.position is None and len(st.trades) == 1


def test_invalid_candle_raises():
    df = bars([(100, 99, 98, 100)])  # High unter Open
    with pytest.raises(ValueError):
        run(df, {})


def test_state_roundtrip_json_and_chunked_run_equal_full_run():
    from tests.conftest import make_ohlc
    from bots.strategies import DonchianBreakout
    df = make_ohlc(500, seed=3)
    strat = DonchianBreakout()
    prepared = strat.prepare(df, 365)
    full = Simulator(strat, Costs(), RiskConfig()).run(prepared)

    sim = Simulator(strat, Costs(), RiskConfig())
    st = sim.run(prepared.iloc[:200])
    for cut in (260, 333, 500):
        st = SimState.from_dict(json.loads(json.dumps(st.to_dict())))
        st = Simulator(strat, Costs(), RiskConfig(), state=st).run(prepared.iloc[:cut])
    assert st.trades == full.trades
    assert [tuple(e) for e in st.equity] == [tuple(e) for e in full.equity]
    assert st.cash == pytest.approx(full.cash, rel=1e-12)
