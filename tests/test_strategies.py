"""Strategien: kein Blick in die Zukunft, Regeln wie beschrieben,
Invarianten auf vielen Zufallsverläufen."""

import numpy as np
import pandas as pd
import pytest

from bots.engine import Costs, RiskConfig, Simulator
from bots.portfolio import (DualMomentum, FundingArbitrage, FundingConfig, run_funding,
                            run_rotation)
from bots.strategies import SINGLE_MARKET, DonchianBreakout, EMAPullback, RSI2MeanReversion
from tests.conftest import bars, make_ohlc

FREQ = {"1d": "1D", "4h": "4h", "1h": "1h"}
BPY = {"1d": 365, "4h": 2190, "1h": 8760}


@pytest.mark.parametrize("name", list(SINGLE_MARKET))
def test_indicators_have_no_lookahead(name):
    """Indikatorwerte bis Bar k dürfen sich nicht ändern, wenn später
    weitere Bars dazukommen."""
    cls = SINGLE_MARKET[name]
    s = cls()
    df = make_ohlc(700, seed=5, freq=FREQ[s.timeframe])
    full = s.prepare(df, BPY[s.timeframe])
    for k in (250, 401, 650):
        part = s.prepare(df.iloc[:k], BPY[s.timeframe])
        pd.testing.assert_frame_equal(part, full.iloc[:k])


@pytest.mark.parametrize("name", list(SINGLE_MARKET))
def test_future_bars_do_not_change_past_trades(name):
    """Werden die Kurse NACH Bar k beliebig verändert, müssen alle Trades
    und die Equity bis Bar k identisch bleiben."""
    cls = SINGLE_MARKET[name]
    s = cls()
    df = make_ohlc(900, seed=9, freq=FREQ[s.timeframe])
    k = 600
    other = make_ohlc(900, seed=99, freq=FREQ[s.timeframe])
    scale = df["close"].iloc[k - 1] / other["open"].iloc[k]
    altered = pd.concat([df.iloc[:k], other.iloc[k:] * scale])
    altered["volume"] = 1.0

    def run(d):
        return Simulator(s, Costs(), RiskConfig()).run(s.prepare(d, BPY[s.timeframe]))

    a, b = run(df), run(altered)
    cut = df.index[k - 1].isoformat()
    assert [e for e in a.equity if e[0] <= cut] == [e for e in b.equity if e[0] <= cut]
    assert [t for t in a.trades if t["exit_time"] <= cut] == [t for t in b.trades if t["exit_time"] <= cut]


@pytest.mark.parametrize("name", list(SINGLE_MARKET))
@pytest.mark.parametrize("seed", range(12))
def test_invariants_on_random_paths(name, seed):
    cls = SINGLE_MARKET[name]
    s = cls()
    df = make_ohlc(1500, seed=seed, freq=FREQ[s.timeframe], drift=0.0, vol=0.02 + seed * 0.003)
    risk = RiskConfig(halt_drawdown=None)
    st = Simulator(s, Costs(), risk).run(s.prepare(df, BPY[s.timeframe]))
    eq = np.array([e[1] for e in st.equity])
    assert (eq > 0).all()
    assert st.cash > -1e-9
    for t in st.trades:
        assert t["exit_time"] >= t["entry_time"]
        if s.sizing == "risk" and t["r_multiple"] is not None and t["reason"] == "stop":
            # Ohne Kurslücke verliert ein Stop-Trade höchstens ~1 R
            assert t["r_multiple"] > -3
    if s.sizing == "risk":
        # Verlust je Trade bleibt klein relativ zum Konto (1 %-Regel, Lücken möglich)
        worst = min([t["pnl"] for t in st.trades], default=0)
        assert worst > -0.05 * max(eq)


def test_donchian_rules():
    s = DonchianBreakout(entry_n=3, exit_n=2, atr_n=2, atr_mult=2.0)
    rows = [(10, 11, 9, 10)] * 4 + [(10, 13, 10, 12.5), (12.5, 13, 12, 12.6), (12.6, 12.7, 8.5, 9), (9, 9.5, 8.8, 9.2)]
    df = s.prepare(bars(rows), 365)
    # Bar 4 schließt über dem 3-Bar-Hoch (11) der Vorbars -> Signal
    assert df["upper"].iloc[4] == 11 and df["close"].iloc[4] > df["upper"].iloc[4]
    st = Simulator(s, Costs(0, 0), RiskConfig(enabled=False)).run(df)
    t = st.trades[0]
    # Einstieg zum Open nach dem Signal; Stop 12,5 - 2*ATR(2)=7,5 nicht getroffen;
    # Schluss 9 unter dem 2-Bar-Tief 10 -> Ausstieg zum nächsten Open (9)
    assert t["entry_price"] == 12.5 and t["exit_price"] == 9 and t["reason"] == "kanal_tief"


def test_rsi2_every_trade_follows_rules():
    s = RSI2MeanReversion()
    reasons = set()
    for seed in range(6):
        df = s.prepare(make_ohlc(2500, seed=seed, drift=0.0008, vol=0.03), 365)
        st = Simulator(s, Costs(0, 0), RiskConfig()).run(df)
        pos = {ts.isoformat(): i for i, ts in enumerate(df.index)}
        for t in st.trades:
            sig = df.iloc[pos[t["entry_time"]] - 1]  # Signal-Bar = Bar vor dem Einstieg
            assert sig["rsi"] < 10 and sig["close"] > sig["sma"]
            assert t["entry_price"] == df["open"].iloc[pos[t["entry_time"]]]
            exit_sig = df.iloc[pos[t["exit_time"]] - 1]
            if t["reason"] == "rsi_hoch":
                assert exit_sig["rsi"] > 70
            elif t["reason"] == "zeitstop":
                assert t["bars_held"] == s.max_bars
            else:
                assert t["reason"] == "stop"
            reasons.add(t["reason"])
    assert {"rsi_hoch", "zeitstop", "stop"} <= reasons


def test_pullback_sets_3r_target():
    s = EMAPullback()
    df = make_ohlc(3000, seed=4, freq="1h", drift=0.0004, vol=0.006)
    st = Simulator(s, Costs(0, 0), RiskConfig(enabled=False)).run(s.prepare(df, 8760))
    tps = [t for t in st.trades if t["reason"] == "take_profit"]
    assert tps, "mindestens ein Ziel sollte erreicht werden"
    for t in tps:
        assert t["r_multiple"] >= 3.0 - 1e-9


def test_dual_momentum_picks_strongest_and_goes_cash():
    idx = pd.date_range("2022-01-02", periods=400, freq="1D", tz="UTC")  # Start Sonntag
    up = np.linspace(100, 300, 400)
    flat = np.linspace(100, 120, 400)
    down = np.linspace(300, 100, 400)

    def mk(c):
        return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=idx)

    s = DualMomentum(lookback=30, vol_n=10)
    data = s.prepare({"A": mk(up), "B": mk(flat)}, 365)
    st = run_rotation(s, data, Costs(0, 0), RiskConfig(enabled=False))
    assert st.holding == "A"
    data = s.prepare({"A": mk(down), "B": mk(down * 0.9)}, 365)
    st = run_rotation(s, data, Costs(0, 0), RiskConfig(enabled=False))
    assert st.holding is None and st.cash == pytest.approx(1000)


def test_funding_collects_positive_and_exits_on_negative():
    idx = pd.date_range("2023-01-01", periods=60, freq="8h", tz="UTC")
    rates = pd.Series([0.0001] * 30 + [-0.0001] * 30, index=idx)
    cfg = FundingConfig(avg_n=3, slippage=0, spot_fee=0, perp_fee=0)
    st = run_funding(FundingArbitrage(cfg), rates)
    # Einstieg nach 3 Zahlungen, danach 27 positive Zahlungen à 0,01 % auf 500 €
    # Ausstieg, sobald der 3er-Schnitt <= 0 ist (nach 2 negativen Zahlungen)
    t = st.trades[0]
    expected = 500 * 0.0001 * 27 - 500 * 0.0001 * 2
    assert t["funding"] == pytest.approx(expected)
    assert not st.active
