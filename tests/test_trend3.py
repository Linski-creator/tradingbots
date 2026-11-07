"""Trendfolger v3: Wochenende, Verlust-Pause, Traum-Trade-Stop, News-Sperre, Lauf."""

import json

import numpy as np
import pandas as pd
import pytest

from bots.hardcore.engine import FxCosts, FxRisk, FxSimulator
from bots.metals import trend3 as T3
from tests.conftest import bars, make_ohlc


def trending(seed=1, freq="1h", n=4000):
    up = make_ohlc(n // 2, seed=seed, freq=freq, vol=0.004, drift=0.0007, start=2000)
    dn = make_ohlc(n // 2, seed=seed + 1, freq=freq, vol=0.004, drift=-0.0007, start=up["close"].iloc[-1])
    dn.index = up.index + pd.Timedelta(freq) * (n // 2)
    return pd.concat([up, dn])


@pytest.mark.parametrize("tf,freq", [("15m", "15min"), ("1h", "1h"), ("4h", "4h")])
def test_causal_and_runs(tf, freq):
    s = T3.TrendFollowerV3(tf)
    df = trending(freq=freq)
    p = s.prepare(df)
    pd.testing.assert_frame_equal(s.prepare(df.iloc[:3000]), p.iloc[:3000])
    st = FxSimulator(s, FxCosts(), FxRisk(min_stop=0.0005, max_leverage=5)).run(p)
    assert st.trades and {t["side"] for t in st.trades} == {"long", "short"}


def test_no_position_over_weekend_and_no_friday_late_entries():
    s = T3.TrendFollowerV3("1h")
    p = s.prepare(trending(seed=5))
    st = FxSimulator(s, FxCosts(), FxRisk(min_stop=0.0005, max_leverage=5)).run(p)
    for t in st.trades:
        a, b = pd.Timestamp(t["entry_time"]), pd.Timestamp(t["exit_time"])
        days = pd.date_range(a.normalize(), b.normalize(), freq="D")
        assert not any(d.weekday() == 5 for d in days), t  # nie über einen Samstag gehalten
        assert not (a.weekday() == 4 and a.hour >= 15), t   # kein Einstieg Freitag spät


def test_dream_trade_trails_and_can_exceed_3r():
    s = T3.TrendFollowerV3("1h")
    p = s.prepare(trending(seed=11, n=6000))
    st = FxSimulator(s, FxCosts(), FxRisk(min_stop=0.0005, max_leverage=5)).run(p)
    rs = [t["r_multiple"] for t in st.trades if t["r_multiple"] is not None]
    assert max(rs) > 3, max(rs)  # Traum-Trades ohne festes Ziel
    reasons = {t["reason"] for t in st.trades}
    assert "take_profit" not in reasons and {"stop", "trendbruch"} & reasons


def test_loss_pause_logic():
    s = T3.TrendFollowerV3("1h")
    now = pd.Timestamp("2025-03-05 12:00", tz="UTC")
    loss = lambda side, h: {"side": side, "pnl": -5.0, "exit_time": (now - pd.Timedelta(hours=h)).isoformat()}
    win = lambda side, h: {"side": side, "pnl": 5.0, "exit_time": (now - pd.Timedelta(hours=h)).isoformat()}
    assert s._paused("long", [loss("long", 10), loss("long", 2)], now)
    assert not s._paused("long", [loss("long", 30), loss("long", 25)], now)       # Pause abgelaufen
    assert not s._paused("long", [loss("long", 10), win("long", 2)], now)         # letzter war Gewinn
    assert not s._paused("short", [loss("long", 10), loss("long", 2)], now)       # andere Richtung frei


def test_news_block_cancels_entry():
    class Once:
        def on_bar(self, i, a, pos):
            return [("order", {"side": 1, "type": "market", "sl": 90})] if i == 0 else []

    df = bars([(100, 100, 100, 100)] * 4, freq="1h")
    st = FxSimulator(Once(), FxCosts(0, 0, 0, 0), FxRisk(max_leverage=50),
                     entry_block=lambda ts: ts == df.index[1]).run(df)
    assert st.position is None and not st.trades


def test_main_runs(tmp_path, monkeypatch):
    def fake(sym, tf):
        df = trending(seed=len(sym), freq=T3.TF[tf], n=3000)
        df["volume"] = 1.0
        return df

    monkeypatch.setattr(T3, "load", fake)
    monkeypatch.setattr(T3, "OUT", tmp_path)
    monkeypatch.setattr(T3, "JOBS", [("GC=F", "1h", None), ("PAXGUSDT", "4h", "2021-01-01")])
    T3.main()
    rows = json.loads((tmp_path / "trend3.json").read_text())
    assert len(rows) == 2 * len(T3.MODES) * 2
