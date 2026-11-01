"""Dukascopy-Dekodierung und Nachbau der Rücksetzer-Strategie."""

import json
import lzma
import struct

import numpy as np
import pandas as pd
import pytest

from bots.hardcore.engine import FxCosts, FxRisk, FxSimulator
from bots.metals import backtest as MB
from bots.metals import dukascopy as DK
from tests.conftest import make_ohlc


def test_decode_bi5_records():
    recs = [(0, 2000_100, 2000_500, 1999_900, 2000_800, 1.5),     # 00:00
            (60, 2000_500, 2000_200, 2000_000, 2000_600, 0.0),     # 00:01, kein Umsatz -> raus
            (120, 2000_200, 2001_000, 2000_100, 2001_200, 2.0)]    # 00:02
    raw = lzma.compress(b"".join(struct.pack(">5if", *r) for r in recs), format=lzma.FORMAT_ALONE)
    day = pd.Timestamp("2024-03-05", tz="UTC")
    df = DK.decode(raw, day, 1000)
    assert list(df.index) == [day, day + pd.Timedelta(minutes=2)]
    first = df.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"]) == (2000.1, 2000.8, 1999.9, 2000.5)
    assert DK.decode(b"", day, 1000).empty


def test_resample_5m():
    idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame({"open": range(10), "high": range(1, 11), "low": range(10), "close": range(10),
                       "volume": 1.0}, index=idx).astype(float)
    r = DK.resample(df, "5min")
    assert len(r) == 2 and r.iloc[0]["open"] == 0 and r.iloc[0]["close"] == 4 and r.iloc[0]["high"] == 5


def test_trend_dip_signal_rules_and_causality():
    s = MB.TrendDip(1.5, 1.5)
    df = make_ohlc(3000, seed=21, freq="15min", vol=0.003, drift=0.0002)
    p = s.prepare(df)
    sig = p[p["signal"]]
    assert len(sig) > 5
    assert (sig["close"] > sig["ema"]).all() and (sig["rsi"] >= 35).all()
    assert (p["rsi"].shift(1)[p["signal"]] < 35).all()
    pd.testing.assert_frame_equal(s.prepare(df.iloc[:2000]), p.iloc[:2000])
    st = FxSimulator(s, FxCosts(), FxRisk(max_leverage=10, min_stop=0.0005)).run(p)
    assert st.trades and all(t["side"] == "long" for t in st.trades)
    for t in st.trades:
        if t["reason"] == "take_profit":
            assert t["exit_price"] >= t["tp"] - 1e-9  # Ziel wird zum Zielpreis (oder besser) erreicht


def test_bar_hours_ignores_weekend_gap():
    idx = pd.date_range("2024-01-05 23:00", periods=3, freq="15min", tz="UTC").append(
        pd.date_range("2024-01-08 00:00", periods=40, freq="15min", tz="UTC"))
    df = make_ohlc(len(idx), seed=1, freq="15min").set_axis(idx)

    class Hold:
        def on_bar(self, i, a, pos):
            return [("order", {"side": 1, "type": "market", "sl": a["close"][i] * 0.9})] if i == 0 else []

    st = FxSimulator(Hold(), FxCosts(0, 0, 0, funding_8h=0.0008), FxRisk(risk_per_trade=0.01)).run(df)
    pos = st.position
    # 42 Bars à 15 min gehalten = 10,5 h -> Haltekosten ~ 0,08 % * 10,5/8 vom Nominal
    assert pos["funding"] == pytest.approx(pos["qty"] * df["close"].iloc[1:].mean() * 0.0008 * 0.25 / 8 * 42,
                                           rel=0.1)


def test_metals_main_runs(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    base = {"XAUUSD": 2000, "XAGUSD": 25, "XPTUSD": 1000, "USDJPY": 140}

    def fake(sym, rule):
        freq = {"5min": "5min", "15min": "15min", "1h": "1h"}[rule]
        df = make_ohlc(3000, seed=len(sym), freq=freq, vol=0.002, drift=0.0001, start=base[sym])
        df.index = pd.date_range("2019-06-01", periods=3000, freq=freq, tz="UTC")
        df["volume"] = rng.lognormal(0, 1, len(df))
        return df

    monkeypatch.setattr(MB, "load", fake)
    monkeypatch.setattr(MB, "OUT", tmp_path)
    MB.main()
    res = json.loads((tmp_path / "metals.json").read_text())
    assert len(res["verdicts"]) == len(MB.TIMEFRAMES) * len(MB.EXITS)
    assert len(res["rows"]) == len(MB.SYMBOLS) * len(MB.TIMEFRAMES) * len(MB.EXITS) * 2
    assert any(r["selection"].get("trades", 0) > 0 for r in res["rows"])
