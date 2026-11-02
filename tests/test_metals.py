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
    base = {"GC=F": 2000, "SI=F": 25, "PL=F": 1000, "JPY=X": 140}

    base["PAXGUSDT"] = 2000

    def fake(sym, tf):
        freq = {"5m": "5min", "15m": "15min", "1h": "1h"}[tf]
        start = "2025-08-01" if sym != "PAXGUSDT" else "2023-11-01"
        df = make_ohlc(3000, seed=len(sym), freq=freq, vol=0.002, drift=0.0001, start=base[sym])
        df.index = pd.date_range(start, periods=3000, freq=freq, tz="UTC")
        df["volume"] = rng.lognormal(0, 1, len(df))
        return df

    monkeypatch.setattr(MB, "load", fake)
    monkeypatch.setattr(MB, "OUT", tmp_path)
    MB.main()
    res = json.loads((tmp_path / "metals.json").read_text())
    n_jobs = sum(len(g["jobs"]) for g in MB.GROUPS.values())
    n_variants = len({(g, tf) for g, spec in MB.GROUPS.items() for _, tf in spec["jobs"]}) * len(MB.EXITS)
    assert len(res["verdicts"]) == n_variants
    assert len(res["rows"]) == n_jobs * len(MB.EXITS) * 2
    assert any(r["selection"].get("trades", 0) > 0 for r in res["rows"])


def test_fetch_retries_then_gives_up(monkeypatch):
    calls = []

    class R:
        def __init__(self, code, content=b"x"):
            self.status_code, self.content = code, content

        def raise_for_status(self):
            if self.status_code >= 400:
                import requests
                raise requests.HTTPError(str(self.status_code))

    seq = [R(503), R(503), R(200, b"daten")]
    monkeypatch.setattr(DK.time, "sleep", lambda s: None)
    monkeypatch.setattr(DK.requests, "get", lambda *a, **k: calls.append(1) or seq.pop(0))
    assert DK._fetch("XAUUSD", pd.Timestamp("2024-01-02", tz="UTC")) == b"daten" and len(calls) == 3
    monkeypatch.setattr(DK.requests, "get", lambda *a, **k: R(503))
    assert DK._fetch("XAUUSD", pd.Timestamp("2024-01-02", tz="UTC")) is None
    monkeypatch.setattr(DK.requests, "get", lambda *a, **k: R(404))
    assert DK._fetch("XAUUSD", pd.Timestamp("2024-01-02", tz="UTC")) == b""


def test_month_hourly_uses_month_start(monkeypatch, tmp_path):
    recs = [(0, 2000_000, 2001_000, 1999_000, 2002_000, 5.0),
            (3600 * 24 * 3 + 7200, 2010_000, 2011_000, 2009_000, 2012_000, 4.0)]  # 4. Tag, 02:00
    raw = lzma.compress(b"".join(struct.pack(">5if", *r) for r in recs), format=lzma.FORMAT_ALONE)
    seen = []
    monkeypatch.setattr(DK, "_fetch", lambda sym, day, url=None: seen.append(url) or raw)
    monkeypatch.setattr(DK, "CACHE", tmp_path)
    df = DK.month_hourly("XAUUSD", pd.Period("2024-03", "M"))
    assert seen[0].endswith("/XAUUSD/2024/02/BID_candles_hour_1.bi5")  # Monat 0-basiert
    assert list(df.index) == [pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-03-04 02:00", tz="UTC")]
