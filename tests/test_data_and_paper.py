"""Datenaufbereitung und kompletter Paper-Lauf mit simulierten Datenabrufen."""

import json

import numpy as np
import pandas as pd
import pytest

from bots import data as D
from bots import paper
from tests.conftest import make_ohlc


def test_timestamp_units_ms_and_us():
    ms = pd.Series([1_700_000_000_000])
    us = pd.Series([1_700_000_000_000_000])
    assert D._to_ts(ms)[0] == D._to_ts(us)[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")


def test_clean_removes_bad_and_duplicates():
    idx = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-03", "2024-01-04"], utc=True)
    df = pd.DataFrame({"open": [10, 10, 10, 0, 10], "high": [11, 11, 11, 11, 5],
                       "low": [9, 9, 9, 9, 4], "close": [10, 10.5, 10.6, 10, 10], "volume": 1}, index=idx)
    out = D.clean(df)
    assert list(out.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True))
    assert out.loc["2024-01-01", "close"] == 10.6  # letzter Duplikat-Eintrag gewinnt


def test_align_drops_off_grid_candles():
    idx = pd.to_datetime(["2018-02-09 09:00", "2018-02-09 09:28:14.789", "2018-02-09 10:00"], utc=True, format="ISO8601")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    assert list(D.align(df, "1h").index) == [idx[0], idx[2]]
    days = pd.to_datetime(["2024-01-01", "2024-01-02 00:00:01"], utc=True)
    assert len(D.align(df.iloc[:2].set_axis(days), "1d")) == 1


def test_metrics_accept_mixed_iso_timestamps():
    from bots.metrics import summarize
    eq = [("2018-02-09T09:00:00+00:00", 1000.0), ("2018-02-09T09:28:14.789000+00:00", 1001.0),
          ("2018-02-10T09:00:00+00:00", 1002.0)]
    assert summarize(eq, [], 1000.0)["final"] == 1002.0


def test_drop_unclosed():
    idx = pd.date_range("2024-01-01 00:00", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
    out = D.drop_unclosed(df, "1h", now=pd.Timestamp("2024-01-01 02:30", tz="UTC"))
    assert len(out) == 2  # Kerze ab 02:00 ist noch nicht geschlossen


def test_resample_4h_only_complete_blocks():
    df = make_ohlc(4 * 6 + 2, freq="1h")
    df = df.drop(df.index[5])  # Lücke im zweiten Block
    out = D.resample_4h(df)
    assert pd.Timestamp("2020-01-01 04:00", tz="UTC") not in out.index
    first = df.iloc[:4]
    r = out.iloc[0]
    assert r["open"] == first["open"].iloc[0] and r["close"] == first["close"].iloc[-1]
    assert r["high"] == first["high"].max() and r["low"] == first["low"].min()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Paper-Lauf in einem leeren Ordner mit erfundenen Kursen (nur Test)."""
    clock = {"n": 0}
    base = {s: make_ohlc(3000, seed=i, freq="1h", drift=0.0002, vol=0.01)
            for i, s in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT", "PAXGUSDT"])}

    def fake_klines(symbol, interval, bars):
        end = 2000 + clock["n"]           # "Jetzt" rückt pro Lauf vor
        h = base[symbol].iloc[:end]
        if interval == "4h":
            h = D.resample_4h(h)
        elif interval == "1d":
            h = h.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                      "close": "last", "volume": "sum"})
            h = h.iloc[:-1]  # angefangener Tag ist nicht abgeschlossen
        return h.iloc[-bars:]

    def fake_funding(symbol, n):
        idx = pd.date_range("2020-01-01", periods=300, freq="8h", tz="UTC")
        s = pd.Series(np.where(np.arange(300) % 50 < 35, 0.0001, -0.00005), index=idx)
        return s.iloc[: 30 + clock["n"] // 8].iloc[-n:]

    monkeypatch.setattr(D, "recent_klines", fake_klines)
    monkeypatch.setattr(D, "recent_funding", fake_funding)
    monkeypatch.setattr(paper, "STATE", tmp_path / "state")
    monkeypatch.setattr(paper, "WARMUP_BARS", {"1h": 1000, "4h": 300, "1d": 60})
    import bots.dashboard as dash
    monkeypatch.setattr(dash, "DOCS", tmp_path / "docs")
    cfg = json.loads((paper.ROOT / "bots.json").read_text())
    # Tagesstrategien brauchen lange Historie; im Test kurze Fenster
    from bots import strategies as S
    monkeypatch.setitem(S.SINGLE_MARKET, "sma200", lambda: S.SMA200Filter(n=20))
    monkeypatch.setitem(S.SINGLE_MARKET, "rsi2", lambda: S.RSI2MeanReversion(trend=20))
    monkeypatch.setitem(S.SINGLE_MARKET, "tsmom", lambda: S.TimeSeriesMomentum(lookback=20, vol_n=10))
    monkeypatch.setitem(S.SINGLE_MARKET, "donchian", lambda: S.DonchianBreakout())
    monkeypatch.setitem(S.SINGLE_MARKET, "chandelier", lambda: S.ChandelierTrend(trend=100))
    cfg_path = tmp_path / "bots.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(paper, "CONFIG", cfg_path)
    return tmp_path, clock


def test_paper_runs_repeatedly_and_writes_dashboard(sandbox):
    tmp, clock = sandbox
    for k in range(40):
        clock["n"] = k * 24  # jeder Lauf einen Tag später
        assert paper.main() == 0
    states = list((tmp / "state").glob("*.json"))
    assert len(states) == 8
    n_trades = {p.stem: len(json.loads(p.read_text())["trades"]) for p in states}
    traded = [k for k, v in n_trades.items() if v > 0]
    assert len(traded) >= 5, n_trades  # der Test muss echte Trades durchspielen
    md = (tmp / "docs" / "DASHBOARD.md").read_text()
    assert "Paper-Trading Dashboard" in md and "Fehler" not in md
    assert (tmp / "docs" / "equity.svg").read_text().startswith("<svg")
    for p in states:
        st = json.loads(p.read_text())
        eq = [e[1] for e in st["equity"]]
        assert eq and min(eq) > 0
        ts = [e[0] for e in st["equity"]]
        assert ts == sorted(ts) and len(ts) == len(set(ts))  # keine Kerze doppelt


def test_paper_skipped_runs_equal_continuous_runs(sandbox):
    """Fallen Läufe aus, holt der Bot alles nach: Ergebnis wie ohne Ausfall."""
    tmp, clock = sandbox
    for k in range(0, 31):
        clock["n"] = k * 12
        paper.main()
    every = {p.name: json.loads(p.read_text()) for p in (tmp / "state").glob("*.json")}
    for p in (tmp / "state").glob("*.json"):
        p.unlink()
    for k in (0, 7, 8, 19, 30):  # viele Läufe ausgelassen
        clock["n"] = k * 12
        paper.main()
    skipped = {p.name: json.loads(p.read_text()) for p in (tmp / "state").glob("*.json")}
    for name in every:
        if name.startswith("funding"):
            continue  # Funding-Fenster ist im Test kürzer als die Pause
        a, b = every[name]["trades"], skipped[name]["trades"]
        assert len(a) == len(b), name
        for x, y in zip(a, b):
            assert (x["entry_time"], x["exit_time"], x["reason"]) == (y["entry_time"], y["exit_time"], y["reason"])
            assert x["pnl"] == pytest.approx(y["pnl"], rel=1e-6, abs=1e-9)
        # EMA/ATR starten je nach Datenfenster minimal anders: Abweichung < 0,0001 %
        assert every[name]["equity"][-1][1] == pytest.approx(skipped[name]["equity"][-1][1], rel=1e-6), name
