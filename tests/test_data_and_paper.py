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
    days = pd.to_datetime(["2024-01-01", "2024-01-02 00:00:01"], utc=True, format="ISO8601")
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


def bot_states(tmp):
    return [p for p in (tmp / "state").glob("*.json") if p.name != "team.json"]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Paper-Lauf in einem leeren Ordner mit erfundenen Kursen (nur Test)."""
    clock = {"n": 0}
    base = {s: make_ohlc(3000, seed=i, freq="1h", drift=0.0002, vol=0.01)
            for i, s in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT", "PAXGUSDT", "BNBUSDT"])}

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
    from bots import news
    monkeypatch.setattr(news, "fetch_events", lambda: None)  # Tests nie online
    monkeypatch.setattr(paper, "STATE", tmp_path / "state")
    monkeypatch.setattr(paper, "WARMUP_BARS", {"1h": 1000, "4h": 300, "1d": 60})
    import bots.dashboard as dash
    monkeypatch.setattr(dash, "DOCS", tmp_path / "docs")
    import bots.journal as jour
    monkeypatch.setattr(jour, "DOCS", tmp_path / "docs")
    import bots.webpage as web
    monkeypatch.setattr(web, "SITE", tmp_path / "site")
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
    states = bot_states(tmp)
    assert len(states) == 8
    n_trades = {p.stem: len(json.loads(p.read_text())["trades"]) for p in states}
    traded = [k for k, v in n_trades.items() if v > 0]
    assert len(traded) >= 5, n_trades  # der Test muss echte Trades durchspielen
    md = (tmp / "docs" / "DASHBOARD.md").read_text()
    assert "Paper-Trading Dashboard" in md and "Fehler" not in md
    assert (tmp / "docs" / "equity.svg").read_text().startswith("<svg")
    html = (tmp / "site" / "index.html").read_text()
    raw = html.split("const DATA = ", 1)[1].split(";\nconst REASONS", 1)[0]
    data = json.loads(raw.replace("<\\/", "</"))
    assert len(data["bots"]) == 8 and data["team"]["start"] == 7000
    team_bots = [b for b in data["bots"] if not b["compare"]]
    assert data["team"]["current"] == pytest.approx(sum(b["current"] for b in team_bots), abs=0.05)
    for b in data["bots"]:
        days = sorted(b["days"])
        if days:  # Tagesgewinne summieren sich zum Gesamtergebnis
            assert sum(v["pnl"] for v in b["days"].values()) == pytest.approx(
                b["days"][days[-1]]["value"] - 1000, abs=0.05)
        assert sum(v["trades"] for v in b["days"].values()) >= len(b["trades"])
    journal = (tmp / "docs" / "JOURNAL.md").read_text()
    assert "Trading-Journal" in journal and "Tagesprotokoll" in journal
    assert journal.count("- **Grund:**") == sum(n_trades.values())
    # jeder Einstiegsgrund ist ein lesbarer Satz, kein Rohdaten-Fallback
    for line in journal.splitlines():
        if line.startswith("- **Grund:**"):
            assert "ema_f " not in line and "upper " not in line and line.strip() != "- **Grund:** –", line
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
    every = {p.name: json.loads(p.read_text()) for p in bot_states(tmp)}
    for p in (tmp / "state").glob("*.json"):
        p.unlink()
    for k in (0, 7, 8, 19, 30):  # viele Läufe ausgelassen
        clock["n"] = k * 12
        paper.main()
    skipped = {p.name: json.loads(p.read_text()) for p in bot_states(tmp)}
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


def test_team_average_fills_start_cash_before_first_value():
    from bots.dashboard import team_average
    a = [("2024-01-01T00:00:00+00:00", 1000.0), ("2024-01-01T02:00:00+00:00", 1100.0)]
    b = [("2024-01-01T01:00:00+00:00", 900.0)]
    avg = dict(team_average([a, b], 1000.0))
    assert avg["2024-01-01T00:00:00+00:00"] == 1000.0           # b noch nicht gestartet
    assert avg["2024-01-01T01:00:00+00:00"] == 950.0            # (1000 + 900) / 2
    assert avg["2024-01-01T02:00:00+00:00"] == 1000.0           # (1100 + 900) / 2


def test_entry_reasons_readable_for_all_strategies():
    from bots.journal import entry_reason
    ctx = {"close": 100.0, "upper": 99.0, "lower": 90.0, "atr": 2.0, "ema_f": 101.0, "ema_s": 100.0,
           "ema_t": 95.0, "ema_m": 99.0, "mom": 0.12, "sma": 97.0, "rsi": 5.0, "vol": 0.5,
           "mom_BTCUSDT": 0.2, "mom_ETHUSDT": -0.1, "avg_rate": 0.0001}
    for strat in ("donchian", "chandelier", "tsmom", "sma200", "rsi2", "pullback", "emacross",
                  "dualmom", "funding"):
        text = entry_reason(strat, ctx)
        assert text and "{" not in text and "None" not in text, (strat, text)
    assert "BTCUSDT 20,0 %" in entry_reason("dualmom", ctx)
    assert entry_reason("donchian", {}) == "–"  # fehlender Kontext stürzt nicht ab


def test_news_parse_and_block_window():
    from bots import news
    raw = [{"title": "CPI m/m", "country": "USD", "date": "2026-10-20T08:30:00-04:00", "impact": "High"},
           {"title": "Retail", "country": "EUR", "date": "2026-10-20T10:00:00+02:00", "impact": "High"},
           {"title": "Kleinkram", "country": "USD", "date": "2026-10-20T10:00:00-04:00", "impact": "Low"},
           {"title": "kaputt", "country": "USD", "date": "kein datum", "impact": "High"}]
    ev = news.parse_events(raw)
    assert [e["title"] for e in ev] == ["CPI m/m"]
    assert ev[0]["time"] == pd.Timestamp("2026-10-20 12:30", tz="UTC")
    blocked = news.entry_blocker(ev)
    assert blocked(pd.Timestamp("2026-10-20 10:30", tz="UTC"))      # 2 h davor
    assert blocked(pd.Timestamp("2026-10-20 13:30", tz="UTC"))      # 1 h danach
    assert not blocked(pd.Timestamp("2026-10-20 10:29", tz="UTC"))
    assert not blocked(pd.Timestamp("2026-10-20 13:31", tz="UTC"))
    assert news.entry_blocker(None) is None and news.entry_blocker([]) is None


def test_entry_waits_during_news_then_executes():
    from bots.engine import Costs, Enter, RiskConfig, Simulator
    from tests.conftest import bars

    class Once:
        name, sizing = "once", "all"

        def __init__(self):
            self.i = -1

        def on_bar(self, v):
            self.i += 1
            return [Enter()] if self.i == 0 else []

    df = bars([(100, 100, 100, 100), (101, 101, 101, 101), (102, 102, 102, 102), (103, 103, 103, 103)],
              freq="1h")
    news_hour = df.index[1]
    sim = Simulator(Once(), Costs(0, 0), RiskConfig(), entry_block=lambda ts: ts == news_hour)
    st = sim.run(df)
    # Signal bei Bar 0, Bar 1 gesperrt -> Einstieg zum Open von Bar 2
    assert st.position.entry_time == df.index[2].isoformat() and st.position.entry_price == 102


def test_team_brake_closes_and_stops_all_team_bots(sandbox):
    tmp, clock = sandbox
    for k in range(25):
        clock["n"] = k * 24
        paper.main()
    (tmp / "state" / "team.json").write_text(json.dumps({"halted": True, "peak": 9999.0}))
    clock["n"] = 25 * 24
    paper.main()
    clock["n"] = 26 * 24
    paper.main()
    for p in bot_states(tmp):
        st = json.loads(p.read_text())
        if p.stem == "video-ema-btc":
            assert not st.get("halted")  # Vergleichs-Bot zählt nicht zum Team
            continue
        assert st["halted"], p.stem
        assert not st.get("position") and not st.get("holding") and not st.get("active"), p.stem
    assert "TEAM-NOTBREMSE AKTIV" in (tmp / "docs" / "DASHBOARD.md").read_text()


def test_update_team_triggers_only_with_complete_data():
    cfg = {"start_cash": 1000, "team_halt_drawdown": 0.2,
           "bots": [{"id": "a"}, {"id": "b"}, {"id": "v", "compare": True}]}
    rep = lambda i, v, c=False: {"bot": {"id": i, "compare": c}, "state": {"equity": [["t", v]]}}
    t = paper.update_team([rep("a", 1000), rep("b", 1000)], cfg, {})
    assert t["peak"] == 2000 and not t.get("halted")
    t = paper.update_team([rep("a", 800), rep("b", 800), rep("v", 1, True)], cfg, t)
    assert t["halted"]  # 1600 = 20 % unter 2000
    t2 = paper.update_team([rep("a", 100)], cfg, {"peak": 2000})  # b fehlt -> keine Entscheidung
    assert not t2.get("halted")


def test_funding_halt_closes_position():
    from bots.portfolio import FundingArbitrage, FundingConfig, FundingState, run_funding
    idx = pd.date_range("2023-01-01", periods=20, freq="8h", tz="UTC")
    rates = pd.Series([0.0002] * 20, index=idx)
    fa = FundingArbitrage(FundingConfig(avg_n=3))
    st = run_funding(fa, rates.iloc[:10])
    assert st.active
    st.halted = True
    st = run_funding(fa, rates, state=st)
    assert not st.active and st.trades[-1]["reason"] == "notbremse"
    assert len(st.trades) == 1  # danach kein neuer Einstieg


def test_daily_summary_counts_and_values():
    from bots.webpage import daily_summary
    eq = [("2026-10-01T00:00:00+00:00", 1000.0), ("2026-10-02T00:00:00+00:00", 1010.0),
          ("2026-10-04T00:00:00+00:00", 1005.0)]
    trades = [{"entry_time": "2026-10-01T00:00:00+00:00", "exit_time": "2026-10-03T05:00:00+00:00"}]
    d = daily_summary(eq, trades, 1000.0, ["2026-10-04T00:00:00+00:00"])
    assert d["2026-10-01"]["opened"] == 1 and d["2026-10-01"]["trades"] == 0
    assert d["2026-10-03"]["trades"] == 1 and d["2026-10-03"]["value"] == 1010.0  # Lücke: Vortageswert
    assert d["2026-10-04"]["opened"] == 1 and d["2026-10-04"]["pnl"] == -5.0
    assert sum(v["pnl"] for v in d.values()) == pytest.approx(5.0)
