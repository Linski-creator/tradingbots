"""Walk-Forward: Fenster, keine Nutzung von Testdaten bei der Auswahl, Lauf."""

import json

import numpy as np
import pandas as pd

from bots.metals import walkforward as WF
from tests.conftest import make_ohlc


def test_windows_do_not_overlap_tests():
    idx = pd.date_range("2024-01-01", periods=24 * 500, freq="1h", tz="UTC")
    w = WF.windows(idx, "200D", "50D")
    assert len(w) >= 5
    for (a, b, c), nxt in zip(w, w[1:]):
        assert a < b < c and b - a == pd.Timedelta("200D")
        assert nxt[1] == c  # nächstes Testfenster beginnt, wo das letzte endete


def test_selection_ignores_test_window(monkeypatch):
    """Wird ein Kurs NACH dem Training verändert, darf das die Auswahl nicht ändern."""
    df = make_ohlc(24 * 300, seed=5, freq="1h", vol=0.004, drift=0.0002, start=2000)
    df["volume"] = 1.0
    monkeypatch.setattr(WF, "GRID", {"ema": [100, 150], "level": [35], "sl": [1.5, 2.0], "tp": [1.0, 2.0]})
    monkeypatch.setattr(WF, "DEFAULT", (150, 35, 2.0, 1.0))
    monkeypatch.setattr(WF, "MIN_TRADES", 3)
    r1 = WF.walk_forward(df, "GC=F", "120D", "60D")
    cut = df.index[0] + pd.Timedelta("120D")
    alt = df.copy()
    alt.loc[alt.index >= cut, ["open", "high", "low", "close"]] *= 1.0  # gleich
    other = make_ohlc(len(df), seed=77, freq="1h", vol=0.004, start=2000).set_axis(df.index)
    scale = df.loc[df.index < cut, "close"].iloc[-1] / other.loc[other.index >= cut, "open"].iloc[0]
    alt.loc[alt.index >= cut, ["open", "high", "low", "close"]] = other.loc[other.index >= cut,
                                                                            ["open", "high", "low", "close"]] * scale
    r2 = WF.walk_forward(alt, "GC=F", "120D", "60D")
    assert r1["windows_detail"][0]["combo"] == r2["windows_detail"][0]["combo"]


def test_main_runs(tmp_path, monkeypatch):
    def fake(sym):
        df = make_ohlc(24 * 400, seed=len(sym), freq="1h", vol=0.004, drift=0.0001, start=2000)
        df["volume"] = 1.0
        return df

    monkeypatch.setattr(WF, "load", fake)
    monkeypatch.setattr(WF, "OUT", tmp_path)
    monkeypatch.setattr(WF, "GRID", {"ema": [150], "level": [35], "sl": [1.5, 2.0], "tp": [1.0, 3.0]})
    monkeypatch.setattr(WF, "MARKETS", [("GC=F", "120D", "60D"), ("PAXGUSDT", "120D", "60D")])
    WF.main()
    res = json.loads((tmp_path / "walkforward.json").read_text())
    assert len(res) == 2 and all(r["windows"] >= 3 for r in res)
