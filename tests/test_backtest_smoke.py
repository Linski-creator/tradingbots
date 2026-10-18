"""Kompletter Backtest-Lauf mit erfundenen Kursen: prüft nur, dass der
Ablauf fehlerfrei durchläuft und alle Ergebnisse schreibt."""

import json

import numpy as np
import pandas as pd

from bots import backtest as B
from tests.conftest import make_ohlc


def test_backtest_main_runs(tmp_path, monkeypatch):
    syms = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "PAXGUSDT"]
    fake = {}
    for i, s in enumerate(syms):
        fake[(s, "1h")] = make_ohlc(1500, seed=i, freq="1h", vol=0.01)
        fake[(s, "4h")] = make_ohlc(1200, seed=i + 10, freq="4h", vol=0.015)
        fake[(s, "1d")] = make_ohlc(900, seed=i + 20, freq="1D")
    fake[("GOLD", "1d")] = make_ohlc(900, seed=99, freq="1D", vol=0.01)
    idx = pd.date_range("2020-01-01", periods=900 * 3, freq="8h", tz="UTC")
    funding = pd.Series(np.where(np.arange(len(idx)) % 60 < 40, 0.0001, -0.00002), index=idx)

    monkeypatch.setattr(B, "MARKETS", syms)
    monkeypatch.setattr(B, "OUT", tmp_path)
    monkeypatch.setattr(B, "load_all", lambda: fake)
    monkeypatch.setattr(B.D, "binance_funding_history", lambda sym: funding)
    B.main()

    res = json.loads((tmp_path / "results.json").read_text())
    ports = [r for r in res if r["strategy"] == "portfolio"]
    assert len(ports) == 3 and all(p["all"]["trades"] == 0 for p in ports)
    assert all("all" in r for r in res)
    corr = json.loads((tmp_path / "correlation.json").read_text())
    assert len(corr) == len(B.PORTFOLIO)
