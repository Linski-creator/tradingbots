"""Trendfolger v2: ADX, höhere Zeitebene ohne Blick in die Zukunft, Regeln, Lauf."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from bots import indicators as ind
from bots.hardcore.engine import FxCosts, FxRisk, FxSimulator
from bots.metals import trend as T
from tests.conftest import bars, make_ohlc


def ref_adx(h, l, c, n):
    """Wilder-ADX, bewusst schlicht als Schleife."""
    N = len(c)
    tr, pdm, mdm = [math.nan] * N, [math.nan] * N, [math.nan] * N
    for i in range(1, N):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0

    def wil(x, start):
        out = [math.nan] * N
        if start + n > N:
            return out
        out[start + n - 1] = sum(x[start:start + n]) / n
        for i in range(start + n, N):
            out[i] = (out[i - 1] * (n - 1) + x[i]) / n
        return out

    atr, sp, sm = wil(tr, 1), wil(pdm, 1), wil(mdm, 1)
    dx = [math.nan] * N
    for i in range(N):
        if not math.isnan(atr[i]):
            p, m = 100 * sp[i] / atr[i], 100 * sm[i] / atr[i]
            dx[i] = 100 * abs(p - m) / (p + m) if p + m > 0 else 0.0
    first = next(i for i in range(N) if not math.isnan(dx[i]))
    return wil(dx, first)


def test_adx_matches_reference():
    df = make_ohlc(300, seed=3, freq="1h")
    got = ind.adx(df, 14).to_numpy()
    exp = np.array(ref_adx(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), 14))
    assert np.array_equal(np.isnan(got), np.isnan(exp))
    m = ~np.isnan(got)
    np.testing.assert_allclose(got[m], exp[m], rtol=1e-9)
    assert ((got[m] >= 0) & (got[m] <= 100)).all()


def test_higher_timeframe_is_causal():
    df = make_ohlc(2000, seed=4, freq="1h")
    full = T.higher_tf_slope(df)
    for k in (700, 1301, 1999):
        part = T.higher_tf_slope(df.iloc[:k])
        pd.testing.assert_series_equal(part, full.iloc[:k], check_names=False)


@pytest.mark.parametrize("variant", list(T.VARIANTS))
def test_strategy_causal_and_trades_both_sides(variant):
    s = T.TrendFollower(**T.VARIANTS[variant])
    up = make_ohlc(3000, seed=8, freq="1h", vol=0.004, drift=0.0006, start=2000)
    down = make_ohlc(3000, seed=9, freq="1h", vol=0.004, drift=-0.0006, start=up["close"].iloc[-1])
    down.index = up.index + pd.Timedelta(hours=3000)
    df = pd.concat([up, down])
    p = s.prepare(df)
    pd.testing.assert_frame_equal(s.prepare(df.iloc[:4000]), p.iloc[:4000])
    st = FxSimulator(s, FxCosts(), FxRisk(max_leverage=5, min_stop=0.0005)).run(p)
    sides = {t["side"] for t in st.trades}
    assert sides == {"long", "short"}, sides
    for t in st.trades:  # Stop liegt immer auf der richtigen Seite
        assert (t["initial_stop"] < t["entry_price"]) if t["side"] == "long" else (t["initial_stop"] > t["entry_price"])


def test_breakeven_moves_stop_only_forward():
    class S(T.TrendFollower):
        def __init__(self):
            super().__init__(htf=False, adx=False, be=True)

    rows = [(100, 100, 100, 100)] * 3 + [(100, 106, 100, 105), (105, 105, 99.5, 101)]
    df = bars(rows, freq="1h").assign(long=False, short=False, swing_low=95.0, swing_high=105.0, atr=5.0)
    df.loc[df.index[0], "long"] = True
    st = FxSimulator(S(), FxCosts(0, 0, 0, 0), FxRisk(max_leverage=50, min_stop=0.0005)).run(df)
    t = st.trades[0]
    # Einstieg 100, Stop 94 (Risiko 6). Bar 3 erreicht +6 -> Stop auf 100; Bar 4 fällt auf 99,5 -> raus bei 100
    assert t["exit_price"] == pytest.approx(100) and t["reason"] == "stop"


def test_main_runs(tmp_path, monkeypatch):
    def fake(sym):
        df = make_ohlc(24 * 300, seed=len(sym), freq="1h", vol=0.004, drift=0.0003, start=2000)
        df["volume"] = 1.0
        return df

    monkeypatch.setattr(T, "load", fake)
    monkeypatch.setattr(T, "OUT", tmp_path)
    monkeypatch.setattr(T, "MARKETS", [("GC=F", None), ("PAXGUSDT", "2020-07-01")])
    T.main()
    rows = json.loads((tmp_path / "trend.json").read_text())
    assert len(rows) == 2 * len(T.VARIANTS) * 2
