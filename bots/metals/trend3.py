"""Trendfolger v3: viele Rücksetzer im Trend mitnehmen, Traum-Trades laufen lassen.

Gegenüber v2:
  - Ausstieg ohne festes Ziel ("Traum-Trade"): ab +1R Stop auf Einstieg, danach
    nachgezogener Stop 2,5 ATR unter dem höchsten Hoch seit Einstieg (Short gespiegelt)
  - Trendbruch: Schluss auf der falschen Seite der EMA 50 -> raus (auch früh, kleiner Verlust)
  - Verlust-Pause: nach 2 Verlusten in Folge in dieselbe Richtung 24 h keine Trades
    in diese Richtung (kein Nachkaufen, kein "jetzt erst recht")
  - Wochenende: ab Freitag 14 Uhr UTC keine neuen Trades, offene Trades werden vor
    20 Uhr UTC geschlossen
  - News-Sperre im Live-Betrieb (über den Simulator)

Einstieg wie v2 (EMA-Staffelung, höhere Zeitebene, ADX, RSI-Ausreißer + Umkehrkerze,
Stop hinter dem Rücksetzer). Parameter vorab festgelegt, keine Optimierung.

Aufruf: python -m bots.metals.trend3   (schreibt results/trend3.json)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D
from .. import indicators as ind
from ..hardcore.engine import FxRisk, FxSimulator
from ..hardcore.search import split
from .backtest import COSTS, stress, yearly
from .trend import higher_tf_slope

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[2] / "results"

TF = {"15m": "15min", "1h": "1h", "4h": "4h"}
HTF = {"15m": "4h", "1h": "4h", "4h": "1D"}
MODES = {"sicher": dict(risk_per_trade=0.01, max_leverage=5, halt_drawdown=0.20),
         "krass": dict(risk_per_trade=0.03, max_leverage=10, halt_drawdown=0.50)}
TRAIL_ATR = 2.5
PAUSE = pd.Timedelta(hours=24)


class TrendFollowerV3:
    wants_history = True

    def __init__(self, tf: str = "1h"):
        self.tf = tf
        self.step = pd.Timedelta(TF[tf])

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        c, h, l = out["close"], out["high"], out["low"]
        e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
        rsi = ind.rsi(c, 14)
        out["atr"] = ind.atr(out, 14)
        out["ema50"] = e50
        slope = higher_tf_slope(out, rule=HTF[self.tf])
        a = ind.adx(out, 14)
        fading = ((a < a.shift(1)).astype(int).rolling(5, min_periods=5).sum() == 5) & (a < 25)
        strong = (a > 20) & ~fading
        up = (c > e200) & (e50 > e200) & (slope > 0) & strong
        dn = (c < e200) & (e50 < e200) & (slope < 0) & strong
        out["long"] = (up & (rsi.rolling(5, min_periods=5).min() < 40) & (c > h.shift(1))).fillna(False)
        out["short"] = (dn & (rsi.rolling(5, min_periods=5).max() > 60) & (c < l.shift(1))).fillna(False)
        out["swing_low"] = l.rolling(5, min_periods=5).min()
        out["swing_high"] = h.rolling(5, min_periods=5).max()
        close_t = out.index + self.step                      # Zeitpunkt, an dem die Kerze schließt
        fri = close_t.weekday == 4
        cutoff = close_t.normalize() + pd.Timedelta(hours=20)
        out["weekend_exit"] = (fri & (close_t + self.step >= cutoff)) | (close_t.weekday >= 5)
        out["no_entry"] = (fri & (close_t.hour >= 14)) | (close_t.weekday >= 5)
        out["ts_ns"] = out.index.as_unit("ns").asi8  # Index kann us-Auflösung haben
        for k in ("long", "short", "weekend_exit", "no_entry"):
            out[k] = out[k].astype(bool)
        return out

    # ------------------------------------------------------------------
    def _paused(self, side: str, trades: list, now: pd.Timestamp) -> bool:
        same = [t for t in trades if t["side"] == side][-2:]
        if len(same) < 2 or any(t["pnl"] >= 0 for t in same):
            return False
        return now - pd.Timestamp(same[-1]["exit_time"]) < PAUSE

    def on_bar(self, i, a, pos, trades=()):
        c, atr = a["close"][i], a["atr"][i]
        if pos is not None:
            side = pos["side"]
            if a["weekend_exit"][i]:
                return [("exit", "wochenende")]
            if (side == 1 and c < a["ema50"][i]) or (side == -1 and c > a["ema50"][i]):
                return [("exit", "trendbruch")]
            if not (math.isfinite(atr) and atr > 0):
                return []
            entry, risk = pos["entry_price"], abs(pos["entry_price"] - pos["initial_sl"])
            e = int(np.searchsorted(a["ts_ns"], pd.Timestamp(pos["entry_time"]).value))
            if side == 1:
                best = float(np.max(a["high"][e:i + 1]))
                if best - entry >= risk:
                    return [("stop", max(entry, best - TRAIL_ATR * atr))]
            else:
                best = float(np.min(a["low"][e:i + 1]))
                if entry - best >= risk:
                    return [("stop", min(entry, best + TRAIL_ATR * atr))]
            return []
        if a["no_entry"][i] or not (math.isfinite(atr) and atr > 0):
            return []
        now = pd.Timestamp(int(a["ts_ns"][i]), tz="UTC")
        for side, name, flag, stop in ((1, "long", "long", a["swing_low"][i] - 0.2 * atr),
                                       (-1, "short", "short", a["swing_high"][i] + 0.2 * atr)):
            if not a[flag][i] or not math.isfinite(stop) or self._paused(name, trades, now):
                continue
            if 0.5 * atr <= abs(c - stop) <= 3 * atr:
                return [("order", {"side": side, "type": "market", "sl": stop,
                                   "tag": "trend_rücksetzer_v3",
                                   "context": {"close": float(c), "atr": float(atr), "ema50": float(a["ema50"][i])}})]
        return []


def load(sym: str, tf: str) -> pd.DataFrame:
    if sym == "PAXGUSDT":
        return D.binance_vision_klines(sym, tf, start="2020-01")
    h = D.yahoo_hourly(sym)
    if tf == "1h":
        return h
    g = h.resample("4h", label="left", closed="left", origin="epoch")
    return g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


JOBS = [("PAXGUSDT", tf, "2024-01-01") for tf in ("15m", "1h", "4h")] + \
       [(s, tf, None) for s in ("GC=F", "SI=F", "PL=F", "JPY=X") for tf in ("1h", "4h")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    rows = []
    for sym, tf, cut in JOBS:
        df = load(sym, tf)
        cut = cut or (df.index[-1] - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        strat = TrendFollowerV3(tf)
        prepared = strat.prepare(df)
        weeks = max((df.index[-1] - df.index[0]).days / 7, 1)
        for mode, rk in MODES.items():
            for cname, costs in (("normal", COSTS[sym]), ("stress", stress(COSTS[sym]))):
                st = FxSimulator(strat, costs, FxRisk(min_stop=0.0005, **rk)).run(prepared)
                m = split(st, cut)
                rs = [t["r_multiple"] for t in st.trades if t.get("r_multiple") is not None]
                reasons = {}
                for t in st.trades:
                    reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
                rows.append({"symbol": sym, "timeframe": tf, "mode": mode, "costs": cname, "split": cut,
                             "trades_per_week": round(len(st.trades) / weeks, 2),
                             "avg_r": round(float(np.mean(rs)), 3) if rs else None,
                             "best_r": round(max(rs), 2) if rs else None,
                             "reasons": reasons, "halted": st.halted, "yearly": yearly(st.equity), **m})
                s, v = m["selection"], m["validation"]
                log.info("%-8s %-3s %-6s %-6s Suche %+6.1f %% (PF %.2f) | Prüfung %+6.1f %% | %.1f/Woche | bester %sR",
                         sym, tf, mode, cname, s.get("total_return_pct", 0), s.get("profit_factor", 0),
                         v.get("total_return_pct", 0), rows[-1]["trades_per_week"], rows[-1]["best_r"])
    (OUT / "trend3.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
