"""Trendfolger v3: viele Rücksetzer im Trend mitnehmen, Traum-Trades laufen lassen.

Gegenüber v2:
  - Ausstieg ohne festes Ziel ("Traum-Trade"): ab +1R Stop auf Einstieg, danach
    nachgezogener Stop 2,5 ATR unter dem höchsten Hoch seit Einstieg (Short gespiegelt)
  - Trendbruch: Schluss auf der falschen Seite der EMA 200 oder höhere Zeitebene kippt
    -> raus (auch früh, kleiner Verlust)
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

    def __init__(self, tf: str = "1h", wide_stop: bool = False, reentry: bool = False, levels: bool = False,
                 partial: bool = False):
        """wide_stop: Stop 1 ATR hinter dem Rücksetzer (statt 0,2), Position entsprechend kleiner.
        reentry:   Wurde Freitag mit Gewinn geschlossen und der Trend steht noch, am Montag
                   direkt wieder einsteigen.
        levels:    Einstieg nur, wenn der Rücksetzer an einer früheren Umkehr-Marke endet
                   ("auf ähnlicher Höhe wie vorher").
        partial:   bei +1R die Hälfte schließen (Teilgewinn), Rest läuft als Traum-Trade."""
        self.tf = tf
        self.step = pd.Timedelta(TF[tf])
        self.wide_stop, self.reentry, self.levels, self.partial = wide_stop, reentry, levels, partial
        self.stop_pad, self.max_stop = (1.0, 4.0) if wide_stop else (0.2, 3.0)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        c, h, l = out["close"], out["high"], out["low"]
        e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
        rsi = ind.rsi(c, 14)
        out["atr"] = ind.atr(out, 14)
        out["ema50"] = e50
        out["ema200"] = e200
        slope = higher_tf_slope(out, rule=HTF[self.tf])
        out["htf_slope"] = slope
        a = ind.adx(out, 14)
        fading = ((a < a.shift(1)).astype(int).rolling(5, min_periods=5).sum() == 5) & (a < 25)
        strong = (a > 20) & ~fading
        up = (c > e200) & (e50 > e200) & (slope > 0) & strong
        dn = (c < e200) & (e50 < e200) & (slope < 0) & strong
        out["long"] = (up & (rsi.rolling(5, min_periods=5).min() < 40) & (c > h.shift(1))).fillna(False)
        out["short"] = (dn & (rsi.rolling(5, min_periods=5).max() > 60) & (c < l.shift(1))).fillna(False)
        out["swing_low"] = l.rolling(5, min_periods=5).min()
        out["swing_high"] = h.rolling(5, min_periods=5).max()
        out["trend_up"], out["trend_dn"] = up.fillna(False), dn.fillna(False)
        if self.levels:
            near_l, near_s = self._near_levels(out)
            out["long"] &= near_l
            out["short"] &= near_s
        # erste Kerzen nach einer Handelspause (Wochenbeginn)
        gap = pd.Series(out.index, index=out.index).diff() > pd.Timedelta(hours=24)
        out["week_open"] = gap.rolling(3, min_periods=1).max().astype(bool)
        close_t = out.index + self.step                      # Zeitpunkt, an dem die Kerze schließt
        fri = close_t.weekday == 4
        cutoff = close_t.normalize() + pd.Timedelta(hours=20)
        out["weekend_exit"] = (fri & (close_t + self.step >= cutoff)) | (close_t.weekday >= 5)
        out["no_entry"] = (fri & (close_t.hour >= 14)) | (close_t.weekday >= 5)
        out["ts_ns"] = out.index.as_unit("ns").asi8  # Index kann us-Auflösung haben
        for k in ("long", "short", "weekend_exit", "no_entry", "trend_up", "trend_dn"):
            out[k] = out[k].astype(bool)
        return out

    @staticmethod
    def _near_levels(df: pd.DataFrame, look: int = 100, k: int = 3, tol: float = 0.5):
        """Liegt das Rücksetzer-Tief (-Hoch) bis 0,5 ATR an einer früheren Umkehr-Marke?
        Umkehr-Marken = Hochs/Tiefs, die 3 Kerzen links und rechts überragen. Sie gelten
        erst 3 Kerzen später als bestätigt (kein Blick in die Zukunft)."""
        h, l = df["high"].to_numpy(), df["low"].to_numpy()
        n = len(df)
        piv = np.full(n, np.nan)  # Wert der bei Bar i bestätigten Marke
        for j in range(k, n - k):
            if h[j] == h[j - k:j + k + 1].max() or l[j] == l[j - k:j + k + 1].min():
                piv[j + k] = h[j] if h[j] == h[j - k:j + k + 1].max() else l[j]
        sl, sh, atr = df["swing_low"].to_numpy(), df["swing_high"].to_numpy(), df["atr"].to_numpy()
        near_l, near_s = np.zeros(n, bool), np.zeros(n, bool)
        for i in range(n):
            if not np.isfinite(atr[i]):
                continue
            w = piv[max(0, i - look):max(0, i - 5)]
            w = w[np.isfinite(w)]
            if w.size:
                near_l[i] = np.abs(w - sl[i]).min() <= tol * atr[i]
                near_s[i] = np.abs(w - sh[i]).min() <= tol * atr[i]
        return pd.Series(near_l, index=df.index), pd.Series(near_s, index=df.index)

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
            # Trendbruch: Schluss jenseits der EMA 200 oder die höhere Zeitebene kippt.
            # (Nicht EMA 50: im Rücksetzer liegt der Kurs oft genau dort, das würde
            # Trades direkt nach dem Einstieg wieder schließen.)
            hs = a["htf_slope"][i]
            if (side == 1 and (c < a["ema200"][i] or hs < 0)) or (side == -1 and (c > a["ema200"][i] or hs > 0)):
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
        pad = self.stop_pad
        for side, name, flag, stop in ((1, "long", "long", a["swing_low"][i] - pad * atr),
                                       (-1, "short", "short", a["swing_high"][i] + pad * atr)):
            signal = a[flag][i]
            if not signal and self.reentry and a["week_open"][i] and trades:
                last = trades[-1]
                fresh = now - pd.Timestamp(last["exit_time"]) < pd.Timedelta(hours=80)
                signal = (last["reason"] == "wochenende" and last["side"] == name and last["pnl"] > 0
                          and fresh and a["trend_up" if side == 1 else "trend_dn"][i])
            if not signal or not math.isfinite(stop) or self._paused(name, trades, now):
                continue
            if 0.5 * atr <= abs(c - stop) <= self.max_stop * atr:
                extra = {"tp1_r": 1.0, "tp1_frac": 0.5} if self.partial else {}
                return [("order", {"side": side, "type": "market", "sl": stop, **extra,
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
