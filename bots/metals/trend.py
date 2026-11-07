"""Trendfolger v2: Rücksetzer im einseitigen Trend handeln, Long UND Short.

Trend (Long-Seite, Short spiegelverkehrt):
  - Kurs > EMA 200 und EMA 50 > EMA 200 (1h)
  - optional: 4h-EMA 50 steigt (höhere Zeitebene bestätigt)
  - optional: ADX > 20 und kein "Trend läuft aus" (ADX fällt 5 Kerzen in Folge und < 25)
Einstieg: RSI(14) war in den letzten 5 Kerzen unter 40 (Ausreißer gegen den Trend)
          UND die Kerze schließt über dem Hoch der Vorkerze (Bestätigung der Umkehr).
Stop: unter dem tiefsten Tief der letzten 5 Kerzen - 0,2 ATR (0,5 bis 3 ATR entfernt).
Ziel: 2R; optional 3R mit Stop auf Einstieg, sobald +1R erreicht ist.

Die Parameter sind vorab festgelegt und werden NICHT optimiert. Getestet wird,
welcher Baustein etwas bringt (Varianten bauen aufeinander auf).

Aufruf: python -m bots.metals.trend   (schreibt results/trend.json)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd

from .. import data as D
from .. import indicators as ind
from ..hardcore.engine import FxRisk, FxSimulator
from ..hardcore.search import split
from .backtest import COSTS, stress, yearly

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[2] / "results"

VARIANTS = {
    "1_basis": dict(htf=False, adx=False, be=False),
    "2_plus_4h": dict(htf=True, adx=False, be=False),
    "3_plus_adx": dict(htf=True, adx=True, be=False),
    "4_plus_breakeven_3R": dict(htf=True, adx=True, be=True),
}
MARKETS = [("PAXGUSDT", "2024-01-01"), ("GC=F", None), ("SI=F", None), ("PL=F", None), ("JPY=X", None)]


def higher_tf_slope(df: pd.DataFrame, rule: str = "4h", n: int = 50, lag: int = 3) -> pd.Series:
    """Steigung der EMA auf der höheren Zeitebene, nur aus ABGESCHLOSSENEN 4h-Kerzen."""
    htf = df["close"].resample(rule, label="left", closed="left").last().dropna()
    e = ind.ema(htf, n)
    slope = e - e.shift(lag)
    known_at = pd.Series(slope.to_numpy(), index=slope.index + pd.Timedelta(rule))  # erst nach Kerzenende bekannt
    return known_at.reindex(df.index, method="ffill")


class TrendFollower:
    def __init__(self, htf: bool = True, adx: bool = True, be: bool = True):
        self.htf, self.use_adx, self.be = htf, adx, be
        self.rr = 3.0 if be else 2.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        c, h, l = out["close"], out["high"], out["low"]
        e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
        rsi = ind.rsi(c, 14)
        out["atr"] = ind.atr(out, 14)
        up = (c > e200) & (e50 > e200)
        dn = (c < e200) & (e50 < e200)
        if self.htf:
            sl = higher_tf_slope(out)
            up &= sl > 0
            dn &= sl < 0
        if self.use_adx:
            a = ind.adx(out, 14)
            fading = ((a < a.shift(1)).astype(int).rolling(5, min_periods=5).sum() == 5) & (a < 25)
            strong = (a > 20) & ~fading
            up &= strong
            dn &= strong
        out["long"] = up & (rsi.rolling(5, min_periods=5).min() < 40) & (c > h.shift(1))
        out["short"] = dn & (rsi.rolling(5, min_periods=5).max() > 60) & (c < l.shift(1))
        out["swing_low"] = l.rolling(5, min_periods=5).min()
        out["swing_high"] = h.rolling(5, min_periods=5).max()
        for k in ("long", "short"):
            out[k] = out[k].fillna(False).astype(bool)
        return out

    def on_bar(self, i, a, pos):
        if pos is not None:
            if self.be:
                risk = abs(pos["entry_price"] - pos["initial_sl"])
                if pos["side"] == 1 and a["high"][i] - pos["entry_price"] >= risk:
                    return [("stop", pos["entry_price"])]
                if pos["side"] == -1 and pos["entry_price"] - a["low"][i] >= risk:
                    return [("stop", pos["entry_price"])]
            return []
        c, atr = a["close"][i], a["atr"][i]
        if not (math.isfinite(atr) and atr > 0):
            return []
        for side, flag, stop in ((1, "long", a["swing_low"][i] - 0.2 * atr),
                                 (-1, "short", a["swing_high"][i] + 0.2 * atr)):
            if not a[flag][i] or not math.isfinite(stop):
                continue
            dist = abs(c - stop)
            if 0.5 * atr <= dist <= 3 * atr:
                return [("order", {"side": side, "type": "market", "sl": stop, "rr": self.rr,
                                   "tag": "trend_rücksetzer",
                                   "context": {"close": float(c), "atr": float(atr)}})]
        return []


def load(sym: str) -> pd.DataFrame:
    if sym == "PAXGUSDT":
        return D.binance_vision_klines(sym, "1h", start="2020-01")
    return D.yahoo_hourly(sym)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    rows = []
    for sym, cut in MARKETS:
        df = load(sym)
        cut = cut or (df.index[-1] - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        for vname, flags in VARIANTS.items():
            strat = TrendFollower(**flags)
            prepared = strat.prepare(df)
            for cname, costs in (("normal", COSTS[sym]), ("stress", stress(COSTS[sym]))):
                st = FxSimulator(strat, costs, FxRisk(risk_per_trade=0.01, max_leverage=5,
                                                     min_stop=0.0005)).run(prepared)
                m = split(st, cut)
                weeks = max(len(df) / (24 * 5), 1)  # Handelswochen (ca. 24h x 5 Tage)
                wins = [t["pnl"] for t in st.trades if t["pnl"] > 0]
                losses = [t["pnl"] for t in st.trades if t["pnl"] <= 0]
                rows.append({"symbol": sym, "variant": vname, "costs": cname, "split": cut,
                             "trades_per_week": round(len(st.trades) / weeks, 2),
                             "long_short": [sum(t["side"] == "long" for t in st.trades),
                                            sum(t["side"] == "short" for t in st.trades)],
                             "avg_win_pct": round(sum(wins) / len(wins) / 10, 2) if wins else 0,
                             "avg_loss_pct": round(sum(losses) / len(losses) / 10, 2) if losses else 0,
                             "yearly": yearly(st.equity), **m})
                s, v = m["selection"], m["validation"]
                log.info("%-8s %-20s %-6s PF %.2f / Prüfung %+.1f %% | Treffer %3.0f %% | %.1f Trades/Woche",
                         sym, vname, cname, s.get("profit_factor", 0), v.get("total_return_pct", 0),
                         s.get("win_rate_pct", 0), rows[-1]["trades_per_week"])
    (OUT / "trend.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
