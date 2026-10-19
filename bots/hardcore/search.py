"""Suche nach einem aktiven 3:1-Bot (Futures, Long und Short in Trendrichtung).

8 Einstiegs-Signale x 2 Zeitebenen x 2 Stop-Abstände = 32 Varianten,
jeweils auf BTC, ETH und SOL. Ziel immer 3 x Risiko (3:1).

Gegen Zufallstreffer zweistufig:
  Suche   : Daten bis 31.12.2024
  Prüfung : ab 01.01.2025 - wird bei der Auswahl nicht angeschaut
Eine Variante BESTEHT nur, wenn
  - in der Suche der Profit-Faktor auf mindestens 2 von 3 Märkten > 1,1 ist und
  - in der Prüfung auch mit Stress-Kosten auf mindestens 2 von 3 Märkten Gewinn bleibt.

Aufruf: python -m bots.hardcore.search   (schreibt results/search.json)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd

from .. import data as D
from .. import indicators as ind
from ..metrics import summarize
from .backtest import COSTS
from .engine import FxRisk, FxSimulator
from .strategies import AsiaBreakout

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[2] / "results"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["15m", "1h"]
ATR_MULTS = [1.5, 2.0]
SPLIT_DATE = "2025-01-01"
START = "2022-01"
SIGNALS = ["donchian", "ema_cross", "pullback", "rsi_hook", "squeeze", "inside_bar", "macd", "asia"]


def _cross_up(a: pd.Series, b) -> pd.Series:
    return (a > b) & (a.shift(1) <= (b.shift(1) if isinstance(b, pd.Series) else b))


def _cross_dn(a: pd.Series, b) -> pd.Series:
    return (a < b) & (a.shift(1) >= (b.shift(1) if isinstance(b, pd.Series) else b))


def signals(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Alle Einstiegssignale (long, short) kausal, jeweils mit Trendfilter EMA 200."""
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    e20, e50, e200 = ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
    up = (c > e200) & (e200 > e200.shift(10))
    dn = (c < e200) & (e200 < e200.shift(10))
    out = {}
    out["donchian"] = (up & (c > ind.highest(h, 20).shift(1)), dn & (c < ind.lowest(l, 20).shift(1)))
    out["ema_cross"] = (up & _cross_up(e20, e50), dn & _cross_dn(e20, e50))
    out["pullback"] = (up & (l <= e20) & (c > e20) & (c > o), dn & (h >= e20) & (c < e20) & (c < o))
    rsi = ind.rsi(c, 14)
    out["rsi_hook"] = (up & _cross_up(rsi, 30), dn & _cross_dn(rsi, 70))
    mid = ind.sma(c, 20)
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    bw = (4 * sd) / mid
    squeezed = (bw <= bw.rolling(100, min_periods=100).min().shift(1) * 1.1).rolling(5, min_periods=1).max() > 0
    out["squeeze"] = (up & squeezed & (c > mid + 2 * sd), dn & squeezed & (c < mid - 2 * sd))
    inside = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
    out["inside_bar"] = (up & inside & (c > h.shift(1)), dn & inside & (c < l.shift(1)))
    macd = ind.ema(c, 12) - ind.ema(c, 26)
    sig = ind.ema(macd.dropna(), 9).reindex(macd.index)
    out["macd"] = (up & _cross_up(macd, sig) & (macd < 0), dn & _cross_dn(macd, sig) & (macd > 0))
    asia = AsiaBreakout().prepare(df)
    fb = asia["first_break"].astype(bool)
    out["asia"] = (up & fb & (c > asia["r_hi"]), dn & fb & (c < asia["r_lo"]))
    return {k: (a.fillna(False).astype(bool), b.fillna(False).astype(bool)) for k, (a, b) in out.items()}


class ThreeToOne:
    """Einstieg per Signal zum nächsten Open, Stop k x ATR(14), Ziel 3R."""

    def __init__(self, signal: str, atr_mult: float):
        self.signal, self.atr_mult = signal, atr_mult
        self.name = f"{signal}_{atr_mult}"

    def prepare(self, df: pd.DataFrame, sigs: dict | None = None) -> pd.DataFrame:
        out = df.copy()
        lg, sh = (sigs or signals(df))[self.signal]
        out["sig_long"], out["sig_short"] = lg, sh
        out["atr"] = ind.atr(df, 14)
        return out

    def on_bar(self, i, a, pos):
        if pos is not None:
            return []
        atr, c = a["atr"][i], a["close"][i]
        if not (math.isfinite(atr) and atr > 0):
            return []
        if a["sig_long"][i]:
            return [("order", {"side": 1, "type": "market", "sl": c - self.atr_mult * atr, "rr": 3.0,
                               "tag": self.signal})]
        if a["sig_short"][i]:
            return [("order", {"side": -1, "type": "market", "sl": c + self.atr_mult * atr, "rr": 3.0,
                               "tag": self.signal})]
        return []


def split(st, cut: str) -> dict:
    eq = st.equity
    a = [e for e in eq if e[0] < cut]
    b = [e for e in eq if e[0] >= cut]
    ta = [t for t in st.trades if t["exit_time"] < cut]
    tb = [t for t in st.trades if t["exit_time"] >= cut]
    sel = summarize(a, ta, 1000.0) if a else {"trades": 0}
    val = summarize(b, tb, a[-1][1] if a else 1000.0) if b else {"trades": 0}
    for m, tr in ((sel, ta), (val, tb)):
        if m.get("years"):
            m["trades_per_day"] = len(tr) / (m["years"] * 365.25)
    return {"selection": sel, "validation": val}


def evaluate(rows: list[dict]) -> list[dict]:
    """Fasst je Variante die drei Märkte zusammen und prüft die Bestehensregeln."""
    by = {}
    for r in rows:
        by.setdefault((r["signal"], r["timeframe"], r["atr_mult"]), {}).setdefault(r["costs"], {})[r["symbol"]] = r
    verdicts = []
    for (sig, tf, k), costs in sorted(by.items()):
        fut, stress = costs.get("futures", {}), costs.get("stress", {})
        pf = {s: fut[s]["selection"].get("profit_factor", 0) for s in fut}
        val = {s: stress[s]["validation"].get("total_return_pct", -100) for s in stress}
        sel_ok = sum(1 for v in pf.values() if v > 1.1) >= 2
        val_ok = sum(1 for v in val.values() if v > 0) >= 2
        verdicts.append({"signal": sig, "timeframe": tf, "atr_mult": k, "selection_pf": pf,
                         "validation_return_stress": val,
                         "validation_return_futures": {s: fut[s]["validation"].get("total_return_pct", -100)
                                                       for s in fut},
                         "trades_per_day": sum(fut[s]["selection"].get("trades_per_day", 0) for s in fut) / max(len(fut), 1),
                         "win_rate": {s: fut[s]["selection"].get("win_rate_pct", 0) for s in fut},
                         "passes_selection": sel_ok, "passes": sel_ok and val_ok})
    return verdicts


def load(symbol: str, interval: str) -> pd.DataFrame:
    return D.binance_vision_klines(symbol, interval, start=START, market="futures")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    rows = []
    for tf in TIMEFRAMES:
        for sym in SYMBOLS:
            df = load(sym, tf)
            log.info("%s %s: %d Kerzen", sym, tf, len(df))
            sigs = signals(df)
            for sig in SIGNALS:
                for k in ATR_MULTS:
                    strat = ThreeToOne(sig, k)
                    prepared = strat.prepare(df, sigs)
                    for cname, costs in COSTS.items():
                        st = FxSimulator(strat, costs, FxRisk()).run(prepared)
                        m = split(st, SPLIT_DATE)
                        rows.append({"signal": sig, "timeframe": tf, "atr_mult": k, "symbol": sym,
                                     "costs": cname, **m})
                        s = m["selection"]
                        log.info("%-10s %-3s %.1f %-8s %-7s PF %.2f  Treffer %3.0f %%  Trades/Tag %.2f",
                                 sig, tf, k, sym, cname, s.get("profit_factor", 0), s.get("win_rate_pct", 0),
                                 s.get("trades_per_day", 0))
    verdicts = evaluate(rows)
    (OUT / "search.json").write_text(json.dumps({"rows": rows, "verdicts": verdicts}, indent=1, default=str),
                                     encoding="utf-8")
    passed = [v for v in verdicts if v["passes"]]
    log.info("Fertig: %d Varianten, %d bestehen die Suche, %d bestehen auch die Prüfung",
             len(verdicts), sum(v["passes_selection"] for v in verdicts), len(passed))


if __name__ == "__main__":
    main()
