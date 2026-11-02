"""Nachbau und Prüfung der TikTok-Strategie auf Gold, Silber, Platin und USD/JPY.

Regel: Kurs über EMA 150 (Aufwärtstrend). RSI(14) fällt unter 35 und
steigt wieder darüber -> Kauf zum nächsten Open. Stop und Ziel als
Vielfache der ATR(14); 1 % Kontorisiko pro Trade, höchstens 10x Hebel.

Weil Stop/Ziel des Originals unbekannt sind, werden 4 Varianten getestet.
  Gruppe A: Gold über den Gold-Token PAXG (5m/15m/1h) ab 2020,
            Suche bis 2023, Prüfung ab 2024
  Gruppe B: Gold-, Silber-, Platin-Future und USD/JPY (Yahoo, 1h, letzte 2 Jahre),
            Suche im ersten, Prüfung im letzten Jahr
Bestehensregeln siehe ``evaluate``; die Prüfdaten werden bei der Auswahl
nicht angeschaut.

Aufruf: python -m bots.metals.backtest   (schreibt results/metals.json)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd

from .. import indicators as ind
from ..hardcore.engine import FxCosts, FxRisk, FxSimulator
from ..hardcore.search import split
from ..metrics import equity_series
from .. import data as D

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[2] / "results"
# Dukascopy ist von den GitHub-Servern aus stark gedrosselt (503), daher:
# Gruppe A: Gold über den Gold-Token PAXG (Binance), 5m/15m/1h ab 2020, Prüfung ab 2024
# Gruppe B: Gold-, Silber-, Platin-Future und USD/JPY (Yahoo), 1h der letzten 2 Jahre,
#           Prüfung im letzten Jahr
SYMBOLS = ["GC=F", "SI=F", "PL=F", "JPY=X"]
GROUPS = {
    "A": {"jobs": [("PAXGUSDT", "5m"), ("PAXGUSDT", "15m"), ("PAXGUSDT", "1h")], "split": "2024-01-01"},
    "B": {"jobs": [(s, "1h") for s in SYMBOLS],
          "split": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)).strftime("%Y-%m-%d")},
}
EXITS = [(1.5, 1.5), (2.0, 1.0), (1.5, 3.0), (3.0, 1.0)]   # (Stop x ATR, Ziel x ATR)
START = "2012-01"

# Typische Kosten beim ECN-Broker (Spread + Kommission) relativ zum Preis,
# dazu Übernacht-Kosten (Swap) als Näherung pro 8 Stunden.
COSTS = {
    "GC=F": FxCosts(maker=0.00010, taker=0.00010, slippage=0.00005, funding_8h=0.00007),
    "SI=F": FxCosts(maker=0.00040, taker=0.00040, slippage=0.00010, funding_8h=0.00007),
    "PL=F": FxCosts(maker=0.00080, taker=0.00080, slippage=0.00020, funding_8h=0.00007),
    "JPY=X": FxCosts(maker=0.00005, taker=0.00005, slippage=0.00003, funding_8h=0.00002),
    "XAUUSD": FxCosts(maker=0.00010, taker=0.00010, slippage=0.00005, funding_8h=0.00007),
    "XAGUSD": FxCosts(maker=0.00040, taker=0.00040, slippage=0.00010, funding_8h=0.00007),
    "XPTUSD": FxCosts(maker=0.00080, taker=0.00080, slippage=0.00020, funding_8h=0.00007),
    "USDJPY": FxCosts(maker=0.00005, taker=0.00005, slippage=0.00003, funding_8h=0.00002),
    # Gold-Token-Daten, aber Kosten wie Gold beim CFD-Broker (dort würde live gehandelt)
    "PAXGUSDT": FxCosts(maker=0.00010, taker=0.00010, slippage=0.00005, funding_8h=0.00007),
}


def stress(c: FxCosts) -> FxCosts:
    return FxCosts(maker=c.taker * 2, taker=c.taker * 2, slippage=c.slippage * 2, funding_8h=c.funding_8h * 2)


class TrendDip:
    """Kauf des Rücksetzers im Aufwärtstrend (Nachbau)."""

    def __init__(self, sl_atr: float, tp_atr: float, ema_n: int = 150, rsi_n: int = 14, level: float = 35):
        self.sl_atr, self.tp_atr, self.ema_n, self.rsi_n, self.level = sl_atr, tp_atr, ema_n, rsi_n, level
        self.name = f"sl{sl_atr}_tp{tp_atr}"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema"] = ind.ema(out["close"], self.ema_n)
        out["rsi"] = ind.rsi(out["close"], self.rsi_n)
        out["atr"] = ind.atr(out, 14)
        prev = out["rsi"].shift(1)
        out["signal"] = (out["close"] > out["ema"]) & (prev < self.level) & (out["rsi"] >= self.level)
        return out

    def on_bar(self, i, a, pos):
        if pos is not None or not a["signal"][i]:
            return []
        c, atr = a["close"][i], a["atr"][i]
        if not (math.isfinite(atr) and atr > 0):
            return []
        return [("order", {"side": 1, "type": "market", "sl": c - self.sl_atr * atr,
                           "tp": c + self.tp_atr * atr, "tag": "rücksetzer"})]


def yearly(equity: list) -> dict:
    eq = equity_series(equity)
    if eq.empty:
        return {}
    y = eq.resample("YE").last()
    start = eq.iloc[0]
    out = {}
    for ts, v in y.items():
        out[str(ts.year)] = round((v / start - 1) * 100, 1)
        start = v
    return out


def evaluate(rows: list[dict]) -> list[dict]:
    """Gruppe A (4 Märkte): besteht mit PF > 1,1 auf >= 2 Märkten in der Suche und
    Gewinn unter Stress-Kosten auf >= 2 Märkten in der Prüfung.
    Gruppe B (nur Gold): PF > 1,1 in der Suche und Gewinn unter Stress in der Prüfung."""
    by = {}
    for r in rows:
        by.setdefault((r["group"], r["timeframe"], r["exit"]), {}).setdefault(r["costs"], {})[r["symbol"]] = r
    out = []
    for (grp, tf, ex), c in sorted(by.items()):
        base, st = c.get("normal", {}), c.get("stress", {})
        pf = {s: base[s]["selection"].get("profit_factor", 0) for s in base}
        val = {s: st[s]["validation"].get("total_return_pct", -100) for s in st}
        need = 2 if grp == "B" else 1
        sel_ok = sum(v > 1.1 for v in pf.values()) >= need
        val_ok = sum(v > 0 for v in val.values()) >= need
        gold = base.get("PAXGUSDT") or base.get("GC=F") or {}
        out.append({"group": grp, "timeframe": tf, "exit": ex, "selection_pf": pf,
                    "validation_return_stress": val,
                    "validation_return_normal": {s: base[s]["validation"].get("total_return_pct", -100) for s in base},
                    "win_rate": {s: base[s]["selection"].get("win_rate_pct", 0) for s in base},
                    "trades_per_day": {s: base[s]["selection"].get("trades_per_day", 0) for s in base},
                    "yearly_gold": gold.get("yearly", {}),
                    "passes_selection": bool(sel_ok), "passes": bool(sel_ok and val_ok)})
    return out


def load(sym: str, tf: str) -> pd.DataFrame:
    if sym == "PAXGUSDT":
        return D.binance_vision_klines(sym, tf, start="2020-01")
    return D.yahoo_hourly(sym)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    rows = []
    for grp, spec in GROUPS.items():
        for sym, tf in spec["jobs"]:
            df = load(sym, tf)
            log.info("%s %s: %d Kerzen ab %s", sym, tf, len(df), df.index[0])
            for sl, tp in EXITS:
                strat = TrendDip(sl, tp)
                prepared = strat.prepare(df)
                for cname, costs in (("normal", COSTS[sym]), ("stress", stress(COSTS[sym]))):
                    st = FxSimulator(strat, costs, FxRisk(risk_per_trade=0.01, max_leverage=10,
                                                         min_stop=0.0005)).run(prepared)
                    m = split(st, spec["split"])
                    rows.append({"group": grp, "symbol": sym, "timeframe": tf, "exit": strat.name, "costs": cname,
                                 "yearly": yearly(st.equity), **m})
                    s = m["selection"]
                    log.info("%s %-3s %-12s %-6s PF %.2f Treffer %3.0f %% Trades/Tag %.2f | Prüfung %+.1f %%",
                             sym, tf, strat.name, cname, s.get("profit_factor", 0), s.get("win_rate_pct", 0),
                             s.get("trades_per_day", 0), m["validation"].get("total_return_pct", 0))
    verdicts = evaluate(rows)
    (OUT / "metals.json").write_text(json.dumps({"rows": rows, "verdicts": verdicts}, indent=1, default=str),
                                     encoding="utf-8")
    log.info("Fertig: %d Varianten, %d bestehen die Suche, %d auch die Prüfung", len(verdicts),
             sum(v["passes_selection"] for v in verdicts), sum(v["passes"] for v in verdicts))


if __name__ == "__main__":
    main()
