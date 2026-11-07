"""Trendfolge auf dem Tageschart, getestet ab 2000 (Gold, Silber, Platin, Yen).

Drei bekannte Regelwerke, Parameter VORAB festgelegt (Lehrbuch-Werte, keine Optimierung):
  donchian: Schluss über dem 55-Tage-Hoch -> Kauf; Stop 2 ATR; raus bei Schluss unter dem 20-Tage-Tief.
  sma:      Kurs > SMA 200 und SMA 50 > SMA 200 -> Kauf; Stop 3 ATR, nachgezogen 3 ATR hinter dem
            höchsten Schluss; raus bei Schluss unter SMA 200.
  pullback: die TikTok-Idee auf Tagesbasis: Trend wie bei sma, RSI(14) war in 5 Tagen < 40 und
            Schluss über dem Vortageshoch; Stop 5-Tage-Tief - 1 ATR; ab +1R Stop 3 ATR hinter dem Hoch.
Short jeweils gespiegelt. Jede Regel läuft "nur long" und "long+short", und jeweils
  - "halten":   Positionen bleiben übers Wochenende offen
  - "flach":    Ausstieg Freitag zum Open, Neueinstieg frühestens Montag (Regel des Nutzers)
Einstieg/Ausstieg immer zum Open des Folgetags. 1 % Risiko je Trade, max. Hebel 5.
Suche: bis Ende 2015, Prüfung: ab 2016 (nie zur Auswahl benutzt).

Zusätzlich: Vermessung aller Wochenend-Lücken (Freitag-Schluss -> Montag-Eröffnung).

Aufruf: python -m bots.metals.daily   (schreibt results/daily.json)
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

log = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parents[2] / "results"
SYMBOLS = ["GC=F", "SI=F", "PL=F", "JPY=X"]
CUT = "2016-01-01"
RULES = ("donchian", "sma", "pullback")


class DailyTrend:
    def __init__(self, rule: str, shorts: bool = True, flat_weekend: bool = False):
        assert rule in RULES
        self.rule, self.shorts, self.flat = rule, shorts, flat_weekend

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        c, h, l = out["close"], out["high"], out["low"]
        out["atr"] = ind.atr(out, 20)
        s50, s200 = ind.sma(c, 50), ind.sma(c, 200)
        out["s200"] = s200
        up = (c > s200) & (s50 > s200)
        dn = (c < s200) & (s50 < s200)
        out["trend_up"], out["trend_dn"] = up, dn
        if self.rule == "donchian":
            out["long"] = c > h.shift(1).rolling(55).max()
            out["short"] = c < l.shift(1).rolling(55).min()
            out["exit_long"] = c < l.shift(1).rolling(20).min()
            out["exit_short"] = c > h.shift(1).rolling(20).max()
        elif self.rule == "sma":
            out["long"], out["short"] = up, dn
            out["exit_long"], out["exit_short"] = c < s200, c > s200
        else:
            rsi = ind.rsi(c, 14)
            out["long"] = up & (rsi.rolling(5).min() < 40) & (c > h.shift(1))
            out["short"] = dn & (rsi.rolling(5).max() > 60) & (c < l.shift(1))
            out["exit_long"], out["exit_short"] = c < s200, c > s200
        out["swing_low"] = l.rolling(5).min()
        out["swing_high"] = h.rolling(5).max()
        # Wochentag der Kerze (0 = Montag). Donnerstag: Signal -> Ausstieg Freitag zum Open.
        out["dow"] = out.index.dayofweek
        out["ts_ns"] = out.index.as_unit("ns").asi8
        for k in ("long", "short", "exit_long", "exit_short", "trend_up", "trend_dn"):
            out[k] = out[k].fillna(False).astype(bool)
        return out

    def on_bar(self, i, a, pos):
        c, atr = a["close"][i], a["atr"][i]
        if not (math.isfinite(atr) and atr > 0):
            return []
        thursday = a["dow"][i] == 3
        if pos is not None:
            side = pos["side"]
            if self.flat and thursday:
                return [("exit", "wochenende")]
            if (side == 1 and a["exit_long"][i]) or (side == -1 and a["exit_short"][i]):
                return [("exit", "trendbruch")]
            e = int(np.searchsorted(a["ts_ns"], pd.Timestamp(pos["entry_time"]).value))
            entry, risk = pos["entry_price"], abs(pos["entry_price"] - pos["initial_sl"])
            if self.rule == "sma":
                best = float(np.max(a["close"][e:i + 1])) if side == 1 else float(np.min(a["close"][e:i + 1]))
                return [("stop", best - side * 3 * atr)]
            if self.rule == "pullback":
                best = float(np.max(a["high"][e:i + 1])) if side == 1 else float(np.min(a["low"][e:i + 1]))
                if side * (best - entry) >= risk:
                    return [("stop", best - side * 3 * atr)]
            return []
        if self.flat and thursday:
            return []   # würde Freitag öffnen und sofort wieder schließen
        sides = ((1, "long"), (-1, "short")) if self.shorts else ((1, "long"),)
        for side, flag in sides:
            if not a[flag][i]:
                continue
            if self.rule == "donchian":
                stop = c - side * 2 * atr
            elif self.rule == "sma":
                stop = c - side * 3 * atr
            else:
                stop = a["swing_low"][i] - atr if side == 1 else a["swing_high"][i] + atr
                if not (0.5 * atr <= abs(c - stop) <= 5 * atr):
                    continue
            return [("order", {"side": side, "type": "market", "sl": float(stop), "tag": f"daily_{self.rule}",
                               "context": {"close": float(c), "sma200": float(a["s200"][i]), "atr": float(atr)}})]
        return []


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo liefert vereinzelt Kerzen mit Hoch < Schluss o. ä. - gerade ziehen."""
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].copy()
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


def weekend_gaps(df: pd.DataFrame) -> dict:
    """Lücke zwischen letztem Schluss vor dem Wochenende und der ersten Eröffnung danach."""
    atr = ind.atr(df, 20).shift(1)
    prev_close = df["close"].shift(1)
    days = pd.Series(df.index, index=df.index).diff().dt.days
    wk = (days >= 2) & atr.notna()
    gap = (df["open"] / prev_close - 1)[wk]
    in_atr = ((df["open"] - prev_close).abs() / atr)[wk]
    worst = gap.abs().sort_values(ascending=False).head(8)
    return {
        "anzahl_wochenenden": int(wk.sum()),
        "median_pct": round(float(gap.abs().median() * 100), 3),
        "p95_pct": round(float(gap.abs().quantile(0.95) * 100), 2),
        "p99_pct": round(float(gap.abs().quantile(0.99) * 100), 2),
        "groesser_1_atr": int((in_atr > 1).sum()),
        "groesser_2_atr": int((in_atr > 2).sum()),
        "schlimmste": [{"datum": str(t.date()), "pct": round(float(gap[t] * 100), 2),
                        "atr": round(float(in_atr[t]), 2)} for t in worst.index],
    }


def gap_damage(trades: list) -> dict:
    """Wie oft hat eine Lücke den Stop übersprungen (Verlust deutlich über 1R)?"""
    stops = [t for t in trades if t["reason"] == "stop" and t["r_multiple"] is not None]
    rs = [t["r_multiple"] for t in stops]
    return {"stops": len(stops), "schlechter_als_-1.3R": sum(r < -1.3 for r in rs),
            "schlechter_als_-2R": sum(r < -2 for r in rs),
            "schlimmster_R": round(min(rs), 2) if rs else None}


def buy_hold(df: pd.DataFrame, start: str, end: str | None = None) -> float:
    s = df["close"][df.index >= start]
    if end:
        s = s[s.index < end]
    years = (s.index[-1] - s.index[0]).days / 365.25
    return round(((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100, 2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(exist_ok=True)
    out = {"gaps": {}, "runs": [], "buy_hold_cagr": {}}
    for sym in SYMBOLS:
        df = tidy(D.yahoo_daily(sym, start="2000-01-01"))
        log.info("%s: %d Tage %s bis %s", sym, len(df), df.index[0].date(), df.index[-1].date())
        out["gaps"][sym] = weekend_gaps(df)
        out["buy_hold_cagr"][sym] = {"suche": buy_hold(df, str(df.index[0].date()), CUT),
                                     "pruefung": buy_hold(df, CUT)}
        for rule in RULES:
            for shorts in (False, True):
                for flat in (False, True):
                    strat = DailyTrend(rule, shorts=shorts, flat_weekend=flat)
                    p = strat.prepare(df)
                    risk = FxRisk(risk_per_trade=0.01, max_leverage=5, min_stop=0.002)
                    st = FxSimulator(strat, COSTS[sym], risk).run(p)
                    st_s = FxSimulator(strat, stress(COSTS[sym]), risk).run(p)
                    m, ms = split(st, CUT), split(st_s, CUT)
                    row = {"symbol": sym, "rule": rule, "shorts": shorts,
                           "weekend": "flach" if flat else "halten",
                           "selection": m["selection"], "validation": m["validation"],
                           "validation_stress": ms["validation"], "yearly": yearly(st.equity),
                           "gap_damage": gap_damage(st.trades),
                           "long_short_pnl": [round(sum(t["pnl"] for t in st.trades if t["side"] == s), 1)
                                              for s in ("long", "short")]}
                    out["runs"].append(row)
                    s, v, vs = m["selection"], m["validation"], ms["validation"]
                    log.info("%-6s %-8s %-11s %-6s | Suche CAGR %+5.1f %% DD %5.1f %% PF %.2f | "
                             "Prüfung CAGR %+5.1f %% DD %5.1f %% PF %.2f (Stress %+5.1f %%) | %.1f Trades/J",
                             sym, rule, "long+short" if shorts else "nur long", "flach" if flat else "halten",
                             s.get("cagr_pct", 0), s.get("max_drawdown_pct", 0), s.get("profit_factor", 0),
                             v.get("cagr_pct", 0), v.get("max_drawdown_pct", 0), v.get("profit_factor", 0),
                             vs.get("cagr_pct", 0), v.get("trades_per_year", 0))
    (OUT / "daily.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
