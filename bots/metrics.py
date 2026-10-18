"""Kennzahlen aus Equity-Kurve und Trades."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def equity_series(equity: list) -> pd.Series:
    if not equity:
        return pd.Series(dtype=float)
    ts, vals = zip(*equity)
    return pd.Series(vals, index=pd.to_datetime(list(ts), utc=True), dtype=float)


def max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    return float((eq / eq.cummax() - 1).min())


def summarize(equity: list, trades: list, start_cash: float) -> dict:
    eq = equity_series(equity)
    if eq.empty:
        return {"trades": 0}
    daily = eq.resample("1D").last().dropna()
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    final = float(eq.iloc[-1])
    rets = daily.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * math.sqrt(365)) if len(rets) > 2 and rets.std() > 0 else 0.0
    monthly = eq.resample("ME").last().dropna()
    mrets = pd.concat([pd.Series([start_cash]), monthly.reset_index(drop=True)]).pct_change().dropna()

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    rs = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    mdd = max_drawdown(eq)
    cagr = (final / start_cash) ** (1 / years) - 1 if final > 0 else -1.0
    return {
        "final": final,
        "total_return_pct": (final / start_cash - 1) * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": mdd * 100,
        "sharpe": sharpe,
        "calmar": (cagr / abs(mdd)) if mdd < 0 else float("inf"),
        "trades": len(trades),
        "trades_per_year": len(trades) / years,
        "win_rate_pct": len(wins) / len(pnls) * 100 if pnls else 0.0,
        "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf"),
        "avg_r": float(np.mean(rs)) if rs else None,
        "positive_months_pct": float((mrets > 0).mean() * 100) if len(mrets) else 0.0,
        "worst_month_pct": float(mrets.min() * 100) if len(mrets) else 0.0,
        "years": years,
    }
