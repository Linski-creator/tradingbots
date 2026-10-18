"""Einfache Webseite fürs Handy/iPad: site/index.html.

Startseite: alle Bots mit Start- und aktuellem Stand. Antippen öffnet einen
Kalender: jeder Tag grün oder rot, Gewinn/Verlust und Anzahl Trades.
Die Daten stecken direkt in der HTML-Datei, die Seite braucht also keinen
Server und keine weiteren Abrufe.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .dashboard import _position_text
from .metrics import equity_series

SITE = Path(__file__).resolve().parent.parent / "site"


def daily_summary(equity: list, trades: list, start_cash: float, extra_entries: list = ()) -> dict:
    """{'YYYY-MM-DD': {'pnl': €, 'value': Konto am Tagesende, 'trades': abgeschlossene
    Trades, 'opened': neue Einstiege}}. Ein Trade zählt am Tag seines Ausstiegs."""
    closed, opened = defaultdict(int), defaultdict(int)
    for t in trades:
        closed[t["exit_time"][:10]] += 1
        opened[t["entry_time"][:10]] += 1
    for ts in extra_entries:  # noch offene Position
        opened[ts[:10]] += 1
    eq = equity_series(equity)
    daily = eq.resample("1D").last() if not eq.empty else eq
    days = set(closed) | set(opened) | {ts.strftime("%Y-%m-%d") for ts in daily.dropna().index}
    out, prev = {}, start_cash
    for day in sorted(days):
        upto = daily[:day].dropna() if not daily.empty else daily
        value = float(upto.iloc[-1]) if len(upto) else start_cash
        out[day] = {"pnl": round(value - prev, 2), "value": round(value, 2),
                    "trades": closed.get(day, 0), "opened": opened.get(day, 0)}
        prev = value
    return out


def _open_entry(st: dict) -> list:
    if st.get("position"):
        return [st["position"]["entry_time"]]
    if st.get("holding") and st.get("entry_time"):
        return [st["entry_time"]]
    if st.get("active") and st.get("current"):
        return [st["current"]["entry_time"]]
    return []


def build_data(reports: list[dict], cfg: dict, errors: list = ()) -> dict:
    start_cash = cfg["start_cash"]
    bots, team_days = [], defaultdict(lambda: {"pnl": 0.0, "trades": 0, "opened": 0})
    for r in reports:
        bot, st = r["bot"], r["state"]
        eq = st.get("equity") or []
        cur = eq[-1][1] if eq else start_cash
        days = daily_summary(eq, st.get("trades", []), start_cash, _open_entry(st))
        trades = [{"entry": t["entry_time"][:16].replace("T", " "), "exit": t["exit_time"][:16].replace("T", " "),
                   "pnl": round(t["pnl"], 2), "reason": t["reason"]}
                  for t in sorted(st.get("trades", []), key=lambda t: t["exit_time"], reverse=True)[:20]]
        bots.append({
            "id": bot["id"], "label": r["label"], "market": bot.get("symbol", "/".join(bot.get("universe", []))),
            "compare": bool(bot.get("compare")), "start": start_cash, "current": round(cur, 2),
            "halted": bool(st.get("halted")), "position": _position_text(r), "days": days, "trades": trades,
        })
        if not bot.get("compare"):
            for d, v in days.items():
                team_days[d]["pnl"] += v["pnl"]
                team_days[d]["trades"] += v["trades"]
                team_days[d]["opened"] += v["opened"]
    team = [b for b in bots if not b["compare"]]
    return {
        "updated": datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC"),
        "level": cfg.get("level", ""),
        "team": {"start": start_cash * len(team), "current": round(sum(b["current"] for b in team), 2),
                 "count": len(team), "halted": bool(cfg.get("_team", {}).get("halted")),
                 "days": {d: {"pnl": round(v["pnl"], 2), "trades": v["trades"], "opened": v["opened"]}
                          for d, v in team_days.items()}},
        "errors": [e["bot"] for e in errors],
        "news": [{"time": e["time"].isoformat(), "title": e["title"]} for e in (cfg.get("_news") or [])
                 if e["time"] >= datetime.now(timezone.utc) - pd.Timedelta(hours=1)][:5],
        "bots": bots,
    }


def write_site(reports: list[dict], errors: list, cfg: dict) -> None:
    SITE.mkdir(exist_ok=True)
    data = json.dumps(build_data(reports, cfg, errors), ensure_ascii=False).replace("</", "<\\/")
    template = (Path(__file__).resolve().parent / "webpage_template.html").read_text(encoding="utf-8")
    (SITE / "index.html").write_text(template.replace("/*__DATA__*/null", data), encoding="utf-8")
