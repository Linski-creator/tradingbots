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

from .dashboard import _position_text
from .metrics import equity_series

SITE = Path(__file__).resolve().parent.parent / "site"


def daily_summary(equity: list, trades: list, start_cash: float, extra_events: list = ()) -> dict:
    """{'YYYY-MM-DD': {'pnl': €, 'value': Konto am Tagesende, 'trades': Anzahl}}.
    Trades zählen am Tag des Einstiegs und am Tag des Ausstiegs (je eine Aktion)."""
    actions = defaultdict(int)
    for t in trades:
        actions[t["entry_time"][:10]] += 1
        actions[t["exit_time"][:10]] += 1
    for ts in extra_events:  # offene Position: Einstieg zählt schon
        actions[ts[:10]] += 1
    out, prev = {}, start_cash
    eq = equity_series(equity)
    if not eq.empty:
        for ts, v in eq.resample("1D").last().dropna().items():
            day = ts.strftime("%Y-%m-%d")
            out[day] = {"pnl": round(v - prev, 2), "value": round(v, 2), "trades": actions.get(day, 0)}
            prev = v
    for day, n in actions.items():
        out.setdefault(day, {"pnl": 0.0, "value": round(prev, 2), "trades": n})
    return out


def _open_entry(st: dict) -> list:
    if st.get("position"):
        return [st["position"]["entry_time"]]
    if st.get("holding") and st.get("entry_time"):
        return [st["entry_time"]]
    if st.get("active") and st.get("current"):
        return [st["current"]["entry_time"]]
    return []


def build_data(reports: list[dict], cfg: dict) -> dict:
    start_cash = cfg["start_cash"]
    bots, team_days = [], defaultdict(lambda: {"pnl": 0.0, "trades": 0})
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
    team = [b for b in bots if not b["compare"]]
    return {
        "updated": datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC"),
        "level": cfg.get("level", ""),
        "team": {"start": start_cash * len(team), "current": round(sum(b["current"] for b in team), 2),
                 "days": {d: {"pnl": round(v["pnl"], 2), "trades": v["trades"]} for d, v in team_days.items()}},
        "bots": bots,
    }


def write_site(reports: list[dict], cfg: dict) -> None:
    SITE.mkdir(exist_ok=True)
    data = json.dumps(build_data(reports, cfg), ensure_ascii=False).replace("</", "<\\/")
    template = (Path(__file__).resolve().parent / "webpage_template.html").read_text()
    (SITE / "index.html").write_text(template.replace("/*__DATA__*/null", data))
