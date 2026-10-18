"""Trading-Journal: docs/JOURNAL.md, bei jedem Paper-Lauf neu geschrieben.

Für jeden Trade: Grund für Ein- und Ausstieg (mit den Indikatorwerten am
Signal), Stop, Ziel, Ergebnis, R-Vielfaches und Dauer. Dazu Kennzahlen je
Bot, offene Positionen und ein Tagesprotokoll des Teams.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .dashboard import _fmt_eur, _pct, team_average
from .metrics import equity_series

DOCS = Path(__file__).resolve().parent.parent / "docs"

EXIT_REASONS = {
    "stop": "Stop-Loss ausgelöst",
    "take_profit": "Ziel erreicht",
    "kanal_tief": "Schluss unter dem 10-Tage-Tief",
    "unter_sma": "Schluss unter der 200-Tage-Linie",
    "momentum_negativ": "90-Tage-Rendite nicht mehr positiv",
    "rsi_hoch": "RSI(2) über 70",
    "zeitstop": "Zeitlimit erreicht",
    "death_cross": "EMA 9 kreuzt EMA 21 nach unten",
    "rotation": "Wochenwechsel: anderer Markt stärker oder Cash",
    "funding_negativ": "Funding-Durchschnitt nicht mehr positiv",
    "notbremse": "Notbremse (Einbruch zu groß)",
    "team_notbremse": "Team-Notbremse (Team-Einbruch zu groß)",
}


def _n(x, d=2) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def entry_reason(strategy: str, c: dict) -> str:
    """Lesbarer Einstiegsgrund aus den Indikatorwerten am Signal."""
    g = c.get
    try:
        if strategy == "donchian":
            return f"Schluss {_n(g('close'))} über dem 20-Tage-Hoch {_n(g('upper'))} (ATR {_n(g('atr'))})"
        if strategy == "chandelier":
            return (f"EMA 20 ({_n(g('ema_f'))}) kreuzt EMA 50 ({_n(g('ema_s'))}) nach oben, "
                    f"Kurs über EMA 200 ({_n(g('ema_t'))})")
        if strategy == "tsmom":
            return f"90-Tage-Rendite positiv: {_n(g('mom') * 100, 1)} %"
        if strategy == "sma200":
            return f"Schluss {_n(g('close'))} über der 200-Tage-Linie {_n(g('sma'))}"
        if strategy == "rsi2":
            return f"RSI(2) {_n(g('rsi'), 1)} unter 10, Kurs über SMA 200 ({_n(g('sma'))})"
        if strategy == "pullback":
            return f"Rücksetzer auf EMA 20 ({_n(g('ema_f'))}) im Aufwärtstrend, grüne Kerze"
        if strategy == "emacross":
            return f"EMA 9 ({_n(g('ema_f'))}) kreuzt EMA 21 ({_n(g('ema_s'))}) nach oben"
        if strategy == "dualmom":
            moms = {k[4:]: v for k, v in c.items() if k.startswith("mom_")}
            return "Stärkster Markt (90 Tage): " + ", ".join(
                f"{k} {_n(v * 100, 1)} %" for k, v in sorted(moms.items(), key=lambda kv: -kv[1]))
        if strategy == "funding":
            return f"Ø Funding {_n(g('avg_rate') * 100, 4)} % pro 8 h (≈ {_n(g('avg_rate') * 3 * 365 * 100, 1)} % p. a.)"
    except (TypeError, KeyError):
        pass
    return ", ".join(f"{k} {_n(v) if isinstance(v, float) else v}" for k, v in c.items()) or "–"


def _dur(a: str, b: str) -> str:
    d = pd.Timestamp(b) - pd.Timestamp(a)
    h = d.total_seconds() / 3600
    return f"{h:.0f} h" if h < 48 else f"{h / 24:.1f} Tage".replace(".", ",")


def _ts(t: str) -> str:
    return pd.Timestamp(t).strftime("%d.%m.%y %H:%M")


def write_journal(reports: list[dict], cfg: dict) -> None:
    DOCS.mkdir(exist_ok=True)
    start_cash = cfg["start_cash"]
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    rows_stats, trades, opens, team_curves = [], [], [], []
    for r in reports:
        bot, st = r["bot"], r["state"]
        name = f"{bot['id']} (Vergleich)" if bot.get("compare") else bot["id"]
        if not bot.get("compare"):
            team_curves.append(st.get("equity") or [])
        tr = st.get("trades", [])
        pnl = [t["pnl"] for t in tr]
        rs = [t["r_multiple"] for t in tr if t.get("r_multiple") is not None]
        rows_stats.append(
            f"| {name} | {len(tr)} | {(sum(p > 0 for p in pnl) / len(pnl) * 100 if pnl else 0):.0f} % | "
            f"{_fmt_eur(sum(pnl))} | {_fmt_eur(max(pnl)) if pnl else '–'} | {_fmt_eur(min(pnl)) if pnl else '–'} | "
            f"{(_n(sum(rs) / len(rs)) + ' R') if rs else '–'} |")
        for t in tr:
            trades.append((name, bot["strategy"], t))
        pos = st.get("position")
        if pos:
            opens.append(f"| {name} | {_ts(pos['entry_time'])} | {_n(pos['entry_price'])} | "
                         f"{_n(pos['stop']) if pos.get('stop') else '–'} | {_n(pos['tp']) if pos.get('tp') else '–'} | "
                         f"{entry_reason(bot['strategy'], pos.get('context') or {})} |")
        elif st.get("holding"):
            opens.append(f"| {name} | {_ts(st['entry_time'])} | {st['holding']} | – | – | "
                         f"{entry_reason('dualmom', st.get('entry_context') or {})} |")
        elif st.get("active") and st.get("current"):
            c = st["current"]
            opens.append(f"| {name} | {_ts(c['entry_time'])} | Absicherung | – | – | "
                         f"{entry_reason('funding', c.get('context') or {})} |")

    lines = [
        "# 📒 Trading-Journal",
        "",
        f"Stand: **{now}** · automatisch bei jedem Lauf erstellt · "
        "[Dashboard](DASHBOARD.md) · [Eigene Notizen](NOTIZEN.md)",
        "",
        "## Kennzahlen je Bot",
        "",
        "| Bot | Trades | Treffer | Summe | Bester | Schlechtester | Ø R |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows_stats,
        "",
        "R = Ergebnis im Verhältnis zum eingeplanten Risiko (1 R = Verlust bis zum Stop).",
        "",
        "## Offene Positionen",
        "",
        "| Bot | Seit | Einstieg | Stop | Ziel | Grund |",
        "|---|---|---:|---:|---:|---|",
        *(opens or ["| – | – | – | – | – | keine offenen Positionen |"]),
        "",
        "## Alle Trades (neueste zuerst)",
        "",
    ]
    if not trades:
        lines += ["Noch keine abgeschlossenen Trades.", ""]
    for i, (name, strat, t) in enumerate(sorted(trades, key=lambda x: x[2]["exit_time"], reverse=True)):
        ok = "🟢" if t["pnl"] > 0 else "🔴"
        r = f" · {_n(t['r_multiple'])} R" if t.get("r_multiple") is not None else ""
        lines += [
            f"### {ok} {name} · {_fmt_eur(t['pnl'])} ({_pct(t.get('return_pct', 0))}){r}",
            "",
            f"- **Einstieg:** {_ts(t['entry_time'])}"
            + (f" zu {_n(t['entry_price'])}" if "entry_price" in t else "")
            + (f" · {t['symbol']}" if "symbol" in t else ""),
            f"- **Grund:** {entry_reason(strat, t.get('context') or {})}",
        ]
        if t.get("initial_stop") or t.get("tp"):
            lines.append(f"- **Stop / Ziel:** {_n(t['initial_stop']) if t.get('initial_stop') else '–'}"
                         f" / {_n(t['tp']) if t.get('tp') else '–'}")
        lines += [
            f"- **Ausstieg:** {_ts(t['exit_time'])}"
            + (f" zu {_n(t['exit_price'])}" if "exit_price" in t else "")
            + f" · {EXIT_REASONS.get(t['reason'], t['reason'])}",
            f"- **Dauer:** {_dur(t['entry_time'], t['exit_time'])}",
            "",
        ]

    # Tagesprotokoll: Team-Durchschnitt am Tagesende und Trades des Tages
    team = equity_series(team_average(team_curves, start_cash))
    lines += ["## Tagesprotokoll (Team)", "", "| Tag | Team Ø | Veränderung | Geöffnet | Geschlossen |",
              "|---|---:|---:|---:|---:|"]
    if not team.empty:
        daily = team.resample("1D").last().dropna()
        opened, closed = defaultdict(int), defaultdict(int)
        for _, _, t in trades:
            opened[t["entry_time"][:10]] += 1
            closed[t["exit_time"][:10]] += 1
        prev = start_cash
        rows = []
        for ts, v in daily.items():
            day = ts.strftime("%Y-%m-%d")
            rows.append(f"| {ts.strftime('%d.%m.%Y')} | {_fmt_eur(v)} | {_pct((v / prev - 1) * 100)} | "
                        f"{opened.get(day, 0)} | {closed.get(day, 0)} |")
            prev = v
        lines += rows[::-1][:60]  # neueste 60 Tage, neueste zuerst
    (DOCS / "JOURNAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
