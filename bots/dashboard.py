"""Schreibt docs/DASHBOARD.md und docs/equity.svg.

Markdown und SVG zeigt GitHub direkt an, auch in privaten Repos und auf
dem iPad (Safari oder GitHub-App), ganz ohne Webserver.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .metrics import equity_series, max_drawdown

DOCS = Path(__file__).resolve().parent.parent / "docs"
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#4b5563",
          "#65a30d", "#9333ea"]


def _fmt_eur(x: float) -> str:
    return f"{x:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(x: float) -> str:
    return f"{x:+.2f} %".replace(".", ",")


def _svg(curves: list[tuple[str, list]], start_cash: float, w: int = 900, h: int = 380) -> str:
    pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 30
    series = [(n, equity_series(e)) for n, e in curves if e]
    if not series:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="60"><text x="10" y="35" ' \
               f'font-family="sans-serif" font-size="14">Noch keine Daten</text></svg>'
    t0 = min(s.index[0] for _, s in series).timestamp()
    t1 = max(s.index[-1] for _, s in series).timestamp()
    lo = min(min(s.min() for _, s in series), start_cash)
    hi = max(max(s.max() for _, s in series), start_cash)
    span_v = (hi - lo) or 1.0
    span_t = (t1 - t0) or 1.0

    def X(t):
        return pad_l + (t - t0) / span_t * (w - pad_l - pad_r)

    def Y(v):
        return pad_t + (hi - v) / span_v * (h - pad_t - pad_b)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h + 24 * ((len(series) + 3) // 4)}" '
             f'font-family="sans-serif" font-size="12">',
             f'<rect width="100%" height="100%" fill="#ffffff"/>']
    for k in range(5):
        v = lo + span_v * k / 4
        y = Y(v)
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#6b7280">{v:,.0f}</text>')
    yb = Y(start_cash)
    parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{yb:.1f}" y2="{yb:.1f}" stroke="#9ca3af" '
                 f'stroke-dasharray="4 3"/>')
    for i, (name, s) in enumerate(series):
        step = max(1, len(s) // 800)
        pts = " ".join(f"{X(t.timestamp()):.1f},{Y(v):.1f}" for t, v in list(s.items())[::step] + [(s.index[-1], s.iloc[-1])])
        c = COLORS[i % len(COLORS)]
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6" points="{pts}"/>')
        lx, ly = 10 + (i % 4) * 220, h + 16 + 24 * (i // 4)
        parts.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="3" fill="{c}"/>'
                     f'<text x="{lx + 18}" y="{ly}" fill="#111827">{name}</text>')
    d0 = datetime.fromtimestamp(t0, timezone.utc).strftime("%d.%m.%Y")
    d1 = datetime.fromtimestamp(t1, timezone.utc).strftime("%d.%m.%Y %H:%M")
    parts.append(f'<text x="{pad_l}" y="{h - 8}" fill="#6b7280">{d0}</text>')
    parts.append(f'<text x="{w - pad_r}" y="{h - 8}" text-anchor="end" fill="#6b7280">{d1} UTC</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _position_text(r: dict) -> str:
    st, kind = r["state"], r["bot"]["kind"]
    if kind == "single":
        p = st.get("position")
        if not p:
            return "Cash"
        txt = f"Long ab {p['entry_price']:,.2f}"
        if p.get("stop"):
            txt += f", Stop {p['stop']:,.2f}"
        if p.get("tp"):
            txt += f", Ziel {p['tp']:,.2f}"
        return txt
    if kind == "rotation":
        return f"hält {st['holding']}" if st.get("holding") else "Cash"
    return "Absicherung aktiv" if st.get("active") else "wartet (Funding zu niedrig)"


def team_average(curves: list[list], start_cash: float) -> list:
    """Durchschnittliches Konto der Team-Bots, stündlich. Vor dem ersten
    Wert eines Bots zählt sein Startkapital."""
    series = [equity_series(e) for e in curves if e]
    if not series:
        return []
    start = min(s.index[0] for s in series).floor("h")
    end = max(s.index[-1] for s in series).ceil("h")
    grid = pd.date_range(start, end, freq="1h")
    aligned = [s.resample("1h").last().reindex(grid).ffill().fillna(start_cash) for s in series]
    avg = sum(aligned) / len(aligned)
    return [(ts.isoformat(), float(v)) for ts, v in avg.items()]


def write_dashboard(reports: list[dict], errors: list[dict], cfg: dict) -> None:
    DOCS.mkdir(exist_ok=True)
    start_cash = cfg["start_cash"]
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    rows, curves, all_trades, team_curves = [], [], [], []
    total, n = 0.0, 0
    for r in reports:
        st, bot = r["state"], r["bot"]
        eq = st.get("equity") or []
        cur = eq[-1][1] if eq else start_cash
        if not bot.get("compare"):
            total += cur
            n += 1
            team_curves.append(eq)
        mdd = max_drawdown(equity_series(eq)) * 100 if eq else 0.0
        trades = st.get("trades", [])
        wins = sum(1 for t in trades if t["pnl"] > 0)
        status = "⛔ Notbremse" if st.get("halted") else "✅ läuft"
        name = f"{bot['id']} (Vergleich)" if bot.get("compare") else bot["id"]
        rows.append(f"| {name} | {r['label']} | {bot.get('symbol', '/'.join(bot.get('universe', [])))} | "
                    f"{_fmt_eur(cur)} | {_pct((cur / start_cash - 1) * 100)} | {mdd:.1f} % | {len(trades)} | "
                    f"{(wins / len(trades) * 100 if trades else 0):.0f} % | {_position_text(r)} | {status} |")
        curves.append((bot["id"], eq))
        for t in trades:
            all_trades.append((t["exit_time"], bot["id"], t))

    n = n or 1
    team = team_average(team_curves, start_cash)
    team_dd = max_drawdown(equity_series(team)) * 100 if team else 0.0
    svg = _svg([("TEAM (Durchschnitt)", team)] + curves, start_cash)
    (DOCS / "equity.svg").write_text(svg, encoding="utf-8")
    lines = [
        "# 📈 Paper-Trading Dashboard",
        "",
        f"Stand: **{now}** · Spielgeld, keine echten Trades · Start je Bot: {_fmt_eur(start_cash)}",
        "",
        f"**Team: {_fmt_eur(total)}** ({_pct((total / (start_cash * n) - 1) * 100)} auf {n} Bots) · "
        f"größter Team-Einbruch: **{team_dd:.1f} %** · Risikostufe: **{cfg.get('level', '-')}**",
        "",
        "Der Vergleichs-Bot (Strategie aus einem YouTube-Video) zählt nicht zum Team.",
        *([f"", f"⛔ **TEAM-NOTBREMSE AKTIV** seit {cfg['_team'].get('halted_at', '')[:16].replace('T', ' ')} UTC: "
           "alle Team-Bots geschlossen und gestoppt. Wieder einschalten: `state/team.json` löschen."]
          if cfg.get("_team", {}).get("halted") else []),
        *([f"", "📅 **News-Filter** (keine neuen Einstiege 2 h vor bis 1 h nach): "
           + " · ".join(f"{e['time'].strftime('%a %d.%m. %H:%M')} UTC {e['title']}" for e in cfg["_news"][:6])]
          if cfg.get("_news") else []),
        "",
        "📒 **[Trading-Journal](JOURNAL.md)** (Grund jedes Trades) · ✍️ **[Eigene Notizen](NOTIZEN.md)**",
        "",
        "![Kontoverlauf](equity.svg)",
        "",
        "| Bot | Strategie | Markt | Konto | Ergebnis | Max. Einbruch | Trades | Treffer | Position | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        *rows,
        "",
    ]
    if errors:
        lines += ["## ⚠️ Fehler beim letzten Lauf", ""]
        lines += [f"- **{e['bot']}**: {e['error']}" for e in errors]
        lines += [""]
    lines += ["## Letzte Trades", "", "| Geschlossen | Bot | Einstieg | Ausstieg | Gewinn | Grund |",
              "|---|---|---:|---:|---:|---|"]
    for ts, bot_id, t in sorted(all_trades, key=lambda x: x[0], reverse=True)[:30]:
        when = ts[:16].replace("T", " ")
        ep = f"{t['entry_price']:,.2f}" if "entry_price" in t else "–"
        xp = f"{t['exit_price']:,.2f}" if "exit_price" in t else "–"
        lines.append(f"| {when} | {bot_id} | {ep} | {xp} | {_fmt_eur(t['pnl'])} | {t['reason']} |")
    if not all_trades:
        lines.append("| – | – | – | – | – | noch keine abgeschlossenen Trades |")
    (DOCS / "DASHBOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
