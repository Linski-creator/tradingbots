"""News-Filter: keine neuen Einstiege rund um wichtige US-Wirtschaftsdaten.

Quelle ist der öffentliche Wochenkalender von Forex Factory
(nfs.faireconomy.media). Berücksichtigt werden Termine mit hoher Wirkung
für USD (z. B. Zinsentscheid, Inflation, Arbeitsmarkt). Ist der Kalender
nicht erreichbar, läuft alles ohne Filter weiter (mit Warnung im Log).
Im Backtest gibt es keinen News-Filter, weil historische Termine fehlen.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
BEFORE = pd.Timedelta(hours=2)
AFTER = pd.Timedelta(hours=1)


def parse_events(raw: list) -> list[dict]:
    out = []
    for e in raw:
        if e.get("impact") != "High" or e.get("country") != "USD":
            continue
        try:
            ts = pd.Timestamp(e["date"]).tz_convert("UTC")
        except (KeyError, ValueError, TypeError):
            continue
        out.append({"time": ts, "title": str(e.get("title", ""))})
    return sorted(out, key=lambda e: e["time"])


def fetch_events() -> Optional[list[dict]]:
    try:
        r = requests.get(URL, timeout=20, headers={"User-Agent": "krypto-bots/1.0"})
        r.raise_for_status()
        return parse_events(r.json())
    except Exception as e:  # noqa: BLE001 - ohne Kalender einfach ohne Filter
        log.warning("Wirtschaftskalender nicht erreichbar (%s), News-Filter aus", e)
        return None


def entry_blocker(events: Optional[list[dict]]) -> Optional[Callable[[pd.Timestamp], bool]]:
    """Funktion, die für einen Zeitpunkt sagt, ob ein Einstieg gesperrt ist:
    2 Stunden vor bis 1 Stunde nach einem wichtigen Termin."""
    if not events:
        return None
    times = [e["time"] for e in events]

    def blocked(ts: pd.Timestamp) -> bool:
        return any(t - BEFORE <= ts <= t + AFTER for t in times)

    return blocked
