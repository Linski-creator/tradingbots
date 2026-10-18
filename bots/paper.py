"""Paper-Trading: wird stündlich von GitHub Actions gestartet.

Pro Bot:
  1. Zustand aus state/<bot>.json laden (Konto, Position, offene Aufträge)
  2. aktuelle, abgeschlossene Kerzen laden
  3. NUR die neuen Kerzen mit genau demselben Simulator wie im Backtest
     verarbeiten (verpasste Läufe werden automatisch nachgeholt)
  4. Zustand speichern und das Dashboard (docs/DASHBOARD.md) neu schreiben

Beim allerersten Lauf startet ein Bot "jetzt": Die vorhandene Historie dient
nur zum Aufwärmen der Indikatoren, gehandelt wird ab der nächsten Kerze.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import data as D
from .dashboard import write_dashboard
from .engine import Costs, RiskConfig, SimState, Simulator
from .portfolio import (DualMomentum, FundingArbitrage, FundingState, RotationState,
                        run_funding, run_rotation)
from .strategies import SINGLE_MARKET

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
CONFIG = ROOT / "bots.json"
BPY = {"1h": 8760, "4h": 2190, "1d": 365}
WARMUP_BARS = {"1h": 1000, "4h": 1000, "1d": 450}


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _save(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)  # atomar: nie eine halb geschriebene Datei


def _risk(cfg: dict, bot: dict | None = None) -> RiskConfig:
    if bot is not None and bot.get("risk") == "ohne":
        return RiskConfig(enabled=False)
    r = cfg.get("risk", {})
    return RiskConfig(enabled=True, risk_per_trade=r.get("risk_per_trade", 0.01),
                      target_vol=r.get("target_vol", 0.30), max_exposure=1.0,
                      halt_drawdown=r.get("halt_drawdown", 0.10))


def run_single_bot(bot: dict, cfg: dict) -> dict:
    strat = SINGLE_MARKET[bot["strategy"]]()
    itv = strat.timeframe
    df = D.recent_klines(bot["symbol"], itv, WARMUP_BARS[itv])
    if len(df) < strat.warmup + 5:
        raise RuntimeError(f"zu wenig Daten für {bot['id']}: {len(df)} Kerzen")
    prepared = strat.prepare(df, BPY[itv])
    path = STATE / f"{bot['id']}.json"
    raw = _load(path)
    if raw is None:
        st = SimState(cash=cfg["start_cash"], peak=cfg["start_cash"], last_ts=df.index[-1].isoformat())
        log.info("%s: neu gestartet ab %s", bot["id"], st.last_ts)
    else:
        st = SimState.from_dict(raw)
        if pd.Timestamp(st.last_ts) < df.index[0]:
            log.warning("%s: Datenlücke - letzte Kerze %s älter als geladene Daten", bot["id"], st.last_ts)
    st = Simulator(strat, Costs(), _risk(cfg, bot), state=st).run(prepared)
    _save(path, st.to_dict())
    return {"bot": bot, "label": strat.label, "state": st.to_dict(), "price": float(df["close"].iloc[-1])}


def run_rotation_bot(bot: dict, cfg: dict) -> dict:
    dm = DualMomentum()
    universe = {s: D.recent_klines(s, "1d", WARMUP_BARS["1d"]) for s in bot["universe"]}
    prepared = dm.prepare(universe, BPY["1d"])
    path = STATE / f"{bot['id']}.json"
    raw = _load(path)
    common = None
    for d in prepared.values():
        common = d.index if common is None else common.intersection(d.index)
    if raw is None:
        st = RotationState(cash=cfg["start_cash"], peak=cfg["start_cash"], last_ts=common[-1].isoformat())
    else:
        st = RotationState(**raw)
    st = run_rotation(dm, prepared, Costs(), _risk(cfg), state=st)
    _save(path, asdict(st))
    prices = {s: float(d["close"].iloc[-1]) for s, d in universe.items()}
    return {"bot": bot, "label": dm.label, "state": asdict(st), "prices": prices}


def run_funding_bot(bot: dict, cfg: dict) -> dict:
    fa = FundingArbitrage()
    rates = D.recent_funding(bot["symbol"], 30)
    path = STATE / f"{bot['id']}.json"
    raw = _load(path)
    if raw is None:
        st = FundingState(cash=cfg["start_cash"], peak=cfg["start_cash"], last_ts=rates.index[-1].isoformat())
    else:
        st = FundingState(**raw)
    st = run_funding(fa, rates, state=st)
    _save(path, asdict(st))
    return {"bot": bot, "label": fa.label, "state": asdict(st)}


RUNNERS = {"single": run_single_bot, "rotation": run_rotation_bot, "funding": run_funding_bot}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = json.loads(CONFIG.read_text())
    STATE.mkdir(exist_ok=True)
    reports, errors = [], []
    for bot in cfg["bots"]:
        try:
            reports.append(RUNNERS[bot["kind"]](bot, cfg))
        except Exception as e:  # noqa: BLE001 - ein Bot darf die anderen nicht stoppen
            log.exception("Bot %s fehlgeschlagen", bot["id"])
            errors.append({"bot": bot["id"], "error": str(e)})
    write_dashboard(reports, errors, cfg)
    return 1 if errors and not reports else 0


if __name__ == "__main__":
    sys.exit(main())
