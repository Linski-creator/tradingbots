"""Gold-4h-Bot, Stand v4.3 (entspricht tradingview/gold_tagestrend.pine auf dem 4h-Chart):

  Trend:   Schluss > SMA 200, SMA 50 > SMA 200, SMA 200 höher als vor 20 Kerzen (B)
  Uhrzeit: nur Signale aus Kerzen, die zwischen 12 und 24 Uhr UTC beginnen (H)
  Stop:    Start 2 ATR unter dem Einstieg, danach 3 ATR hinter dem besten Schluss (nur nach oben)
  Raus:    Stop oder Schluss unter der SMA 200
  Nachkauf (nur "Krass"): bis max_adds Zukäufe, wenn die ganze Position schon risikofrei ist
           und der Kurs 1 R über dem letzten Kauf schließt
  Crash-Schutz: über max_leverage = 1 in den Bot-Einstellungen (Position max. 1x Konto)
"""

from __future__ import annotations

from .filter_study import Filtered


class Gold4h(Filtered):
    def __init__(self, max_adds: int = 0, flags: str = "H"):
        super().__init__(50, 200, 3.0, 20, 2 / 3, flags)
        self.max_adds = max_adds

    def on_bar(self, i, a, pos):
        acts = super().on_bar(i, a, pos)
        if pos is None or pos.get("adds", 0) >= self.max_adds or any(x[0] == "exit" for x in acts):
            return acts
        r0 = pos.get("first_entry", pos["entry_price"]) - pos["initial_sl"]
        last = pos.get("last_add") or pos.get("first_entry", pos["entry_price"])
        if pos["sl"] >= pos["entry_price"] and a["close"][i] >= last + r0:
            acts = list(acts) + [("add", None)]
        return acts
