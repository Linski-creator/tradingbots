# Krypto-Bots: Backtest und Paper-Trading

8 regelbasierte Strategien, die mit **Spielgeld** (je 1.000 €) gegen echte Live-Kurse
handeln. Es gibt keine echten Trades, kein Börsenkonto und keine API-Schlüssel.

👉 **Aktueller Stand: [docs/DASHBOARD.md](docs/DASHBOARD.md)**. Lässt sich auch auf dem iPad öffnen.

## Die Bots

| Bot | Strategie | Typ | Zeitebene |
|---|---|---|---|
| dualmom | Dual Momentum: stärkster Markt aus BTC/ETH/Gold-Token, sonst Cash | Trend + Rotation | 1 Tag |
| sma200 | Über der 200-Tage-Linie investiert, darunter Cash | Trend | 1 Tag |
| tsmom | Investiert, solange die 90-Tage-Rendite positiv ist | Trend | 1 Tag |
| donchian | Neues 20-Tage-Hoch kaufen, Stop 2 ATR, raus bei 10-Tage-Tief | Trend | 1 Tag |
| chandelier | EMA 20/50-Kreuzung über EMA 200, nachgezogener Stop 3 ATR | Trend | 4 Stunden |
| rsi2 | RSI(2) < 10 über SMA 200 kaufen, raus bei RSI > 70 oder nach 10 Tagen | Gegenbewegung | 1 Tag |
| pullback | Rücksetzer auf EMA 20 im Aufwärtstrend, Stop unter dem Swing-Tief, Ziel 3:1 | Trend-Pullback | 1 Stunde |
| funding | Spot long + Perp short, kassiert positive Funding-Raten | marktneutral | 8 Stunden |

Welcher Bot auf welchem Markt läuft, steht in [`bots.json`](bots.json).

## Risikoregeln (für alle Bots)

- Nur Spot, nur Long oder Cash: **kein Hebel, keine Liquidation**
- Höchstens **1 % Kontorisiko** pro Trade (Abstand zum Stop inkl. Gebühren)
- Trend-Filter-Bots: Positionsgröße nach Volatilität (Ziel 30 % p. a.), nie über 100 %
- **Notbremse:** Fällt ein Bot 10 % unter seinen Höchststand, schließt er und stoppt
- Gebühren 0,1 % pro Seite plus 0,05 % Slippage bei jedem Trade

## Wie es läuft

GitHub Actions startet die Bots jede Stunde ([`.github/workflows/paper.yml`](.github/workflows/paper.yml)):

1. Alle Tests laufen. Ist einer rot, handelt kein Bot.
2. Jeder Bot lädt seinen Zustand aus `state/`, holt die neuen, abgeschlossenen Kerzen und
   verarbeitet sie mit **demselben Simulator wie der Backtest**.
3. Zustand und Dashboard werden gespeichert. Verpasste Läufe werden beim nächsten Mal nachgeholt.

**Manuell starten:** Tab *Actions*, dann *Paper-Trading*, dann *Run workflow*.

## Ausführungsregeln (gegen geschönte Ergebnisse)

- Ein Signal entsteht zum Schluss einer Kerze und wird erst zum **Open der nächsten** ausgeführt.
- Stop und Ziel in derselben Kerze: Es zählt der **Stop**.
- Öffnet eine Kerze schon jenseits des Stops (Kurslücke), wird zum Open verkauft.
- Indikatoren sind auf "Blick in die Zukunft" getestet (`tests/test_strategies.py`).

## Backtest

```bash
pip install -r requirements.txt
python -m bots.backtest      # lädt Historie von Binance Vision, schreibt results/
python -m pytest -q          # alle Tests
```

## Bekannte Vereinfachungen

- Funding-Arbitrage: Die Absicherung gilt als perfekt. Basis-Schwankungen und Börsenrisiko
  sind nicht enthalten. Live kommen die Funding-Raten von Binance oder ersatzweise Kraken Futures.
- EMA und ATR werden im Paper-Trading auf einem Fenster von 450 bis 1.000 Kerzen berechnet.
  Die Abweichung zum Backtest liegt unter 0,0001 %.
- Vol-Sizing legt die Positionsgröße beim Einstieg fest. Während des Trades wird nicht umgeschichtet.
