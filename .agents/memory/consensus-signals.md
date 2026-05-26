---
name: 4-signal consensus system
description: How buy/sell consensus is calculated from 4 technical signals.
---
**Signals:** MACD histogram crossover, RSI threshold cross (≤40 Buy, ≥60 Sell), UT Bot, 9/21 EMA cross.

**Thresholds:** Strong Buy=4/4, Buy=3/4, Strong Sell=4/4, Sell=3/4, Neutral=anything else.

**Why:** Added EMA cross (9 vs 21 EMA) as 4th signal. RSI thresholds widened from 35/55 to 40/60 for more sensitivity. Downtrend (below 200 SMA) suppresses Buy signals and sets strength=0.

**Strength pips:** JS strengthPips() shows 4 pips (was 3) with n/4 label.
