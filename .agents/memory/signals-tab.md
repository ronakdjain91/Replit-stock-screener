---
name: Signals tab architecture
description: How the unified Buy/Sell Calls tab is built — merged tech+fund data via /api/data/signals.
---
The old three tabs (Technical, Fund+Tech, Paper) were replaced with two tabs (Buy/Sell Calls, Paper Trade).

**How:** /api/data/signals in app.py merges tech rows (Nifty 100) with fund rows (Nifty 50 subset) keyed by symbol. Fund fields (Fund Score, Final Score, Recommendation, Sector, P/E, D/E, ROE) are null for stocks not in Nifty 50.

**Why:** Users wanted a single view combining technical signals with fundamental context, eliminating tab-switching.

**How to apply:** loadData() always fetches /api/data/signals regardless of which scan name triggered it. The signals table has 15 columns (Stock, Price, %Chg, Consensus, Strength, Trend, RSI, MACD, UT Bot, EMA X, Vol, Fund Score, Rec, Final Score, Chart).

The HTML element IDs use ss- prefix for stats (ss-total, ss-strong-buy, ss-buy, ss-sell, ss-vol, ss-fund). Filter IDs: f-consensus, f-trend, f-volspike, f-rec, f-pct. Period/Interval pill IDs: tech-period-pills, tech-interval-pills, fund-period-pills, fund-interval-pills.
