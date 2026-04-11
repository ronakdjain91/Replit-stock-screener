# Nifty Stock Scanner

A Flask web dashboard for scanning Nifty 100/50 stocks using technical and fundamental analysis.

## Project Structure

- `app.py` — Flask web server (port 5000)
- `scanner.py` — Core scan logic (technical + fundamental), importable by app.py
- `main.py` — Legacy standalone technical scanner (MACD, RSI, UT Bot, weekly data)
- `2nd.py` — Legacy standalone fundamental + technical scorer
- `templates/index.html` — Web UI (dark theme, DataTables, filterable)
- `nifty50_signals.csv` — Technical scan output
- `nifty_fund_scan.csv` — Fundamental scan output
- `custom_stocks.json` — User-added custom stocks (persisted)

## Web UI Features

### Technical Signals Tab (Nifty 100)
- Summary stats cards: Total, MACD Buy/Sell, RSI Buy/Sell, UT Bot Buy
- Clean table: Stock, % Change, Price, RSI (color-coded), MACD Hist, MACD Signal, RSI Signal, UT Bot
- Filters: MACD signal, RSI signal, UT Bot signal, RSI Range (oversold/healthy/overbought), % Move
- Export filtered CSV, Download full CSV

### Fund + Tech Score Tab (Nifty 50)
- Summary stats: Total, Buy/Hold/Sell counts, Avg Fund/Tech scores
- Clean table: Symbol, Price, P/E, D/E, ROE%, RSI, Fund Score (bar), Tech Score (bar), Final Score (bar), Recommendation
- Filters: Recommendation, Min Fund Score, Min Final Score
- Export filtered CSV, Download full CSV

### Manage Stocks
- Add custom stocks (validated via yfinance, auto-appends .NS)
- Remove custom stocks from list
- View all default Nifty 100 and Nifty 50 stocks
- Custom stocks included in next scan run

## Running

The Flask app runs on port 5000 via the "Start application" workflow.
`python app.py`

## Dependencies

- Flask, pandas, numpy, yfinance, pandas-ta
