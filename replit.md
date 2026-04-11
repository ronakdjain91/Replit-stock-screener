# Nifty Stock Scanner

A Flask web dashboard for scanning Nifty 100/50 stocks using technical and fundamental analysis.

## Project Structure

- `app.py` — Flask web server (port 5000)
- `main.py` — Nifty 100 technical scanner (MACD, RSI, UT Bot signals, weekly data)
- `2nd.py` — Nifty 50 fundamental + technical scorer (daily data, P/E, D/E, ROE, Revenue)
- `templates/index.html` — Web UI (dark theme, DataTables, filterable)
- `nifty50_signals.csv` — Output from main.py

## Web UI Features

- Two tabs: Nifty 100 Technical Signals + Nifty 50 Fund+Tech Score
- Filterable tables (by MACD/RSI/UT Bot signal or Recommendation)
- Quick search across all columns
- Export filtered CSV via DataTables
- Download full CSV from server
- Run Scanner buttons to trigger live scans in background

## Running

The Flask app runs on port 5000 via the "Start application" workflow.
`python app.py`

## Dependencies

- Flask, pandas, numpy, yfinance, pandas-ta
