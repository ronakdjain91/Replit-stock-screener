import os
import json
import threading
import pandas as pd
from flask import Flask, render_template, jsonify, send_file, request
from datetime import datetime
from scanner import (
    DEFAULT_NIFTY100, DEFAULT_NIFTY50,
    run_technical_scan, run_fundamental_scan
)

app = Flask(__name__)

CUSTOM_STOCKS_FILE = "custom_stocks.json"
TECH_CSV = "nifty50_signals.csv"
FUND_CSV = "nifty_fund_scan.csv"

scan_status = {"tech": "idle", "fund": "idle"}
scan_results = {"tech": None, "fund": None}


def load_custom_stocks():
    if os.path.exists(CUSTOM_STOCKS_FILE):
        with open(CUSTOM_STOCKS_FILE) as f:
            return json.load(f)
    return []


def save_custom_stocks(stocks):
    with open(CUSTOM_STOCKS_FILE, "w") as f:
        json.dump(stocks, f)


def get_tech_stocks():
    custom = load_custom_stocks()
    merged = list(dict.fromkeys(DEFAULT_NIFTY100 + custom))
    return merged


def get_fund_stocks():
    custom = load_custom_stocks()
    merged = list(dict.fromkeys(DEFAULT_NIFTY50 + custom))
    return merged


def do_tech_scan():
    scan_status["tech"] = "running"
    try:
        stocks = get_tech_stocks()
        results = run_technical_scan(stocks)
        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(TECH_CSV, index=False)
        scan_results["tech"] = results
        scan_status["tech"] = "done"
    except Exception as e:
        scan_status["tech"] = f"error:{e}"


def do_fund_scan():
    scan_status["fund"] = "running"
    try:
        stocks = get_fund_stocks()
        results = run_fundamental_scan(stocks)
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("Final Score", ascending=False)
            df.to_csv(FUND_CSV, index=False)
        scan_results["fund"] = results
        scan_status["fund"] = "done"
    except Exception as e:
        scan_status["fund"] = f"error:{e}"


def load_tech_csv():
    if not os.path.exists(TECH_CSV):
        return []
    df = pd.read_csv(TECH_CSV)
    spark_cols = [c for c in df.columns if c.startswith('_')]
    df = df.drop(columns=spark_cols, errors='ignore')
    df = df.fillna('')
    rows = df.to_dict(orient='records')
    # Normalize to clean column names
    clean = []
    for r in rows:
        stock = r.get('Stock', '')
        price = r.get('Price', r.get('Current Price', ''))
        pct = r.get('% Change', r.get('Last Day % Movement', ''))
        rsi = r.get('RSI', r.get('RSI_1', ''))
        macd_h = r.get('MACD Hist', r.get('MACD_1', ''))
        macd_sig = r.get('MACD Signal', '---')
        rsi_sig = r.get('RSI Signal', '---')
        ut = r.get('UT Bot', r.get('UT Bot Signal', '---'))
        chart = r.get('Chart', r.get('TradingView Link', ''))
        clean.append({
            'Stock': stock,
            'Price': round(float(price), 2) if price != '' else '',
            '% Change': round(float(pct), 2) if pct != '' else '',
            'RSI': round(float(rsi), 1) if rsi != '' else '',
            'MACD Hist': round(float(macd_h), 3) if macd_h != '' else '',
            'MACD Signal': macd_sig,
            'RSI Signal': rsi_sig,
            'UT Bot': ut,
            'Chart': chart
        })
    return clean


def load_fund_csv():
    if not os.path.exists(FUND_CSV):
        return []
    df = pd.read_csv(FUND_CSV)
    df = df.fillna('')
    return df.to_dict(orient='records')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data/tech")
def data_tech():
    rows = scan_results["tech"] if scan_results["tech"] is not None else load_tech_csv()
    if rows:
        rows = [{k: v for k, v in r.items() if not str(k).startswith('_')} for r in rows]
    mtime = os.path.getmtime(TECH_CSV) if os.path.exists(TECH_CSV) else None
    generated_at = datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M") if mtime else None
    return jsonify({"rows": rows or [], "generated_at": generated_at})


@app.route("/api/data/fund")
def data_fund():
    rows = scan_results["fund"] if scan_results["fund"] is not None else load_fund_csv()
    mtime = os.path.getmtime(FUND_CSV) if os.path.exists(FUND_CSV) else None
    generated_at = datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M") if mtime else None
    return jsonify({"rows": rows or [], "generated_at": generated_at})


@app.route("/api/run/<scanner>", methods=["POST"])
def run_scanner(scanner):
    if scanner == "tech" and scan_status["tech"] != "running":
        t = threading.Thread(target=do_tech_scan)
        t.daemon = True
        t.start()
        return jsonify({"status": "started"})
    elif scanner == "fund" and scan_status["fund"] != "running":
        t = threading.Thread(target=do_fund_scan)
        t.daemon = True
        t.start()
        return jsonify({"status": "started"})
    return jsonify({"status": scan_status.get(scanner, "unknown")})


@app.route("/api/status/<scanner>")
def status(scanner):
    return jsonify({"status": scan_status.get(scanner, "unknown")})


@app.route("/api/stocks", methods=["GET"])
def list_stocks():
    custom = load_custom_stocks()
    return jsonify({
        "default_tech": DEFAULT_NIFTY100,
        "default_fund": DEFAULT_NIFTY50,
        "custom": custom
    })


@app.route("/api/stocks", methods=["POST"])
def add_stock():
    body = request.get_json()
    symbol = body.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    custom = load_custom_stocks()
    if symbol in custom:
        return jsonify({"error": "Already in list", "symbol": symbol}), 409
    # Quick validate via yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = t.fast_info
        if not hasattr(info, 'last_price') or info.last_price is None:
            return jsonify({"error": f"Could not validate {symbol}"}), 400
    except Exception:
        pass
    custom.append(symbol)
    save_custom_stocks(custom)
    return jsonify({"success": True, "symbol": symbol, "custom": custom})


@app.route("/api/stocks/<symbol>", methods=["DELETE"])
def remove_stock(symbol):
    symbol = symbol.upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    custom = load_custom_stocks()
    if symbol not in custom:
        return jsonify({"error": "Not in custom list"}), 404
    custom.remove(symbol)
    save_custom_stocks(custom)
    return jsonify({"success": True, "custom": custom})


@app.route("/api/download/tech")
def download_tech():
    if not os.path.exists(TECH_CSV):
        return jsonify({"error": "No file"}), 404
    return send_file(TECH_CSV, as_attachment=True, download_name="nifty_technical_signals.csv")


@app.route("/api/download/fund")
def download_fund():
    if not os.path.exists(FUND_CSV):
        return jsonify({"error": "No file"}), 404
    return send_file(FUND_CSV, as_attachment=True, download_name="nifty_fundamental_scan.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
