import os
import io
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
scan_params = {
    "tech": {"period": "1y", "interval": "1wk"},
    "fund": {"period": "1y", "interval": "1d"}
}


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
    return list(dict.fromkeys(DEFAULT_NIFTY100 + custom))


def get_fund_stocks():
    custom = load_custom_stocks()
    return list(dict.fromkeys(DEFAULT_NIFTY50 + custom))


def do_tech_scan(period, interval):
    scan_status["tech"] = "running"
    scan_params["tech"] = {"period": period, "interval": interval}
    try:
        stocks = get_tech_stocks()
        results = run_technical_scan(stocks, period=period, interval=interval)
        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(TECH_CSV, index=False)
        scan_results["tech"] = results
        scan_status["tech"] = "done"
    except Exception as e:
        scan_status["tech"] = f"error:{e}"


def do_fund_scan(period, interval):
    scan_status["fund"] = "running"
    scan_params["fund"] = {"period": period, "interval": interval}
    try:
        stocks = get_fund_stocks()
        results = run_fundamental_scan(stocks, period=period, interval=interval)
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
    df = df.drop(columns=[c for c in df.columns if c.startswith('_')], errors='ignore')
    df = df.fillna('')
    clean = []
    for r in df.to_dict(orient='records'):
        stock = r.get('Stock', '')
        price = r.get('Price', r.get('Current Price', ''))
        pct = r.get('% Change', r.get('Last Day % Movement', ''))
        rsi = r.get('RSI', r.get('RSI_1', ''))
        macd_h = r.get('MACD Hist', r.get('MACD_1', ''))
        clean.append({
            'Stock': stock,
            'Price': round(float(price), 2) if price != '' else '',
            '% Change': round(float(pct), 2) if pct != '' else '',
            'RSI': round(float(rsi), 1) if rsi != '' else '',
            'MACD Hist': round(float(macd_h), 3) if macd_h != '' else '',
            'MACD Signal': r.get('MACD Signal', '---'),
            'RSI Signal': r.get('RSI Signal', '---'),
            'UT Bot': r.get('UT Bot', r.get('UT Bot Signal', '---')),
            'Chart': r.get('Chart', r.get('TradingView Link', ''))
        })
    return clean


def load_fund_csv():
    if not os.path.exists(FUND_CSV):
        return []
    df = pd.read_csv(FUND_CSV)
    df = df.fillna('')
    return df.to_dict(orient='records')


def normalize_symbol(raw):
    sym = raw.strip().upper().replace(" ", "")
    if not sym:
        return None
    if not sym.endswith(".NS"):
        sym += ".NS"
    return sym


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
    return jsonify({
        "rows": rows or [],
        "generated_at": generated_at,
        "params": scan_params["tech"]
    })


@app.route("/api/data/fund")
def data_fund():
    rows = scan_results["fund"] if scan_results["fund"] is not None else load_fund_csv()
    mtime = os.path.getmtime(FUND_CSV) if os.path.exists(FUND_CSV) else None
    generated_at = datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M") if mtime else None
    return jsonify({
        "rows": rows or [],
        "generated_at": generated_at,
        "params": scan_params["fund"]
    })


@app.route("/api/run/<scanner>", methods=["POST"])
def run_scanner(scanner):
    body = request.get_json(silent=True) or {}
    period = body.get("period", "1y")
    interval = body.get("interval", "1wk" if scanner == "tech" else "1d")

    if scanner == "tech" and scan_status["tech"] != "running":
        t = threading.Thread(target=do_tech_scan, args=(period, interval))
        t.daemon = True
        t.start()
        return jsonify({"status": "started"})
    elif scanner == "fund" and scan_status["fund"] != "running":
        t = threading.Thread(target=do_fund_scan, args=(period, interval))
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
    symbol = normalize_symbol(body.get("symbol", ""))
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    custom = load_custom_stocks()
    if symbol in custom:
        return jsonify({"error": "Already in list", "symbol": symbol}), 409
    custom.append(symbol)
    save_custom_stocks(custom)
    return jsonify({"success": True, "symbol": symbol, "custom": custom})


@app.route("/api/stocks/upload", methods=["POST"])
def upload_stocks():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.csv'):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        content = f.read().decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

    # Find the column most likely to contain symbols
    symbol_col = None
    priority = ['symbol', 'stock', 'ticker', 'scrip', 'name', 'code']
    lower_cols = {c.lower(): c for c in df.columns}
    for p in priority:
        if p in lower_cols:
            symbol_col = lower_cols[p]
            break
    if symbol_col is None:
        symbol_col = df.columns[0]

    raw_symbols = df[symbol_col].dropna().astype(str).tolist()
    custom = load_custom_stocks()
    all_known = set(DEFAULT_NIFTY100 + DEFAULT_NIFTY50 + custom)

    added, skipped, invalid = [], [], []
    for raw in raw_symbols:
        sym = normalize_symbol(raw)
        if not sym:
            continue
        if sym in all_known or sym in [normalize_symbol(c) for c in custom]:
            skipped.append(sym)
        elif len(sym) > 20 or not sym.replace('.NS','').replace('-','').replace('&','').isalnum():
            invalid.append(sym)
        else:
            if sym not in custom:
                custom.append(sym)
                added.append(sym)
            else:
                skipped.append(sym)

    save_custom_stocks(custom)
    return jsonify({
        "success": True,
        "added": added,
        "skipped": skipped,
        "invalid": invalid,
        "total_custom": len(custom)
    })


@app.route("/api/stocks/<symbol>", methods=["DELETE"])
def remove_stock(symbol):
    symbol = normalize_symbol(symbol) or symbol.upper()
    custom = load_custom_stocks()
    if symbol not in custom:
        return jsonify({"error": "Not in custom list"}), 404
    custom.remove(symbol)
    save_custom_stocks(custom)
    return jsonify({"success": True, "custom": custom})


@app.route("/api/stocks/clear", methods=["DELETE"])
def clear_custom_stocks():
    save_custom_stocks([])
    return jsonify({"success": True, "custom": []})


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
