import os
import io
import math
import json
import threading
import pandas as pd
from flask import Flask, render_template, jsonify, send_file, request, Response
from datetime import datetime
from scanner import (
    DEFAULT_NIFTY100, DEFAULT_NIFTY50,
    run_technical_scan, run_fundamental_scan,
    safe_float
)

app = Flask(__name__)

CUSTOM_STOCKS_FILE = "custom_stocks.json"
TECH_CSV = "nifty50_signals.csv"
FUND_CSV = "nifty_fund_scan.csv"

scan_status   = {"tech": "idle", "fund": "idle"}
scan_results  = {"tech": None,   "fund": None}
scan_progress = {"tech": {"current": 0, "total": 0},
                 "fund": {"current": 0, "total": 0}}
scan_params   = {
    "tech": {"period": "1y", "interval": "1wk"},
    "fund": {"period": "1y", "interval": "1d"}
}


# ── JSON sanitizer ────────────────────────────────────────────────────────────
def _clean_value(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def clean_rows(rows):
    return [{k: _clean_value(v) for k, v in row.items()} for row in rows]


def safe_jsonify(payload):
    text = json.dumps(
        payload,
        default=lambda v: None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
    )
    return Response(text, mimetype='application/json')


# ── Stock management ──────────────────────────────────────────────────────────
def load_custom_stocks():
    if os.path.exists(CUSTOM_STOCKS_FILE):
        try:
            with open(CUSTOM_STOCKS_FILE) as f:
                return json.load(f)
        except Exception:
            return []
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


# ── Background scan threads ───────────────────────────────────────────────────
def do_tech_scan(period, interval):
    scan_status["tech"] = "running"
    scan_params["tech"] = {"period": period, "interval": interval}
    scan_progress["tech"] = {"current": 0, "total": 0}
    try:
        stocks = get_tech_stocks()
        scan_progress["tech"]["total"] = len(stocks)

        def progress_cb(current, total):
            scan_progress["tech"] = {"current": current, "total": total}

        results = run_technical_scan(stocks, period=period, interval=interval,
                                     progress_cb=progress_cb)
        if results:
            df = pd.DataFrame(results)
            df.to_csv(TECH_CSV, index=False)
        scan_results["tech"] = clean_rows(results)
        scan_status["tech"]  = "done"
    except Exception as e:
        print(f"[do_tech_scan] {e}")
        scan_status["tech"] = f"error:{e}"


def do_fund_scan(period, interval):
    scan_status["fund"] = "running"
    scan_params["fund"] = {"period": period, "interval": interval}
    scan_progress["fund"] = {"current": 0, "total": 0}
    try:
        stocks = get_fund_stocks()
        scan_progress["fund"]["total"] = len(stocks)

        def progress_cb(current, total):
            scan_progress["fund"] = {"current": current, "total": total}

        results = run_fundamental_scan(stocks, period=period, interval=interval,
                                       progress_cb=progress_cb)
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("Final Score", ascending=False)
            df.to_csv(FUND_CSV, index=False)
        scan_results["fund"] = clean_rows(results)
        scan_status["fund"]  = "done"
    except Exception as e:
        print(f"[do_fund_scan] {e}")
        scan_status["fund"] = f"error:{e}"


# ── CSV loaders ───────────────────────────────────────────────────────────────
def _float_or_none(v):
    if v is None or v == '' or (isinstance(v, float) and math.isnan(v)):
        return None
    return safe_float(v)


def _str_or(row, key, default='---'):
    val = row.get(key, '')
    return str(val) if val and str(val) not in ('', 'nan', 'None') else default


def load_tech_csv():
    if not os.path.exists(TECH_CSV):
        return []
    try:
        df = pd.read_csv(TECH_CSV, dtype=str)
    except Exception:
        return []
    rows = []
    for _, r in df.iterrows():
        stock = r.get('Stock', '')
        if not stock:
            continue
        price = _float_or_none(r.get('Price', ''))
        pct   = _float_or_none(r.get('% Change', ''))
        rsi   = _float_or_none(r.get('RSI', ''))
        macdh = _float_or_none(r.get('MACD Hist', ''))
        rows.append({
            'Stock':        str(stock),
            'Consensus':    _str_or(r, 'Consensus', 'Neutral'),
            'Strength':     _float_or_none(r.get('Strength', '')) or 0,
            'Trend':        _str_or(r, 'Trend', '---'),
            'Vol Spike':    _str_or(r, 'Vol Spike', 'No'),
            'Vol Ratio':    _float_or_none(r.get('Vol Ratio', '')),
            'Price':        round(price, 2) if price is not None else None,
            '% Change':     round(pct,   2) if pct   is not None else None,
            'RSI':          round(rsi,   1) if rsi   is not None else None,
            'MACD Hist':    round(macdh, 3) if macdh is not None else None,
            'MACD Signal':  _str_or(r, 'MACD Signal', '---'),
            'RSI Signal':   _str_or(r, 'RSI Signal',  '---'),
            'UT Bot':       _str_or(r, 'UT Bot',       '---'),
            'Beta':         _float_or_none(r.get('Beta', '')),
            '200 SMA':      _float_or_none(r.get('200 SMA', '')),
            'Last Scanned': _str_or(r, 'Last Scanned', ''),
            'Chart':        _str_or(r, 'Chart', ''),
        })
    return rows


def load_fund_csv():
    if not os.path.exists(FUND_CSV):
        return []
    try:
        df = pd.read_csv(FUND_CSV, dtype=str)
    except Exception:
        return []
    rows = []
    for _, r in df.iterrows():
        sym = r.get('Symbol', '')
        if not sym:
            continue
        rows.append({
            'Symbol':         str(sym),
            'Sector':         _str_or(r, 'Sector', '---'),
            'Price':          _float_or_none(r.get('Price', '')),
            'P/E':            _float_or_none(r.get('P/E', '')),
            'Sector Med P/E': _float_or_none(r.get('Sector Med P/E', '')),
            'D/E':            _float_or_none(r.get('D/E', '')),
            'ROE':            _float_or_none(r.get('ROE', '')),
            'RSI':            _float_or_none(r.get('RSI', '')),
            'Fund Score':     _float_or_none(r.get('Fund Score', '')),
            'Tech Score':     _float_or_none(r.get('Tech Score', '')),
            'Final Score':    _float_or_none(r.get('Final Score', '')),
            'Recommendation': _str_or(r, 'Recommendation', '---'),
            'Chart':          _str_or(r, 'Chart', ''),
        })
    return rows


def normalize_symbol(raw):
    sym = str(raw).strip().upper().replace(" ", "")
    if not sym:
        return None
    if not sym.endswith(".NS"):
        sym += ".NS"
    return sym


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data/tech")
def data_tech():
    rows = scan_results["tech"] if scan_results["tech"] is not None else load_tech_csv()
    mtime = os.path.getmtime(TECH_CSV) if os.path.exists(TECH_CSV) else None
    generated_at = datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M") if mtime else None
    return safe_jsonify({"rows": rows or [], "generated_at": generated_at, "params": scan_params["tech"]})


@app.route("/api/data/fund")
def data_fund():
    rows = scan_results["fund"] if scan_results["fund"] is not None else load_fund_csv()
    mtime = os.path.getmtime(FUND_CSV) if os.path.exists(FUND_CSV) else None
    generated_at = datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M") if mtime else None
    return safe_jsonify({"rows": rows or [], "generated_at": generated_at, "params": scan_params["fund"]})


@app.route("/api/run/<scanner>", methods=["POST"])
def run_scanner(scanner):
    body     = request.get_json(silent=True) or {}
    period   = body.get("period", "1y")
    interval = body.get("interval", "1wk" if scanner == "tech" else "1d")

    if scanner == "tech" and scan_status["tech"] != "running":
        t = threading.Thread(target=do_tech_scan, args=(period, interval), daemon=True)
        t.start()
        return jsonify({"status": "started"})
    elif scanner == "fund" and scan_status["fund"] != "running":
        t = threading.Thread(target=do_fund_scan, args=(period, interval), daemon=True)
        t.start()
        return jsonify({"status": "started"})
    return jsonify({"status": scan_status.get(scanner, "unknown")})


@app.route("/api/status/<scanner>")
def status(scanner):
    return jsonify({"status": scan_status.get(scanner, "unknown")})


@app.route("/api/progress/<scanner>")
def progress(scanner):
    return jsonify(scan_progress.get(scanner, {"current": 0, "total": 0}))


@app.route("/api/stocks", methods=["GET"])
def list_stocks():
    custom = load_custom_stocks()
    return jsonify({"default_tech": DEFAULT_NIFTY100, "default_fund": DEFAULT_NIFTY50, "custom": custom})


@app.route("/api/stocks", methods=["POST"])
def add_stock():
    body   = request.get_json(silent=True) or {}
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
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    symbol_col = None
    priority   = ['symbol', 'stock', 'ticker', 'scrip', 'name', 'code']
    lower_cols = {c.lower(): c for c in df.columns}
    for p in priority:
        if p in lower_cols:
            symbol_col = lower_cols[p]
            break
    if symbol_col is None:
        symbol_col = df.columns[0]

    raw_symbols = df[symbol_col].dropna().astype(str).tolist()
    custom      = load_custom_stocks()
    all_known   = set(DEFAULT_NIFTY100 + DEFAULT_NIFTY50 + custom)
    added, skipped, invalid = [], [], []

    for raw in raw_symbols:
        sym  = normalize_symbol(raw)
        if not sym:
            continue
        base = sym.replace('.NS', '').replace('-', '').replace('&', '')
        if not base.isalnum() or len(sym) > 24:
            invalid.append(sym)
        elif sym in all_known:
            skipped.append(sym)
        else:
            if sym not in custom:
                custom.append(sym)
                added.append(sym)
            else:
                skipped.append(sym)

    save_custom_stocks(custom)
    return jsonify({"success": True, "added": added, "skipped": skipped,
                    "invalid": invalid, "total_custom": len(custom)})


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
