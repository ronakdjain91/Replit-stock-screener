import curl_cffi.requests as requests_cffi
_orig_request = requests_cffi.Session.request
def _patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, *args, **kwargs)
requests_cffi.Session.request = _patched_request

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
import paper_trade as pt

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

def load_stocks_from_folder():
    folder = "StockList"
    stocks = []
    if os.path.exists(folder):
        import glob
        for f in glob.glob(os.path.join(folder, "*.csv")):
            try:
                df = pd.read_csv(f)
                if 'SYMBOL' in df.columns:
                    for sym in df['SYMBOL']:
                        sym = str(sym).strip()
                        if sym and " " not in sym and sym != "nan":
                            if not sym.endswith(".NS"):
                                sym += ".NS"
                            stocks.append(sym)
            except Exception as e:
                print(f"Error reading {f}: {e}")
    return stocks

def save_custom_stocks(stocks):
    with open(CUSTOM_STOCKS_FILE, "w") as f:
        json.dump(stocks, f)

def get_tech_stocks():
    custom = load_custom_stocks()
    folder_stocks = load_stocks_from_folder()
    return list(dict.fromkeys(DEFAULT_NIFTY50 + DEFAULT_NIFTY100 + custom + folder_stocks))

def get_fund_stocks():
    custom = load_custom_stocks()
    folder_stocks = load_stocks_from_folder()
    return list(dict.fromkeys(DEFAULT_NIFTY50 + DEFAULT_NIFTY100 + custom + folder_stocks))


# ── Background scan threads ───────────────────────────────────────────────────
def do_tech_scan(period, interval, rsi_buy_thresh=30, rsi_sell_thresh=70):
    scan_status["tech"] = "running"
    scan_params["tech"] = {"period": period, "interval": interval, "rsi_buy": rsi_buy_thresh, "rsi_sell": rsi_sell_thresh}
    scan_progress["tech"] = {"current": 0, "total": 0}
    try:
        stocks = get_tech_stocks()
        scan_progress["tech"]["total"] = len(stocks)

        def progress_cb(current, total):
            scan_progress["tech"] = {"current": current, "total": total}

        results = run_technical_scan(stocks, period=period, interval=interval,
                                     progress_cb=progress_cb, 
                                     rsi_buy_thresh=rsi_buy_thresh, rsi_sell_thresh=rsi_sell_thresh)
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
            'EMA Cross':       _str_or(r, 'EMA Cross',       '---'),
            'Bollinger':       _str_or(r, 'Bollinger',       '---'),
            'Supertrend':      _str_or(r, 'Supertrend',      '---'),
            'StochRSI':        _str_or(r, 'StochRSI',        '---'),
            'VWAP':            _float_or_none(r.get('VWAP', '')),
            '200 EMA Cross':   _str_or(r, '200 EMA Cross',   '---'),
            'Trend Score':     _float_or_none(r.get('Trend Score',    '')),
            'Trend Reasons':   _str_or(r, 'Trend Reasons',   ''),
            'Trend Warnings':  _str_or(r, 'Trend Warnings',  ''),
            'Beta':            _float_or_none(r.get('Beta', '')),
            '200 SMA':      _float_or_none(r.get('200 SMA', '')),
            '50 SMA':       _float_or_none(r.get('50 SMA', '')),
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
            'ROA':            _float_or_none(r.get('ROA', '')),
            'Div Yield':      _float_or_none(r.get('Div Yield', '')),
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


@app.route("/api/data/signals")
def data_signals():
    tech_rows = scan_results["tech"] if scan_results["tech"] is not None else load_tech_csv()
    fund_rows = scan_results["fund"] if scan_results["fund"] is not None else load_fund_csv()

    # Build fund lookup keyed by symbol (with and without .NS)
    fund_map = {}
    for row in (fund_rows or []):
        sym = row.get('Symbol', '')
        if sym:
            fund_map[sym] = row
            fund_map[sym.replace('.NS', '')] = row

    merged = []
    for tr in (tech_rows or []):
        stock = tr.get('Stock', '')
        fr = fund_map.get(stock) or fund_map.get(stock.replace('.NS', '')) or {}
        merged.append({
            'Stock':          tr.get('Stock'),
            'Price':          tr.get('Price'),
            '% Change':       tr.get('% Change'),
            'Consensus':      tr.get('Consensus'),
            'Strength':       tr.get('Strength'),
            'Trend':          tr.get('Trend'),
            'RSI':            tr.get('RSI'),
            'MACD Signal':    tr.get('MACD Signal'),
            'UT Bot':         tr.get('UT Bot'),
            'EMA Cross':       tr.get('EMA Cross'),
            'Bollinger':       tr.get('Bollinger'),
            'Supertrend':      tr.get('Supertrend'),
            'StochRSI':        tr.get('StochRSI'),
            'VWAP':            tr.get('VWAP'),
            '200 EMA Cross':   tr.get('200 EMA Cross'),
            'Trend Score':     tr.get('Trend Score'),
            'Trend Reasons':   tr.get('Trend Reasons'),
            'Trend Warnings':  tr.get('Trend Warnings'),
            'Vol Ratio':      tr.get('Vol Ratio'),
            'Vol Spike':      tr.get('Vol Spike'),
            'Beta':           tr.get('Beta'),
            '200 SMA':        tr.get('200 SMA'),
            '50 SMA':         tr.get('50 SMA'),
            'Last Scanned':   tr.get('Last Scanned'),
            'Chart':          tr.get('Chart'),
            'Fund Score':     fr.get('Fund Score'),
            'Final Score':    fr.get('Final Score'),
            'Recommendation': fr.get('Recommendation'),
            'Sector':         fr.get('Sector'),
            'P/E':            fr.get('P/E'),
            'D/E':            fr.get('D/E'),
            'ROE':            fr.get('ROE'),
            'ROA':            fr.get('ROA'),
            'Div Yield':      fr.get('Div Yield'),
        })

    tech_mtime = os.path.getmtime(TECH_CSV) if os.path.exists(TECH_CSV) else None
    fund_mtime = os.path.getmtime(FUND_CSV) if os.path.exists(FUND_CSV) else None
    mtimes = [t for t in [tech_mtime, fund_mtime] if t is not None]
    last_mtime = max(mtimes) if mtimes else None
    generated_at = datetime.fromtimestamp(last_mtime).strftime("%d %b %Y, %H:%M") if last_mtime else None
    tech_at = datetime.fromtimestamp(tech_mtime).strftime("%d %b, %H:%M") if tech_mtime else None
    fund_at = datetime.fromtimestamp(fund_mtime).strftime("%d %b, %H:%M") if fund_mtime else None

    return safe_jsonify({
        "rows":        merged,
        "generated_at": generated_at,
        "tech_at":     tech_at,
        "fund_at":     fund_at,
        "tech_params": scan_params["tech"],
        "fund_params": scan_params["fund"],
    })


@app.route("/api/download/combined")
def download_combined():
    import csv as csv_mod
    from io import StringIO
    tech_rows = scan_results["tech"] if scan_results["tech"] is not None else load_tech_csv()
    fund_rows = scan_results["fund"] if scan_results["fund"] is not None else load_fund_csv()
    fund_map = {}
    for row in (fund_rows or []):
        sym = row.get('Symbol', '')
        if sym:
            fund_map[sym] = row
            fund_map[sym.replace('.NS', '')] = row
    merged = []
    for tr in (tech_rows or []):
        stock = tr.get('Stock', '')
        fr = fund_map.get(stock) or fund_map.get(stock.replace('.NS', '')) or {}
        merged.append({
            'Stock':          stock,
            'Price':          tr.get('Price', ''),
            '% Change':       tr.get('% Change', ''),
            'Consensus':      tr.get('Consensus', ''),
            'Strength':       tr.get('Strength', ''),
            'Trend':          tr.get('Trend', ''),
            'RSI':            tr.get('RSI', ''),
            'MACD Signal':    tr.get('MACD Signal', ''),
            'UT Bot':         tr.get('UT Bot', ''),
            'EMA Cross':      tr.get('EMA Cross', ''),
            'Bollinger':      tr.get('Bollinger', ''),
            'Supertrend':     tr.get('Supertrend', ''),
            'StochRSI':       tr.get('StochRSI', ''),
            'VWAP':           tr.get('VWAP', ''),
            '200 EMA Cross':  tr.get('200 EMA Cross', ''),
            'Vol Ratio':      tr.get('Vol Ratio', ''),
            'Vol Spike':      tr.get('Vol Spike', ''),
            '50 SMA':         tr.get('50 SMA', ''),
            '200 SMA':        tr.get('200 SMA', ''),
            'Beta':           tr.get('Beta', ''),
            'Fund Score':     fr.get('Fund Score', ''),
            'Final Score':    fr.get('Final Score', ''),
            'Recommendation': fr.get('Recommendation', ''),
            'Sector':         fr.get('Sector', ''),
            'P/E':            fr.get('P/E', ''),
            'D/E':            fr.get('D/E', ''),
            'ROE':            fr.get('ROE', ''),
            'ROA':            fr.get('ROA', ''),
            'Div Yield':      fr.get('Div Yield', ''),
        })
    if not merged:
        return jsonify({"error": "No data available. Run a scan first."}), 404
    si = StringIO()
    writer = csv_mod.DictWriter(si, fieldnames=list(merged[0].keys()))
    writer.writeheader()
    writer.writerows(merged)
    from flask import make_response
    out = make_response(si.getvalue())
    out.headers["Content-Disposition"] = "attachment; filename=nifty_combined_scan.csv"
    out.headers["Content-type"] = "text/csv"
    return out


@app.route("/api/run/report_sync", methods=["POST"])
def run_report_sync():
    data = request.json or {}
    t_period = data.get("tech_period", "1y")
    t_interval = data.get("tech_interval", "1wk")
    f_period = data.get("fund_period", "1y")
    f_interval = data.get("fund_interval", "1d")

    stocks = get_tech_stocks()
    
    # Run synchronously
    r_tech = run_technical_scan(stocks, period=t_period, interval=t_interval, progress_cb=lambda c,t: None)
    if r_tech:
        scan_results["tech"] = r_tech
        save_tech_csv(r_tech)

    r_fund = run_fundamental_scan(stocks, period=f_period, interval=f_interval, progress_cb=lambda c,t: None)
    if r_fund:
        scan_results["fund"] = r_fund
        save_fund_csv(r_fund)

    return jsonify({"status": "success"})


@app.route("/api/custom_stocks", methods=["GET", "POST"])
def api_custom_stocks():
    if request.method == "GET":
        return jsonify(load_custom_stocks())
    else:
        stocks = request.json.get("stocks", [])
        save_custom_stocks(stocks)
        return jsonify({"status": "success"})


@app.route("/api/run/tech", methods=["POST"])
def api_run_tech():
    data = request.json or {}
    period = data.get("period", "1y")
    interval = data.get("interval", "1wk")
    rsi_buy = data.get("rsi_buy", 30)
    rsi_sell = data.get("rsi_sell", 70)
    if scan_status["tech"] == "running":
        return jsonify({"status": "already_running"})
    t = threading.Thread(target=do_tech_scan, args=(period, interval, rsi_buy, rsi_sell))
    t.daemon = True
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/run/<scanner>", methods=["POST"])
def run_scanner(scanner):
    body     = request.get_json(silent=True) or {}
    period   = body.get("period", "1y")
    interval = body.get("interval", "1wk" if scanner == "tech" else "1d")
    rsi_buy  = int(body.get("rsi_buy", 30))
    rsi_sell = int(body.get("rsi_sell", 70))

    if scanner == "tech" and scan_status["tech"] != "running":
        t = threading.Thread(target=do_tech_scan, args=(period, interval, rsi_buy, rsi_sell), daemon=True)
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
    unified = list(dict.fromkeys(DEFAULT_NIFTY50 + DEFAULT_NIFTY100))
    return jsonify({"default_list": unified, "custom": custom})


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


# ── Paper Trading Routes ──────────────────────────────────────────────────────
@app.route("/api/paper/portfolio")
def paper_portfolio():
    with_prices = request.args.get('prices', 'true').lower() == 'true'
    portfolio = pt.get_portfolio(with_prices=with_prices)
    return safe_jsonify(portfolio)


@app.route("/api/paper/trade", methods=["POST"])
def paper_add_trade():
    body   = request.get_json(silent=True) or {}
    symbol = normalize_symbol(body.get('symbol', ''))
    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400
    action = body.get('action', 'Buy')
    if action not in ('Buy', 'Sell'):
        return jsonify({'error': 'Action must be Buy or Sell'}), 400
    try:
        quantity    = int(body.get('quantity', 1))
        entry_price = float(body.get('entry_price', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid quantity or price'}), 400
    if quantity <= 0:
        return jsonify({'error': 'Quantity must be > 0'}), 400
    if entry_price <= 0:
        return jsonify({'error': 'Entry price must be > 0'}), 400
    notes = str(body.get('notes', ''))
    trade = pt.add_trade(symbol, action, quantity, entry_price, notes)
    return jsonify({'success': True, 'trade': trade})


@app.route("/api/paper/trade/<trade_id>/close", methods=["PUT"])
def paper_close_trade(trade_id):
    body = request.get_json(silent=True) or {}
    try:
        exit_price = float(body.get('exit_price', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid exit price'}), 400
    if exit_price <= 0:
        return jsonify({'error': 'Exit price must be > 0'}), 400
    trade = pt.close_trade(trade_id, exit_price)
    if trade is None:
        return jsonify({'error': 'Trade not found or already closed'}), 404
    return jsonify({'success': True, 'trade': trade})


@app.route("/api/paper/trade/<trade_id>", methods=["DELETE"])
def paper_delete_trade(trade_id):
    ok = pt.delete_trade(trade_id)
    if not ok:
        return jsonify({'error': 'Trade not found'}), 404
    return jsonify({'success': True})


@app.route("/api/paper/capital", methods=["PUT"])
def paper_set_capital():
    body = request.get_json(silent=True) or {}
    try:
        capital = float(body.get('capital', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid capital amount'}), 400
    if capital <= 0:
        return jsonify({'error': 'Capital must be > 0'}), 400
    pt.set_capital(capital)
    return jsonify({'success': True, 'capital': capital})


@app.route("/api/paper/download")
def paper_download():
    import csv as csv_mod
    from io import StringIO
    portfolio = pt.get_portfolio(with_prices=True)
    si = StringIO()
    cw = csv_mod.writer(si)
    cw.writerow(["ID", "Symbol", "Action", "Quantity", "Entry Price", "Entry Date", "Status", "Exit Price", "Exit Date", "Realized PNL", "Unrealized PNL", "Notes"])
    for t in portfolio.get("open_trades", []) + portfolio.get("closed_trades", []):
        cw.writerow([
            t.get("id"), t.get("symbol"), t.get("action"), t.get("quantity"), t.get("entry_price"), 
            t.get("entry_date"), t.get("status"), t.get("exit_price", ""), t.get("exit_date", ""),
            t.get("realized_pnl", ""), t.get("unrealized_pnl", ""), t.get("notes", "")
        ])
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=paper_trades.csv"}
    )



@app.route("/api/paper/price/<symbol>")
def paper_get_price(symbol):
    sym    = normalize_symbol(symbol)
    prices = pt.fetch_prices({sym})
    price  = prices.get(sym)
    if price is None:
        return jsonify({'error': 'Could not fetch price'}), 404
    return jsonify({'symbol': sym, 'price': price})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
