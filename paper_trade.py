import json
import math
import os
import uuid
import pandas as pd
import yfinance as yf
from datetime import datetime

_DATA_DIR  = os.environ.get("DATA_DIR", ".")
PAPER_FILE = os.path.join(_DATA_DIR, "paper_trades.json")
DEFAULT_CAPITAL = 100000.0


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_float(v, ndigits=None):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits) if ndigits is not None else f
    except (TypeError, ValueError):
        return None


# ── Persistence ───────────────────────────────────────────────────────────────

def load_data():
    if os.path.exists(PAPER_FILE):
        try:
            with open(PAPER_FILE) as f:
                d = json.load(f)
                d.setdefault('capital', DEFAULT_CAPITAL)
                d.setdefault('trades',  [])
                return d
        except Exception:
            pass
    return {'capital': DEFAULT_CAPITAL, 'trades': []}


def save_data(data):
    with open(PAPER_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ── Trade operations ──────────────────────────────────────────────────────────

def add_trade(symbol, action, quantity, entry_price, notes=''):
    data  = load_data()
    trade = {
        'id':          str(uuid.uuid4())[:8],
        'symbol':      symbol,
        'action':      action,
        'quantity':    int(quantity),
        'entry_price': float(entry_price),
        'entry_date':  datetime.now().strftime("%d %b %Y, %H:%M"),
        'status':      'open',
        'exit_price':  None,
        'exit_date':   None,
        'realized_pnl': None,
        'notes':       notes,
    }
    data['trades'].append(trade)
    save_data(data)
    return trade


def close_trade(trade_id, exit_price):
    data = load_data()
    for trade in data['trades']:
        if trade['id'] == trade_id and trade['status'] == 'open':
            ep = float(exit_price)
            trade['exit_price'] = ep
            trade['exit_date']  = datetime.now().strftime("%d %b %Y, %H:%M")
            trade['status']     = 'closed'
            qty = trade['quantity']
            enp = trade['entry_price']
            if trade['action'] == 'Buy':
                trade['realized_pnl'] = round((ep - enp) * qty, 2)
            else:
                trade['realized_pnl'] = round((enp - ep) * qty, 2)
            save_data(data)
            return trade
    return None


def delete_trade(trade_id):
    data   = load_data()
    before = len(data['trades'])
    data['trades'] = [t for t in data['trades'] if t['id'] != trade_id]
    if len(data['trades']) < before:
        save_data(data)
        return True
    return False


def set_capital(amount):
    data = load_data()
    data['capital'] = float(amount)
    save_data(data)


# ── Price fetching ────────────────────────────────────────────────────────────

def fetch_prices(symbols):
    """Return {symbol: latest_price} for each symbol."""
    prices = {}
    if not symbols:
        return prices
    sym_list = list(symbols)
    try:
        df = yf.download(sym_list, period='5d', interval='1d',
                         progress=False, auto_adjust=True)
        if df.empty:
            return prices
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close']
            for sym in sym_list:
                if sym in close.columns:
                    last = close[sym].dropna()
                    if not last.empty:
                        prices[sym] = _safe_float(last.iloc[-1], 2)
        else:
            sym = sym_list[0]
            col = 'Close' if 'Close' in df.columns else df.columns[0]
            last = df[col].dropna()
            if not last.empty:
                prices[sym] = _safe_float(last.iloc[-1], 2)
    except Exception as e:
        print(f"[paper] Price fetch error: {e}")
    return prices


# ── Portfolio assembly ────────────────────────────────────────────────────────

def get_portfolio(with_prices=True):
    data   = load_data()
    trades = data['trades']
    capital = data['capital']

    open_trades   = [t for t in trades if t['status'] == 'open']
    closed_trades = [t for t in trades if t['status'] == 'closed']

    prices = {}
    if with_prices and open_trades:
        syms   = {t['symbol'] for t in open_trades}
        prices = fetch_prices(syms)

    total_invested   = 0.0
    total_current    = 0.0
    total_unrealized = 0.0

    enriched_open = []
    for t in open_trades:
        qty      = t['quantity']
        enp      = t['entry_price']
        invested = enp * qty
        cur_p    = prices.get(t['symbol'])

        if cur_p is not None:
            cur_val = cur_p * qty
            pnl     = round(((cur_p - enp) * qty) if t['action'] == 'Buy'
                            else ((enp - cur_p) * qty), 2)
            pnl_pct = round((pnl / invested) * 100, 2) if invested else 0.0
        else:
            cur_val = invested
            pnl     = None
            pnl_pct = None

        total_invested   += invested
        total_current    += (cur_p * qty if cur_p is not None else invested)
        total_unrealized += (pnl if pnl is not None else 0.0)

        enriched_open.append({
            **t,
            'invested':           round(invested, 2),
            'current_price':      cur_p,
            'current_value':      round(cur_val, 2),
            'unrealized_pnl':     pnl,
            'unrealized_pnl_pct': pnl_pct,
        })

    realized_pnl = sum(
        t['realized_pnl'] for t in closed_trades
        if t['realized_pnl'] is not None
    )

    available = max(capital - total_invested, 0.0)

    return {
        'capital':      capital,
        'open_trades':  enriched_open,
        'closed_trades': list(reversed(closed_trades)),
        'summary': {
            'total_invested':   round(total_invested,   2),
            'current_value':    round(total_current,    2),
            'unrealized_pnl':   round(total_unrealized, 2),
            'realized_pnl':     round(realized_pnl,     2),
            'total_pnl':        round(total_unrealized + realized_pnl, 2),
            'available':        round(available,        2),
            'open_count':       len(open_trades),
            'closed_count':     len(closed_trades),
        },
    }
