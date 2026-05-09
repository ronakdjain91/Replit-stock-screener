import math
import pandas as pd
import numpy as np
import yfinance as yf


DEFAULT_NIFTY100 = [
    "HDFCBANK.NS", "RELIANCE.NS", "INFY.NS", "TCS.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "ITC.NS", "SBIN.NS", "HCLTECH.NS",
    "ASIANPAINT.NS", "AXISBANK.NS", "LT.NS", "BHARTIARTL.NS",
    "MARUTI.NS", "NESTLEIND.NS", "POWERGRID.NS", "SUNPHARMA.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "WIPRO.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GAIL.NS",
    "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "IOC.NS", "INDUSINDBK.NS",
    "JSWSTEEL.NS", "NTPC.NS", "ONGC.NS", "SHREECEM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TECHM.NS", "UPL.NS", "TATAELXSI.NS", "WOCKPHARMA.NS",
    "ZEEL.NS", "ADANIGREEN.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BAJAJHLDNG.NS", "BANDHANBNK.NS", "BERGEPAINT.NS", "COLPAL.NS", "DABUR.NS",
    "DLF.NS", "GODREJCP.NS", "HDFCLIFE.NS", "HINDPETRO.NS", "ICICIPRULI.NS",
    "IDEA.NS", "IGL.NS", "INDIGO.NS", "LUPIN.NS", "MANAPPURAM.NS", "MARICO.NS",
    "NHPC.NS", "NMDC.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PNB.NS",
    "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", "SAIL.NS", "SBILIFE.NS",
    "SIEMENS.NS", "TATACHEM.NS", "TATACONSUM.NS", "UBL.NS", "ICICIGI.NS",
    "GLENMARK.NS", "SUNTV.NS", "PNBHOUSING.NS", "ABCAPITAL.NS", "INDIAMART.NS",
    "CUB.NS", "DEEPAKNTR.NS", "ABBOTINDIA.NS", "APOLLOTYRE.NS", "M&M.NS",
    "ADANIENT.NS", "VEDL.NS"
]

DEFAULT_NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS",
    "ITC.NS", "HCLTECH.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "NESTLEIND.NS", "TITAN.NS", "ULTRACEMCO.NS", "POWERGRID.NS",
    "ONGC.NS", "NTPC.NS", "INDUSINDBK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
    "BRITANNIA.NS", "DIVISLAB.NS", "EICHERMOT.NS", "GRASIM.NS", "HDFCLIFE.NS",
    "IOC.NS", "JSWSTEEL.NS", "WIPRO.NS", "TATASTEEL.NS", "COALINDIA.NS",
    "SBILIFE.NS", "BPCL.NS", "ADANIENT.NS", "TECHM.NS", "M&M.NS",
    "CIPLA.NS", "HEROMOTOCO.NS", "INDIGO.NS", "SHREECEM.NS", "TATAMOTORS.NS",
    "UPL.NS", "HINDALCO.NS"
]

VALID_PERIODS = {"3mo", "6mo", "1y", "2y", "5y"}
VALID_INTERVALS_TECH = {"1d", "1wk", "1mo"}
VALID_INTERVALS_FUND = {"1d", "1wk"}


# ── Utility ──────────────────────────────────────────────────────────────────

def safe_float(v, ndigits=None):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits) if ndigits is not None else f
    except (TypeError, ValueError):
        return None


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def generate_tv_link(symbol):
    return f"https://in.tradingview.com/symbols/NSE:{symbol.replace('.NS', '/')}/"


# ── Native indicator implementations ─────────────────────────────────────────

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df, period=10):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _sma(series, w):
    return series.rolling(window=w, min_periods=1).mean()


# ── UT Bot (no pandas-ta) ─────────────────────────────────────────────────────

def calculate_ut_bot(df, a=1, c=10):
    try:
        atr = _atr(df, period=c)
        nLoss = a * atr
        trailing_stop = pd.Series(0.0, index=df.index)
        close = df['Close']
        for i in range(1, len(df)):
            prev_stop = trailing_stop.iloc[i - 1]
            src = close.iloc[i]
            src_prev = close.iloc[i - 1]
            nl = nLoss.iloc[i]
            if pd.isna(src) or pd.isna(src_prev) or pd.isna(nl):
                trailing_stop.iloc[i] = prev_stop
                continue
            if src > prev_stop and src_prev > prev_stop:
                trailing_stop.iloc[i] = max(prev_stop, src - nl)
            elif src < prev_stop and src_prev < prev_stop:
                trailing_stop.iloc[i] = min(prev_stop, src + nl)
            else:
                trailing_stop.iloc[i] = src - nl if src > prev_stop else src + nl
        buy = (close > trailing_stop) & (close.shift(1) <= trailing_stop.shift(1))
        sell = (close < trailing_stop) & (close.shift(1) >= trailing_stop.shift(1))
        return buy, sell
    except Exception:
        false_series = pd.Series(False, index=df.index)
        return false_series, false_series


# ── Signal helpers ────────────────────────────────────────────────────────────

def signal_macd(hist):
    h = hist.dropna()
    if len(h) < 2:
        return '---'
    if h.iloc[-1] > 0 and h.iloc[-2] < 0:
        return 'Buy'
    if h.iloc[-2] > 0 and h.iloc[-1] < 0:
        return 'Sell'
    return '---'


def signal_rsi(rsi_series):
    r = rsi_series.dropna()
    if len(r) < 2:
        return '---'
    cur, prev = r.iloc[-1], r.iloc[-2]
    if cur <= 35 and prev > 35:
        return 'Buy'
    if cur >= 55 and prev < 55:
        return 'Sell'
    return '---'


# ── Technical scan ────────────────────────────────────────────────────────────

def run_technical_scan(stocks, period='1y', interval='1wk'):
    if period not in VALID_PERIODS:
        period = '1y'
    if interval not in VALID_INTERVALS_TECH:
        interval = '1wk'

    results = []
    for stock in stocks:
        try:
            df = yf.download(stock, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            df = flatten_columns(df)
            if 'Close' not in df.columns:
                continue
            df['Close'] = df['Close'].ffill()
            close = df['Close'].dropna()
            if len(close) < 3:
                continue

            # Need High/Low for ATR — ensure they exist
            for col in ('High', 'Low'):
                if col not in df.columns:
                    df[col] = df['Close']
                else:
                    df[col] = df[col].ffill()

            rsi_series = _rsi(close)
            _, _, hist = _macd(close)
            buy_sig, sell_sig = calculate_ut_bot(df)

            macd_sig = signal_macd(hist)
            rsi_sig = signal_rsi(rsi_series)
            ut_sig = 'Buy' if buy_sig.iloc[-1] else ('Sell' if sell_sig.iloc[-1] else '---')

            valid_closes = close
            if len(valid_closes) < 2:
                continue
            price = safe_float(valid_closes.iloc[-1], 2)
            prev_price = safe_float(valid_closes.iloc[-2])
            if price is None or prev_price is None or prev_price == 0:
                continue

            pct_chg = safe_float(((price - prev_price) / prev_price) * 100, 2)
            rsi_val = safe_float(rsi_series.dropna().iloc[-1], 1) if not rsi_series.dropna().empty else None
            macd_hist_val = safe_float(hist.dropna().iloc[-1], 3) if not hist.dropna().empty else None

            results.append({
                'Stock':       stock,
                'Price':       price,
                '% Change':    pct_chg,
                'RSI':         rsi_val,
                'MACD Hist':   macd_hist_val,
                'MACD Signal': macd_sig,
                'RSI Signal':  rsi_sig,
                'UT Bot':      ut_sig,
                'Chart':       generate_tv_link(stock),
            })
        except Exception as e:
            print(f"[tech] Error {stock}: {e}")
    return results


# ── Fundamental scan ──────────────────────────────────────────────────────────

def _fund_score(ticker_obj):
    info = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        pass

    score = 0
    pe = safe_float(info.get('trailingPE') or info.get('forwardPE'))
    if pe and pe > 0:
        score += 2 if pe < 15 else (1 if pe < 25 else 0)

    dte = safe_float(info.get('debtToEquity'))
    if dte is not None:
        score += 1 if dte < 50 else (0.5 if dte < 100 else 0)

    roe = safe_float(info.get('returnOnEquity') or info.get('returnOnAssets'))
    if roe is not None:
        score += 1 if roe >= 0.15 else (0.5 if roe > 0.08 else 0)

    try:
        fin = ticker_obj.quarterly_financials
        if isinstance(fin, pd.DataFrame) and not fin.empty:
            rev_row = None
            for cand in ['Total Revenue', 'TotalRevenue', 'Revenue']:
                if cand in fin.index:
                    rev_row = fin.loc[cand]
                    break
            if rev_row is None:
                rev_row = fin.iloc[0]
            rv = rev_row.dropna().astype(float)
            if len(rv) >= 3:
                growth = (rv.iloc[0] - rv.iloc[-1]) / (abs(rv.iloc[-1]) + 1e-6)
                if growth > 0.03:
                    score += 1
    except Exception:
        pass

    return round(score, 2), {"pe": pe, "de": dte, "roe": roe}


def run_fundamental_scan(stocks, period='1y', interval='1d'):
    if period not in VALID_PERIODS:
        period = '1y'
    if interval not in VALID_INTERVALS_FUND:
        interval = '1d'

    results = []
    for sym in stocks:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period=period, interval=interval, actions=False)
            if hist is None or hist.empty or len(hist) < 10:
                continue
            hist = flatten_columns(hist)
            close_col = 'Close' if 'Close' in hist.columns else hist.columns[0]
            close = hist[close_col].ffill().dropna()
            if len(close) < 10:
                continue

            s50 = _sma(close, 50)
            s200 = _sma(close, 200)
            rsi_val = safe_float(_rsi(close).iloc[-1], 1)
            _, _, mhist = _macd(close)
            macd_h = safe_float(mhist.iloc[-1], 3)
            price = safe_float(close.iloc[-1], 2)
            if price is None:
                continue

            tscore = 0
            s200_last = safe_float(s200.iloc[-1])
            s50_last = safe_float(s50.iloc[-1])
            if s200_last and price > s200_last:
                tscore += 2
            elif s50_last and price > s50_last:
                tscore += 1
            if s50_last and s200_last and s50_last > s200_last:
                tscore += 1
            if rsi_val is not None:
                tscore += 1 if 30 < rsi_val < 70 else (0.5 if rsi_val < 30 else 0)
            if macd_h is not None and macd_h > 0:
                tscore += 0.5

            fscore, fmeta = _fund_score(tk)
            final = round(0.6 * (fscore / 5) + 0.4 * (tscore / 5), 3)
            rec = 'Sell'
            if fscore >= 2.5:
                rec = 'Buy' if final >= 0.7 else ('Hold' if final >= 0.45 else 'Sell')

            roe_display = safe_float(fmeta['roe'] * 100, 1) if fmeta['roe'] is not None else None

            results.append({
                'Symbol':         sym,
                'Price':          price,
                'P/E':            safe_float(fmeta['pe'], 1),
                'D/E':            safe_float(fmeta['de'], 1),
                'ROE':            roe_display,
                'RSI':            rsi_val,
                'Fund Score':     fscore,
                'Tech Score':     round(tscore, 2),
                'Final Score':    final,
                'Recommendation': rec,
                'Chart':          generate_tv_link(sym)
            })
        except Exception as e:
            print(f"[fund] Error {sym}: {e}")
    return results
