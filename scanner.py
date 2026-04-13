import math
import pandas as pd
import numpy as np
import pandas_ta as ta
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


def safe_float(v, ndigits=None):
    """Convert to float; return None if NaN, Inf or unconvertible."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits) if ndigits is not None else f
    except (TypeError, ValueError):
        return None


def flatten_columns(df):
    """Flatten MultiIndex columns that yfinance sometimes returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]
    return df


def generate_tv_link(symbol):
    sym = symbol.replace(".NS", "")
    return f"https://in.tradingview.com/symbols/NSE:{sym}/"


def calculate_ut_bot(df, a=1, c=10):
    atr = df.ta.atr(length=c)
    if atr is None or atr.isna().all():
        return pd.Series(0.0, index=df.index), pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    nLoss = a * atr
    trailing_stop = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        prev_stop = trailing_stop.iloc[i - 1]
        src = df['Close'].iloc[i]
        src_prev = df['Close'].iloc[i - 1]
        if pd.isna(src) or pd.isna(src_prev) or pd.isna(nLoss.iloc[i]):
            trailing_stop.iloc[i] = prev_stop
            continue
        if src > prev_stop and src_prev > prev_stop:
            trailing_stop.iloc[i] = max(prev_stop, src - nLoss.iloc[i])
        elif src < prev_stop and src_prev < prev_stop:
            trailing_stop.iloc[i] = min(prev_stop, src + nLoss.iloc[i])
        else:
            trailing_stop.iloc[i] = src - nLoss.iloc[i] if src > prev_stop else src + nLoss.iloc[i]
    buy = (df['Close'] > trailing_stop) & (df['Close'].shift(1) <= trailing_stop.shift(1))
    sell = (df['Close'] < trailing_stop) & (df['Close'].shift(1) >= trailing_stop.shift(1))
    return trailing_stop, buy, sell


def signal_macd(df):
    col = 'MACDh_12_26_9'
    if col not in df.columns or df[col].dropna().shape[0] < 2:
        return '---'
    last = df[col].dropna()
    if last.iloc[-1] > 0 and last.iloc[-2] < 0:
        return 'Buy'
    if last.iloc[-2] > 0 and last.iloc[-1] < 0:
        return 'Sell'
    return '---'


def signal_rsi(df):
    col = 'RSI_14'
    if col not in df.columns or df[col].dropna().shape[0] < 2:
        return '---'
    vals = df[col].dropna()
    cur, prev = vals.iloc[-1], vals.iloc[-2]
    if cur <= 35 and prev > 35:
        return 'Buy'
    if cur >= 55 and prev < 55:
        return 'Sell'
    return '---'


def get_close_series(df):
    """Prefer 'Close' over 'Adj Close'; drop leading NaN rows."""
    for col in ('Close', 'Adj Close'):
        if col in df.columns:
            s = df[col].dropna()
            if not s.empty:
                return s
    return None


def run_technical_scan(stocks, period='1y', interval='1wk'):
    if period not in VALID_PERIODS:
        period = '1y'
    if interval not in VALID_INTERVALS_TECH:
        interval = '1wk'

    results = []
    for stock in stocks:
        try:
            df = yf.download(
                stock, period=period, progress=False,
                interval=interval, auto_adjust=True
            )
            if df is None or df.empty:
                continue

            df = flatten_columns(df)

            # Ensure we have a usable Close column
            close = get_close_series(df)
            if close is None or len(close) < 3:
                continue

            # Rebuild df with clean Close so indicators work
            df['Close'] = df['Close'].ffill()

            macd_df = df.ta.macd()
            rsi_series = df.ta.rsi(length=14)
            trailing_stop, buy, sell = calculate_ut_bot(df)

            if macd_df is not None:
                df = pd.concat([df, macd_df], axis=1)
            if rsi_series is not None:
                df = pd.concat([df, rsi_series.rename('RSI_14')], axis=1)

            macd_sig = signal_macd(df)
            rsi_sig = signal_rsi(df)
            ut_sig = 'Buy' if buy.iloc[-1] else ('Sell' if sell.iloc[-1] else '---')

            # Use the last two non-NaN close prices for price & % change
            valid_closes = df['Close'].dropna()
            if len(valid_closes) < 2:
                continue

            price = safe_float(valid_closes.iloc[-1], 2)
            prev_price = safe_float(valid_closes.iloc[-2])
            if price is None or prev_price is None or prev_price == 0:
                continue

            pct_chg = safe_float(((price - prev_price) / prev_price) * 100, 2)

            rsi_col = 'RSI_14'
            rsi_val = safe_float(df[rsi_col].dropna().iloc[-1], 1) if rsi_col in df.columns and not df[rsi_col].dropna().empty else None

            macd_col = 'MACDh_12_26_9'
            macd_hist = safe_float(df[macd_col].dropna().iloc[-1], 3) if macd_col in df.columns and not df[macd_col].dropna().empty else None

            results.append({
                'Stock': stock,
                'Price': price,
                '% Change': pct_chg,
                'RSI': rsi_val,
                'MACD Hist': macd_hist,
                'MACD Signal': macd_sig,
                'RSI Signal': rsi_sig,
                'UT Bot': ut_sig,
                'Chart': generate_tv_link(stock),
            })
        except Exception as e:
            print(f"[tech] Error {stock}: {e}")
    return results


def _sma(series, w):
    return series.rolling(window=w, min_periods=1).mean()


def _rsi(series, w=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / w, adjust=False).mean()
    al = loss.ewm(alpha=1 / w, adjust=False).mean()
    rs = ag / (al.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50)


def _macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl


def _fund_score(ticker_obj):
    info = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        pass

    score = 0
    pe = info.get('trailingPE') or info.get('forwardPE')
    pe = safe_float(pe)
    if pe and pe > 0:
        score += 2 if pe < 15 else (1 if pe < 25 else 0)

    dte = safe_float(info.get('debtToEquity'))
    if dte is not None:
        score += 1 if dte < 50 else (0.5 if dte < 100 else 0)

    roe_raw = info.get('returnOnEquity') or info.get('returnOnAssets')
    roe = safe_float(roe_raw)
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
                if 30 < rsi_val < 70:
                    tscore += 1
                elif rsi_val < 30:
                    tscore += 0.5
            if macd_h is not None and macd_h > 0:
                tscore += 0.5

            fscore, fmeta = _fund_score(tk)
            fund_pct = fscore / 5
            tech_pct = tscore / 5
            final = round(0.6 * fund_pct + 0.4 * tech_pct, 3)

            rec = 'Sell'
            if fscore >= 2.5:
                rec = 'Buy' if final >= 0.7 else ('Hold' if final >= 0.45 else 'Sell')

            roe_display = None
            if fmeta['roe'] is not None:
                roe_display = safe_float(fmeta['roe'] * 100, 1)

            results.append({
                'Symbol': sym,
                'Price': price,
                'P/E': safe_float(fmeta['pe'], 1),
                'D/E': safe_float(fmeta['de'], 1),
                'ROE': roe_display,
                'RSI': rsi_val,
                'Fund Score': fscore,
                'Tech Score': round(tscore, 2),
                'Final Score': final,
                'Recommendation': rec,
                'Chart': generate_tv_link(sym)
            })
        except Exception as e:
            print(f"[fund] Error {sym}: {e}")
    return results
