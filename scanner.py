import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta


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


def generate_tv_link(symbol):
    sym = symbol.replace(".NS", "")
    return f"https://in.tradingview.com/symbols/NSE:{sym}/"


def calculate_ut_bot(df, a=1, c=10):
    atr = df.ta.atr(length=c)
    nLoss = a * atr
    trailing_stop = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        prev_stop = trailing_stop.iloc[i - 1]
        src = df['Close'].iloc[i]
        src_prev = df['Close'].iloc[i - 1]
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
    if 'MACDh_12_26_9' not in df.columns:
        return '---'
    if df['MACDh_12_26_9'].iloc[-1] > 0 and df['MACDh_12_26_9'].iloc[-2] < 0:
        return 'Buy'
    elif df['MACDh_12_26_9'].iloc[-2] > 0 and df['MACDh_12_26_9'].iloc[-1] < 0:
        return 'Sell'
    return '---'


def signal_rsi(df):
    if 'RSI_14' not in df.columns:
        return '---'
    cur, prev = df['RSI_14'].iloc[-1], df['RSI_14'].iloc[-2]
    if cur <= 35 and prev > 35:
        return 'Buy'
    elif cur >= 55 and prev < 55:
        return 'Sell'
    return '---'


def run_technical_scan(stocks, period='1y', interval='1wk'):
    if period not in VALID_PERIODS:
        period = '1y'
    if interval not in VALID_INTERVALS_TECH:
        interval = '1wk'

    results = []
    for stock in stocks:
        try:
            df = yf.download(stock, period=period, progress=False, interval=interval, auto_adjust=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns.get_level_values(0)]
            if 'Close' not in df.columns:
                continue
            if len(df) < 3:
                continue

            macd = df.ta.macd()
            rsi = df.ta.rsi(length=14)
            trailing_stop, buy, sell = calculate_ut_bot(df)
            df = pd.concat([df, macd, rsi, trailing_stop.rename('TrailingStop')], axis=1)

            macd_sig = signal_macd(df)
            rsi_sig = signal_rsi(df)
            ut_sig = 'Buy' if buy.iloc[-1] else ('Sell' if sell.iloc[-1] else '---')

            price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            pct_chg = ((price - prev_price) / prev_price) * 100

            rsi_val = float(df['RSI_14'].iloc[-1]) if 'RSI_14' in df.columns else None
            macd_hist = float(df['MACDh_12_26_9'].iloc[-1]) if 'MACDh_12_26_9' in df.columns else None

            results.append({
                'Stock': stock,
                'Price': round(price, 2),
                '% Change': round(pct_chg, 2),
                'RSI': round(rsi_val, 1) if rsi_val is not None else None,
                'MACD Hist': round(macd_hist, 3) if macd_hist is not None else None,
                'MACD Signal': macd_sig,
                'RSI Signal': rsi_sig,
                'UT Bot': ut_sig,
                'Chart': generate_tv_link(stock),
            })
        except Exception as e:
            print(f"Error {stock}: {e}")
    return results


def _sma(series, w):
    return series.rolling(window=w, min_periods=1).mean()


def _rsi(series, w=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    ag = gain.ewm(alpha=1/w, adjust=False).mean()
    al = loss.ewm(alpha=1/w, adjust=False).mean()
    rs = ag / (al.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50)


def _macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl


def _fund_score(ticker_obj):
    info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
    score = 0
    pe = info.get('trailingPE') or info.get('forwardPE')
    if pe and pe > 0:
        score += 2 if pe < 15 else (1 if pe < 25 else 0)
    dte = info.get('debtToEquity')
    if dte is not None:
        score += 1 if dte < 50 else (0.5 if dte < 100 else 0)
    roe = info.get('returnOnEquity') or info.get('returnOnAssets')
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
            close = hist['Close']
            s50 = _sma(close, 50)
            s200 = _sma(close, 200)
            rsi_val = float(_rsi(close).iloc[-1])
            _, _, mhist = _macd(close)
            macd_h = float(mhist.iloc[-1])
            price = float(close.iloc[-1])

            tscore = 0
            if price > s200.iloc[-1]: tscore += 2
            elif price > s50.iloc[-1]: tscore += 1
            if s50.iloc[-1] > s200.iloc[-1]: tscore += 1
            if 30 < rsi_val < 70: tscore += 1
            elif rsi_val < 30: tscore += 0.5
            if macd_h > 0: tscore += 0.5

            fscore, fmeta = _fund_score(tk)
            fund_pct = fscore / 5
            tech_pct = tscore / 5
            final = round(0.6 * fund_pct + 0.4 * tech_pct, 3)

            if fscore < 2.5:
                rec = 'Sell'
            else:
                rec = 'Buy' if final >= 0.7 else ('Hold' if final >= 0.45 else 'Sell')

            results.append({
                'Symbol': sym,
                'Price': round(price, 2),
                'P/E': round(fmeta['pe'], 1) if fmeta['pe'] else None,
                'D/E': round(fmeta['de'], 1) if fmeta['de'] else None,
                'ROE': round(fmeta['roe'] * 100, 1) if fmeta['roe'] else None,
                'RSI': round(rsi_val, 1),
                'Fund Score': fscore,
                'Tech Score': round(tscore, 2),
                'Final Score': final,
                'Recommendation': rec,
                'Chart': generate_tv_link(sym)
            })
        except Exception as e:
            print(f"Error {sym}: {e}")
    return results
