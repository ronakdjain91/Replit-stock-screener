# nifty_scanner.py
# Requirements: yfinance, pandas, numpy
# pip install yfinance pandas numpy

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------- CONFIG --------
# If you want ALL Nifty constituents, replace this list accordingly.
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","HDFC.NS","INFY.NS","ICICIBANK.NS","KOTAKBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","LT.NS","AXISBANK.NS","ITC.NS","HCLTECH.NS","BHARTIARTL.NS",
    "ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","NESTLEIND.NS","TITAN.NS","ULTRACEMCO.NS",
    "POWERGRID.NS","ONGC.NS","NTPC.NS","INDUSINDBK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS",
    "BRITANNIA.NS","DIVISLAB.NS","EICHERMOT.NS","GRASIM.NS","HDFCLIFE.NS","IOC.NS","JSWSTEEL.NS",
    "ONGC.NS","WIPRO.NS","TATASTEEL.NS","COALINDIA.NS","SBILIFE.NS","BPCL.NS","ADANIENT.NS",
    "TECHM.NS","M&M.NS","CIPLA.NS","HEROMOTOCO.NS","INDIGO.NS","SHREECEM.NS","TATAMOTORS.NS",
    "UPL.NS","HINDALCO.NS"
]
# remove duplicates if any
NIFTY50 = list(dict.fromkeys(NIFTY50))

PERIOD = "1y"
INTERVAL = "1d"
TODAY = datetime.now().date()
START = TODAY - timedelta(days=365)

# -------- HELPERS: technical indicators --------
def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# -------- FUNDAMENTAL SCORING (simple, conservative checks) --------
def fundamentals_score(ticker_obj):
    info = ticker_obj.info if hasattr(ticker_obj, "info") else {}
    score = 0
    max_score = 5

    # 1) P/E (lower is better moderately) - use trailingPE
    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe and pe > 0:
        if pe < 15:
            score += 2
        elif pe < 25:
            score += 1
    # 2) Debt to equity (use debtToEquity)
    dte = info.get("debtToEquity")
    if dte is not None:
        if dte < 50:
            score += 1
        elif dte < 100:
            score += 0.5
    # 3) Return proxy: returnOnEquity or roa
    roe = info.get("returnOnEquity") or info.get("returnOnAssets")
    if roe is not None:
        # info returns decimal like 0.15 for 15%
        if roe >= 0.15:
            score += 1
        elif roe > 0.08:
            score += 0.5
    # 4) Revenue trend: check last 4 quarters revenue growth from quarterly financials
    try:
        fin = ticker_obj.quarterly_financials
        if isinstance(fin, pd.DataFrame) and not fin.empty:
            # fin has rows like 'Total Revenue' or 'TotalRevenue' depending on ticker data
            # we will approximate by using 'Total Revenue' if available else use 'TotalRevenue' else skip
            rev_row = None
            for candidate in ["Total Revenue", "TotalRevenue", "totalRevenue", "Revenue"]:
                if candidate in fin.index:
                    rev_row = fin.loc[candidate]
                    break
            if rev_row is None:
                # try using the largest row (heuristic)
                rev_row = fin.iloc[0]
            # rev_row is recent quarters left to right; check slope
            rev_vals = rev_row.dropna().astype(float)
            if len(rev_vals) >= 3:
                # simple linear trend: last vs first quarter
                growth = (rev_vals.iloc[0] - rev_vals.iloc[-1]) / (abs(rev_vals.iloc[-1]) + 1e-6)
                # if positive growth (recent higher than earlier), good
                if growth > 0.03:
                    score += 1
    except Exception:
        pass

    return score, max_score, {"pe": pe, "de_ratio": dte, "roe": roe}

# -------- TECHNICAL SCORING --------
def technicals_score(df):
    # df is OHLCV with Datetime index
    close = df["Close"]
    s50 = sma(close, 50)
    s200 = sma(close, 200)
    rsi_val = rsi(close).iloc[-1]
    macd_line, signal_line, hist = macd(close)
    macd_hist = hist.iloc[-1]

    score = 0
    max_score = 5

    # 1) Price vs SMA50 and SMA200
    price = close.iloc[-1]
    if price > s200.iloc[-1]:
        score += 2  # long-term bullish
    elif price > s50.iloc[-1]:
        score += 1  # mid-term okay

    # 2) SMA50 vs SMA200 (golden/death cross)
    if s50.iloc[-1] > s200.iloc[-1]:
        score += 1

    # 3) RSI: moderate is good for midterm (not overheated)
    if 30 < rsi_val < 70:
        score += 1
    elif rsi_val < 30:
        score += 0.5  # oversold - contrarian buy signal sometimes

    # 4) MACD histogram positive
    if macd_hist > 0:
        score += 0.5

    return score, max_score, {"price": price, "s50": s50.iloc[-1], "s200": s200.iloc[-1], "rsi": float(rsi_val), "macd_hist": float(macd_hist)}

# -------- COMBINE LOGIC & RULES --------
def recommend(fund_score, fund_max, tech_score, tech_max, fund_threshold=2.5):
    # Require fundamentals first: fund_score must exceed fund_threshold to be considered for buy
    fund_pct = fund_score / fund_max
    tech_pct = tech_score / tech_max

    # Weighted final score: fundamentals 60%, technicals 40% (midterm investor)
    final_score = 0.6 * fund_pct + 0.4 * tech_pct

    # thresholds can be tuned
    if fund_score < fund_threshold:
        rec = "Sell"  # fundamentals weak -> avoid
    else:
        if final_score >= 0.7:
            rec = "Buy"
        elif final_score >= 0.45:
            rec = "Hold"
        else:
            rec = "Sell"
    return rec, final_score

# -------- MAIN SCAN --------
def scan_universe(ticker_list):
    results = []
    total = len(ticker_list)
    for idx, sym in enumerate(ticker_list, 1):
        print(f"[{idx}/{total}] Scanning {sym} ...", end=" ")

        try:
            tk = yf.Ticker(sym)
            # fetch 1 year daily
            hist = tk.history(period=PERIOD, interval=INTERVAL, actions=False)
            if hist is None or hist.empty:
                print("No price data. Skipping.")
                continue

            # Fundamental scoring
            fscore, fmax, fmeta = fundamentals_score(tk)

            # If fundamentals do not meet minimum, still compute technicals for info but restrict buy
            tscore, tmax, tmeta = technicals_score(hist)

            rec, final_score = recommend(fscore, fmax, tscore, tmax)

            # TradingView link (NSE format)
            tv_symbol = sym.replace(".NS", "")
            tradingview_link = f"https://in.tradingview.com/symbols/NSE:{tv_symbol}/"

            results.append({
                "symbol": sym,
                "current_price": round(tmeta["price"], 2),
                "fund_score": round(fscore, 2),
                "fund_max": fmax,
                "tech_score": round(tscore, 2),
                "tech_max": tmax,
                "final_score": round(final_score, 3),
                "recommendation": rec,
                "tradingview": tradingview_link,
                "fundamental_meta": fmeta,
                "technical_meta": tmeta
            })
            print("Done.")
        except Exception as e:
            print("Error:", str(e))
            continue

    df = pd.DataFrame(results)
    # sort by final score descending
    df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Starting Nifty scanner (1y data). This will take a few minutes.")
    out = scan_universe(NIFTY50)
    if out.empty:
        print("No results.")
    else:
        # Pretty print top buys
        print("\nTop recommendations (Buy / Hold / Sell):")
        print(out[["symbol", "current_price", "fund_score", "tech_score", "final_score", "recommendation", "tradingview"]].to_string(index=False))

        # Save CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nifty_scan_{ts}.csv"
        out.to_csv(filename, index=False)
        print(f"\nSaved results to {filename}")