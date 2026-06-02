import math
import threading
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


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


# ── Utility ───────────────────────────────────────────────────────────────────

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


# ── Native indicators ─────────────────────────────────────────────────────────

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
    low  = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _sma(series, w):
    return series.rolling(window=w, min_periods=1).mean()


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


# ── UT Bot with dynamic multiplier ───────────────────────────────────────────

def calculate_ut_bot(df, a=1.0, c=10):
    try:
        atr = _atr(df, period=c)
        nLoss = a * atr
        trailing_stop = pd.Series(0.0, index=df.index)
        close = df['Close']
        for i in range(1, len(df)):
            prev_stop = trailing_stop.iloc[i - 1]
            src      = close.iloc[i]
            src_prev = close.iloc[i - 1]
            nl       = nLoss.iloc[i]
            if pd.isna(src) or pd.isna(src_prev) or pd.isna(nl):
                trailing_stop.iloc[i] = prev_stop
                continue
            if src > prev_stop and src_prev > prev_stop:
                trailing_stop.iloc[i] = max(prev_stop, src - nl)
            elif src < prev_stop and src_prev < prev_stop:
                trailing_stop.iloc[i] = min(prev_stop, src + nl)
            else:
                trailing_stop.iloc[i] = src - nl if src > prev_stop else src + nl
        buy  = (close > trailing_stop) & (close.shift(1) <= trailing_stop.shift(1))
        sell = (close < trailing_stop) & (close.shift(1) >= trailing_stop.shift(1))
        return buy, sell
    except Exception:
        false_s = pd.Series(False, index=df.index)
        return false_s, false_s


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
    if cur <= 40 and prev > 40:
        return 'Buy'
    if cur >= 60 and prev < 60:
        return 'Sell'
    return '---'


def signal_ema_cross(close):
    """9-period EMA crossing 21-period EMA — short-term momentum signal."""
    if len(close) < 22:
        return '---'
    ema9  = _ema(close, 9)
    ema21 = _ema(close, 21)
    if ema9.iloc[-1] > ema21.iloc[-1] and ema9.iloc[-2] <= ema21.iloc[-2]:
        return 'Buy'
    if ema9.iloc[-1] < ema21.iloc[-1] and ema9.iloc[-2] >= ema21.iloc[-2]:
        return 'Sell'
    return '---'


# ── Consensus ─────────────────────────────────────────────────────────────────

def _consensus(macd_sig, rsi_sig, ut_sig, ema_sig='---'):
    """Return (consensus_label, signal_count) using up to 4 signals."""
    signals = [macd_sig, rsi_sig, ut_sig, ema_sig]
    buys  = sum(1 for s in signals if s == 'Buy')
    sells = sum(1 for s in signals if s == 'Sell')
    if buys == 4:   return 'Strong Buy',  4
    if buys == 3:   return 'Buy',          3
    if sells == 4:  return 'Strong Sell',  4
    if sells == 3:  return 'Sell',         3
    if buys == 2:   return 'Neutral',      2
    if sells == 2:  return 'Neutral',      2
    return 'Neutral', max(buys, sells)


# ── Beta helper ───────────────────────────────────────────────────────────────

def _compute_beta(stock_close, nifty_close):
    try:
        sr = stock_close.pct_change().dropna()
        nr = nifty_close.pct_change().dropna()
        aligned = pd.concat([sr, nr], axis=1, join='inner').dropna()
        if len(aligned) < 30:
            return None
        cov = aligned.cov().iloc[0, 1]
        var = aligned.iloc[:, 1].var()
        if var == 0:
            return None
        return safe_float(cov / var, 2)
    except Exception:
        return None


def _atr_mult_from_beta(beta):
    if beta is None or beta < 0.8:
        return 1.0
    if beta <= 1.2:
        return 1.5
    return 2.0


# ── Technical scan (parallel) ─────────────────────────────────────────────────

def run_technical_scan(stocks, period='1y', interval='1wk', progress_cb=None):
    if period not in VALID_PERIODS:
        period = '1y'
    if interval not in VALID_INTERVALS_TECH:
        interval = '1wk'

    # Fetch Nifty benchmark once for beta
    nifty_close = None
    try:
        nifty_df = yf.download("^NSEI", period="1y", interval="1d",
                               progress=False, auto_adjust=True)
        nifty_df = flatten_columns(nifty_df)
        if not nifty_df.empty and 'Close' in nifty_df.columns:
            nifty_close = nifty_df['Close'].ffill().dropna()
    except Exception as e:
        print(f"[tech] Nifty benchmark fetch failed: {e}")

    results = []
    lock    = threading.Lock()
    counter = [0]
    total   = len(stocks)

    def process_stock(stock):
        try:
            # Always fetch 1y daily for 200-SMA, volume, beta
            daily = yf.download(stock, period="1y", interval="1d",
                                progress=False, auto_adjust=True)
            daily = flatten_columns(daily)

            # Fetch selected period/interval for signal indicators
            if interval == '1d' and period == '1y':
                sig_df = daily
            else:
                sig_df = yf.download(stock, period=period, interval=interval,
                                     progress=False, auto_adjust=True)
                sig_df = flatten_columns(sig_df)

            if daily is None or daily.empty or 'Close' not in daily.columns:
                return None
            if sig_df is None or sig_df.empty or 'Close' not in sig_df.columns:
                return None

            # ── Daily: 200-SMA and trend ──
            daily['Close'] = daily['Close'].ffill()
            daily_close = daily['Close'].dropna()
            if len(daily_close) < 5:
                return None

            sma50      = _sma(daily_close, 50)
            sma200     = _sma(daily_close, 200)
            sma50_val  = safe_float(sma50.iloc[-1], 2)
            sma200_val = safe_float(sma200.iloc[-1], 2)
            price_d    = safe_float(daily_close.iloc[-1], 2)

            # 50 SMA slope: rising if current > 10 bars ago (≈ 2 trading weeks)
            sma50_prev   = safe_float(sma50.iloc[-10], 2) if len(sma50) >= 10 else sma50_val
            sma50_rising = (sma50_val is not None and sma50_prev is not None
                            and sma50_val > sma50_prev)

            # Trend = SMA alignment + slope
            if (price_d is not None and sma50_val is not None and sma200_val is not None
                    and price_d > sma50_val and sma50_val > sma200_val and sma50_rising):
                trend = 'Uptrend'
            elif (price_d is not None and sma50_val is not None and sma200_val is not None
                    and price_d < sma50_val and sma50_val < sma200_val):
                trend = 'Downtrend'
            else:
                trend = 'Sideways'

            # ── Daily: volume spike ──
            vol_ratio = None
            vol_spike = False
            if 'Volume' in daily.columns:
                vol = daily['Volume'].ffill().dropna()
                if len(vol) >= 21:
                    avg_vol   = float(vol.iloc[-21:-1].mean())
                    today_vol = float(vol.iloc[-1])
                    if avg_vol > 0:
                        vol_ratio = safe_float(today_vol / avg_vol, 2)
                        vol_spike = (vol_ratio is not None and vol_ratio >= 1.5)

            # ── Beta ──
            beta     = _compute_beta(daily_close, nifty_close) if nifty_close is not None else None
            atr_mult = _atr_mult_from_beta(beta)

            # ── Signal df: indicators ──
            sig_df['Close'] = sig_df['Close'].ffill()
            close = sig_df['Close'].dropna()
            if len(close) < 3:
                return None

            for col in ('High', 'Low'):
                if col not in sig_df.columns:
                    sig_df[col] = sig_df['Close']
                else:
                    sig_df[col] = sig_df[col].ffill()

            rsi_series = _rsi(close)
            _, _, hist = _macd(close)
            buy_sig, sell_sig = calculate_ut_bot(sig_df, a=atr_mult, c=10)

            macd_sig = signal_macd(hist)
            rsi_sig  = signal_rsi(rsi_series)
            ut_sig   = ('Buy'  if buy_sig.iloc[-1]  else
                        'Sell' if sell_sig.iloc[-1] else '---')
            ema_sig  = signal_ema_cross(close)

            # ── 50 SMA (from signal df) ──
            sma50_s   = _sma(close, 50)
            sma50_val = safe_float(sma50_s.iloc[-1], 2)

            # ── Price / % change ──
            if len(close) < 2:
                return None
            price      = safe_float(close.iloc[-1], 2)
            prev_price = safe_float(close.iloc[-2])
            if price is None or prev_price is None or prev_price == 0:
                return None
            pct_chg       = safe_float(((price - prev_price) / prev_price) * 100, 2)
            rsi_val       = safe_float(rsi_series.dropna().iloc[-1], 1)  if not rsi_series.dropna().empty else None
            macd_hist_val = safe_float(hist.dropna().iloc[-1], 3)        if not hist.dropna().empty      else None

            # ── Consensus (4 signals) ──
            consensus, strength = _consensus(macd_sig, rsi_sig, ut_sig, ema_sig)

            # Suppress Buy signals when price is below 200 SMA (downtrend)
            if trend == 'Downtrend' and consensus in ('Buy', 'Strong Buy'):
                consensus = 'No Signal'
                strength  = 0

            return {
                'Stock':        stock,
                'Consensus':    consensus,
                'Strength':     strength,
                'Trend':        trend,
                'Vol Spike':    'Yes' if vol_spike else 'No',
                'Vol Ratio':    vol_ratio,
                'Price':        price,
                '% Change':     pct_chg,
                'RSI':          rsi_val,
                'MACD Hist':    macd_hist_val,
                'MACD Signal':  macd_sig,
                'RSI Signal':   rsi_sig,
                'UT Bot':       ut_sig,
                'EMA Cross':    ema_sig,
                'Beta':         beta,
                '200 SMA':      sma200_val,
                '50 SMA':       sma50_val,
                'Last Scanned': datetime.now().strftime("%d %b %Y, %H:%M"),
                'Chart':        generate_tv_link(stock),
            }
        except Exception as e:
            print(f"[tech] Error {stock}: {e}")
            return None
        finally:
            with lock:
                counter[0] += 1
                if progress_cb:
                    progress_cb(counter[0], total)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_stock, s) for s in stocks]
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.append(r)

    return results


# ── Fundamental scan (parallel + sector-relative P/E) ────────────────────────

def _pe_relative_score(pe, sector_median_pe):
    """Score P/E relative to sector median instead of absolute thresholds."""
    if pe is None or pe <= 0:
        return 0
    if sector_median_pe is None or sector_median_pe <= 0:
        return 1 if pe < 25 else 0
    ratio = pe / sector_median_pe
    if ratio < 0.8:
        return 2
    if ratio <= 1.1:
        return 1
    return 0


def _fetch_fund_raw(sym, period, interval):
    """Fetch all data needed for one stock in the fundamental scan."""
    tk   = yf.Ticker(sym)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        pass

    hist = tk.history(period=period, interval=interval, actions=False)
    if hist is None or hist.empty or len(hist) < 10:
        return None
    hist = flatten_columns(hist)
    close_col = 'Close' if 'Close' in hist.columns else hist.columns[0]
    close = hist[close_col].ffill().dropna()
    if len(close) < 10:
        return None

    qfin = None
    try:
        qfin = tk.quarterly_financials
    except Exception:
        pass

    pe     = safe_float(info.get('trailingPE') or info.get('forwardPE'))
    dte    = safe_float(info.get('debtToEquity'))
    roe    = safe_float(info.get('returnOnEquity') or info.get('returnOnAssets'))
    pb     = safe_float(info.get('priceToBook'))
    eg     = safe_float(info.get('earningsGrowth'))
    sector = info.get('sector') or 'Unknown'

    return {
        'sym':    sym,
        'close':  close,
        'info':   info,
        'qfin':   qfin,
        'pe':     pe,
        'dte':    dte,
        'roe':    roe,
        'pb':     pb,
        'eg':     eg,
        'sector': sector,
    }


def run_fundamental_scan(stocks, period='1y', interval='1d', progress_cb=None):
    if period not in VALID_PERIODS:
        period = '1y'
    if interval not in VALID_INTERVALS_FUND:
        interval = '1d'

    lock    = threading.Lock()
    counter = [0]
    total   = len(stocks)
    raw_list = []

    # ── Phase 1: parallel data fetch ──
    def fetch_one(sym):
        try:
            return _fetch_fund_raw(sym, period, interval)
        except Exception as e:
            print(f"[fund] Fetch error {sym}: {e}")
            return None
        finally:
            with lock:
                counter[0] += 1
                if progress_cb:
                    progress_cb(counter[0], total)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_one, s) for s in stocks]
        for future in as_completed(futures):
            r = future.result()
            if r:
                raw_list.append(r)

    # ── Phase 2: sector median P/E ──
    sector_pes = {}
    for item in raw_list:
        if item['pe'] and item['pe'] > 0 and item['sector']:
            sector_pes.setdefault(item['sector'], []).append(item['pe'])
    sector_medians = {
        s: float(np.median(pes))
        for s, pes in sector_pes.items() if pes
    }

    # ── Phase 3: score each stock ──
    results = []
    for item in raw_list:
        try:
            sym    = item['sym']
            close  = item['close']
            info   = item['info']
            qfin   = item['qfin']
            pe     = item['pe']
            dte    = item['dte']
            roe    = item['roe']
            pb     = item.get('pb')
            eg     = item.get('eg')
            sector = item['sector']
            sector_med_pe = sector_medians.get(sector)

            price = safe_float(close.iloc[-1], 2)
            if price is None:
                continue

            # Technical sub-score
            s50  = _sma(close, 50)
            s200 = _sma(close, 200)
            rsi_val  = safe_float(_rsi(close).iloc[-1], 1)
            _, _, mh = _macd(close)
            macd_h   = safe_float(mh.iloc[-1], 3)

            tscore    = 0
            s200_last = safe_float(s200.iloc[-1])
            s50_last  = safe_float(s50.iloc[-1])
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

            # Fundamental sub-score (sector-relative P/E)
            fscore = _pe_relative_score(pe, sector_med_pe)

            if dte is not None:
                fscore += 1 if dte < 50 else (0.5 if dte < 100 else 0)
            if roe is not None:
                fscore += 1 if roe >= 0.15 else (0.5 if roe > 0.08 else 0)

            # P/B ratio — value indicator (lower = more undervalued)
            if pb is not None and pb > 0:
                fscore += 1 if pb < 1.5 else (0.5 if pb < 3.0 else 0)

            # Earnings growth — profitability momentum
            if eg is not None:
                fscore += 1 if eg > 0.15 else (0.5 if eg > 0 else 0)

            # Revenue growth — prefer year-over-year comparison
            try:
                if isinstance(qfin, pd.DataFrame) and not qfin.empty:
                    rev_row = None
                    for cand in ['Total Revenue', 'TotalRevenue', 'Revenue']:
                        if cand in qfin.index:
                            rev_row = qfin.loc[cand]
                            break
                    if rev_row is None:
                        rev_row = qfin.iloc[0]
                    rv = rev_row.dropna().astype(float)
                    if len(rv) >= 5:
                        growth = (rv.iloc[0] - rv.iloc[4]) / (abs(rv.iloc[4]) + 1e-6)
                        fscore += 1 if growth > 0.10 else (0.5 if growth > 0 else 0)
                    elif len(rv) >= 3:
                        growth = (rv.iloc[0] - rv.iloc[-1]) / (abs(rv.iloc[-1]) + 1e-6)
                        if growth > 0.03:
                            fscore += 1
            except Exception:
                pass

            fscore = round(fscore, 2)
            final  = round(0.6 * (fscore / 5) + 0.4 * (tscore / 5), 3)
            rec    = 'Sell'
            if fscore >= 2.5:
                rec = 'Buy' if final >= 0.65 else ('Hold' if final >= 0.45 else 'Sell')

            roe_display = safe_float(roe * 100, 1) if roe is not None else None

            results.append({
                'Symbol':           sym,
                'Sector':           sector,
                'Price':            price,
                'P/E':              safe_float(pe, 1),
                'Sector Med P/E':   safe_float(sector_med_pe, 1),
                'D/E':              safe_float(dte, 1),
                'ROE':              roe_display,
                'RSI':              rsi_val,
                'Fund Score':       fscore,
                'Tech Score':       round(tscore, 2),
                'Final Score':      final,
                'Recommendation':   rec,
                'Chart':            generate_tv_link(sym),
            })
        except Exception as e:
            print(f"[fund] Score error {item.get('sym','?')}: {e}")

    return results
