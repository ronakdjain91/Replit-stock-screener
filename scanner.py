import math
import threading
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


# ── Stock Universe (updated June 2026) ────────────────────────────────────────
# Nifty 100 = Nifty 50 + Nifty Next 50

DEFAULT_NIFTY50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BEL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "MCDOWELL-N.NS", "WIPRO.NS"
]

DEFAULT_NIFTY_NEXT50 = [
    "ABB.NS", "ABBOTINDIA.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BANKBARODA.NS", "BERGEPAINT.NS", "BOSCHLTD.NS", "BPCL.NS",
    "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS",
    "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAL.NS",
    "HAVELLS.NS", "ICICIPRULI.NS", "ICICIGI.NS", "INDIGO.NS",
    "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS", "JSWENERGY.NS",
    "LICI.NS", "LUPIN.NS", "MARICO.NS", "NAUKRI.NS",
    "NHPC.NS", "NMDC.NS", "OFSS.NS", "PFC.NS",
    "PIDILITIND.NS", "PNB.NS", "RECLTD.NS", "SAIL.NS",
    "SIEMENS.NS", "SRF.NS", "TATAELXSI.NS", "TATAPOWER.NS",
    "TORNTPHARM.NS", "TRENT.NS", "TVSMOTORS.NS", "UPL.NS",
    "VEDL.NS", "VBL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS",
    "DEEPAKNTR.NS", "SHREECEM.NS"
]

DEFAULT_NIFTY100 = DEFAULT_NIFTY50 + DEFAULT_NIFTY_NEXT50

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
    return f"https://in.tradingview.com/symbols/NSE:{symbol.replace('.NS', '')}"


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


def _adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength (>25 = strong trend)."""
    try:
        tr_list, pdm_list, ndm_list = [], [], []
        for i in range(1, len(close)):
            h  = float(high.iloc[i]);  l  = float(low.iloc[i])
            ph = float(high.iloc[i-1]); pl = float(low.iloc[i-1])
            pc = float(close.iloc[i-1])
            tr   = max(h - l, abs(h - pc), abs(l - pc))
            pdm  = max(h - ph, 0) if (h - ph) > (pl - l) else 0
            ndm  = max(pl - l, 0) if (pl - l) > (h - ph) else 0
            tr_list.append(tr); pdm_list.append(pdm); ndm_list.append(ndm)
        tr_s  = pd.Series(tr_list,  dtype=float)
        pdm_s = pd.Series(pdm_list, dtype=float)
        ndm_s = pd.Series(ndm_list, dtype=float)
        atr   = tr_s.ewm(span=period, adjust=False).mean()
        pdi   = 100 * pdm_s.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        ndi   = 100 * ndm_s.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
        adx   = dx.ewm(span=period, adjust=False).mean()
        return safe_float(adx.dropna().iloc[-1], 1) if not adx.dropna().empty else None
    except Exception:
        return None

def _bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band


def _vwma(close, volume, window=20):
    if volume is None or volume.empty:
        return _sma(close, window)
    vp = close * volume
    return vp.rolling(window=window, min_periods=1).sum() / volume.rolling(window=window, min_periods=1).sum()


def _supertrend(high, low, close, period=10, multiplier=3):
    atr = _atr(pd.DataFrame({'High': high, 'Low': low, 'Close': close}), period)
    hl2 = (high + low) / 2
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(0.0, index=close.index)
    direction = pd.Series(1, index=close.index)

    for i in range(1, len(close)):
        if close.iloc[i] > final_upperband.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < final_lowerband.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1 and final_lowerband.iloc[i] < final_lowerband.iloc[i-1]:
                final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            if direction.iloc[i] == -1 and final_upperband.iloc[i] > final_upperband.iloc[i-1]:
                final_upperband.iloc[i] = final_upperband.iloc[i-1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = final_lowerband.iloc[i]
        else:
            supertrend.iloc[i] = final_upperband.iloc[i]

    return supertrend, direction


def _stoch_rsi(close, period=14, smoothK=3, smoothD=3):
    rsi = _rsi(close, period)
    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
    k = stoch_rsi.rolling(window=smoothK).mean()
    d = k.rolling(window=smoothD).mean()
    return k, d


def _analyze_trend(daily_close, daily_high=None, daily_low=None, daily_vol=None):
    """
    Multi-factor trend analysis.
    Factors: MA alignment, HH+HL structure, RSI, volume, breakout, MACD, ADX.
    Returns: (classification, score_0_to_100, reasons_list, warnings_list)
    """
    reasons  = []
    warnings = []
    score    = 0
    n        = len(daily_close)

    if n < 30:
        return 'No Uptrend', 0, reasons, warnings

    price = float(daily_close.iloc[-1])

    # ── 1. Moving Average Alignment (max 30 pts) ──────────────────────────────
    sma20  = float(_sma(daily_close, 20).iloc[-1])
    sma50  = float(_sma(daily_close, 50).iloc[-1])
    sma200 = float(_sma(daily_close, 200).iloc[-1]) if n >= 200 else None

    ma_pts = 0
    ma_ok  = []
    if price > sma20:               ma_pts += 10; ma_ok.append("Price>20DMA")
    if sma20  > sma50:              ma_pts += 10; ma_ok.append("20>50DMA")
    if sma200 and sma50 > sma200:   ma_pts += 10; ma_ok.append("50>200DMA")
    score += ma_pts

    if ma_pts == 30:
        reasons.append(f"Full MA alignment ({', '.join(ma_ok)})")
    elif ma_pts >= 20:
        reasons.append(f"Partial MA alignment ({', '.join(ma_ok)})")
    elif ma_pts >= 10:
        reasons.append(f"Weak MA alignment ({ma_ok[0]})")

    if sma200 and price < sma200:
        warnings.append("Price below 200 DMA — long-term bearish structure")

    # ── 2. HH + HL Price Structure (max 20 pts) ────────────────────────────────
    chunk_size = 10
    n_chunks   = min(6, n // chunk_size)
    chunks     = []
    for i in range(n_chunks - 1, -1, -1):
        end   = n - i * chunk_size
        start = end - chunk_size
        if start >= 0:
            ch = daily_close.iloc[start:end]
            chunks.append((float(ch.max()), float(ch.min())))

    hh_hl_streak = 0
    for i in range(1, len(chunks)):
        if chunks[i][0] > chunks[i-1][0] and chunks[i][1] > chunks[i-1][1]:
            hh_hl_streak += 1
        else:
            hh_hl_streak = 0

    if hh_hl_streak >= 3:
        score += 20; reasons.append(f"Strong HH+HL structure ({hh_hl_streak} consecutive periods)")
    elif hh_hl_streak >= 2:
        score += 12; reasons.append(f"HH+HL structure developing ({hh_hl_streak} periods)")
    elif hh_hl_streak >= 1:
        score += 5;  reasons.append("Early HH+HL pattern (1 period)")
    else:
        warnings.append("No consistent HH+HL pattern")

    # ── 3. RSI Trend Strength (max 15 pts) ────────────────────────────────────
    rsi_d = _rsi(daily_close)
    rsi_v = float(rsi_d.iloc[-1])

    if 55 <= rsi_v <= 70:
        score += 15; reasons.append(f"RSI in healthy uptrend zone ({rsi_v:.0f})")
    elif 50 <= rsi_v < 55:
        score += 8;  reasons.append(f"RSI above midpoint ({rsi_v:.0f})")
    elif rsi_v > 75:
        score += 5
        warnings.append(f"RSI overbought ({rsi_v:.0f}) — reversal risk, avoid new entries")
    elif rsi_v < 45:
        warnings.append(f"RSI weak ({rsi_v:.0f}) — uptrend momentum lacking")

    # ── 4. Volume Confirmation (max 15 pts) ────────────────────────────────────
    if daily_vol is not None and len(daily_vol) >= 20:
        vol_clean  = daily_vol.ffill().dropna()
        avg_vol    = float(vol_clean.iloc[-20:].mean())
        recent_vol = float(vol_clean.iloc[-1])
        pct_5d     = float(daily_close.pct_change(5).iloc[-1]) if n >= 6 else 0
        vol_ratio  = recent_vol / avg_vol if avg_vol > 0 else 1.0

        if pct_5d > 0 and vol_ratio >= 1.2:
            score += 15; reasons.append(f"Volume confirming price rise ({vol_ratio:.1f}× avg)")
        elif pct_5d > 0 and vol_ratio >= 0.8:
            score += 7;  reasons.append("Volume neutral with rising price")
        elif pct_5d > 0 and vol_ratio < 0.7:
            warnings.append("Rising price on declining volume — weak conviction")
        elif pct_5d < 0 and vol_ratio >= 1.2:
            warnings.append("High volume on price decline — distribution signal")

    # ── 5. Breakout from Resistance (max 10 pts) ──────────────────────────────
    high_52w = float(daily_close.iloc[-min(252, n):].max())
    high_20d = float(daily_close.iloc[-min(20, n):].max())

    if price >= high_52w * 0.99:
        score += 10; reasons.append("At/near 52-week high (breakout zone)")
    elif price >= high_20d * 0.995:
        score += 5;  reasons.append("Breaking 20-day resistance")

    # ── 6. MACD Momentum (max 10 pts) ─────────────────────────────────────────
    _, _, macd_h = _macd(daily_close)
    h_now  = float(macd_h.iloc[-1])
    h_prev = float(macd_h.iloc[-2]) if n >= 2 else h_now

    if h_now > 0 and h_now > h_prev:
        score += 10; reasons.append("MACD bullish and strengthening")
    elif h_now > 0:
        score += 5;  reasons.append("MACD above signal line (bullish)")
    elif h_now < 0 and h_now < h_prev:
        warnings.append("MACD bearish and weakening")
    elif h_now < 0:
        warnings.append("MACD below signal line (bearish)")

    # ── 7. ADX Trend Strength (bonus 5 pts) ────────────────────────────────────
    if daily_high is not None and daily_low is not None and len(daily_high) >= 28:
        adx_v = _adx(daily_high, daily_low, daily_close)
        if adx_v is not None:
            if adx_v >= 25:
                score += 5; reasons.append(f"ADX {adx_v:.0f} — strong trend confirmed")
            elif adx_v < 20:
                warnings.append(f"ADX {adx_v:.0f} — weak/no directional trend")

    score = min(int(round(score)), 100)

    if score >= 65:
        classification = 'Strong Uptrend'
    elif score >= 35:
        classification = 'Moderate Uptrend'
    else:
        classification = 'No Uptrend'

    return classification, score, reasons, warnings


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


def signal_rsi(rsi_series, buy_thresh=30, sell_thresh=70):
    r = rsi_series.dropna()
    if len(r) < 2:
        return '---'
    cur, prev = r.iloc[-1], r.iloc[-2]
    if cur <= buy_thresh and prev > buy_thresh:
        return 'Buy'
    if cur >= sell_thresh and prev < sell_thresh:
        return 'Sell'
    return '---'


def signal_bb(close, lower_band, upper_band):
    if len(close) < 2:
        return '---'
    if close.iloc[-1] > lower_band.iloc[-1] and close.iloc[-2] <= lower_band.iloc[-2]:
        return 'Buy'
    if close.iloc[-1] < upper_band.iloc[-1] and close.iloc[-2] >= upper_band.iloc[-2]:
        return 'Sell'
    return '---'


def signal_supertrend(direction):
    if len(direction) < 2:
        return '---'
    if direction.iloc[-1] == 1 and direction.iloc[-2] == -1:
        return 'Buy'
    if direction.iloc[-1] == -1 and direction.iloc[-2] == 1:
        return 'Sell'
    return '---'


def signal_stoch_rsi(k, d):
    if len(k) < 2 or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return '---'
    if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] < 20:
        return 'Buy'
    if k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] > 80:
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

def _consensus(macd_sig, rsi_sig, ut_sig, ema_sig, bb_sig='---', st_sig='---', stoch_sig='---'):
    """Return (consensus_label, signal_count) using up to 7 signals."""
    signals = [macd_sig, rsi_sig, ut_sig, ema_sig, bb_sig, st_sig, stoch_sig]
    buys  = sum(1 for s in signals if s == 'Buy')
    sells = sum(1 for s in signals if s == 'Sell')
    
    if buys >= 5:   return 'Strong Buy',  buys
    if buys >= 4:   return 'Buy',         buys
    if sells >= 5:  return 'Strong Sell', sells
    if sells >= 4:  return 'Sell',        sells
    
    if buys >= 2:   return 'Neutral',     buys
    if sells >= 2:  return 'Neutral',     sells
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

def run_technical_scan(stocks, period='1y', interval='1wk', progress_cb=None, rsi_buy_thresh=30, rsi_sell_thresh=70):
    if period not in VALID_PERIODS:
        period = '1y'
    if interval not in VALID_INTERVALS_TECH:
        interval = '1wk'

    # Fetch Nifty benchmark once for beta
    nifty_close = None
    try:
        nifty_df = yf.Ticker("^NSEI").history(period="1y", interval="1d")
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
            ticker = yf.Ticker(stock)
            # Always fetch 1y daily for 200-SMA, volume, beta
            daily = ticker.history(period="1y", interval="1d")

            # Fetch selected period/interval for signal indicators
            if interval == '1d' and period == '1y':
                sig_df = daily
            else:
                sig_df = ticker.history(period=period, interval=interval)

            if daily is None or daily.empty or 'Close' not in daily.columns:
                return None
            if sig_df is None or sig_df.empty or 'Close' not in sig_df.columns:
                return None

            # ── Daily: 200-SMA and trend ──
            daily['Close'] = daily['Close'].ffill()
            daily_close = daily['Close'].dropna()
            if len(daily_close) < 5:
                return None

            # ── Reference SMAs (kept for display) ──
            sma50_val  = safe_float(_sma(daily_close, 50).iloc[-1], 2)
            sma200_val = safe_float(_sma(daily_close, 200).iloc[-1], 2) if len(daily_close) >= 200 else None

            # ── High / Low / Volume series for trend analysis ──
            daily_high_s = daily['High'].ffill()   if 'High'   in daily.columns else None
            daily_low_s  = daily['Low'].ffill()    if 'Low'    in daily.columns else None
            daily_vol_s  = daily['Volume'].ffill() if 'Volume' in daily.columns else None

            # ── Multi-factor trend analysis ──
            trend, trend_score, trend_reasons, trend_warnings = _analyze_trend(
                daily_close, daily_high_s, daily_low_s, daily_vol_s
            )
            trend_reasons_str  = ' | '.join(trend_reasons)
            trend_warnings_str = ' | '.join(trend_warnings)

            # ── 200 EMA cross — did price cross 200-day EMA in last 10 candles? ──
            ema200       = _ema(daily_close, 200) if len(daily_close) >= 200 else _ema(daily_close, len(daily_close))
            ema200_cross = '---'
            look = min(10, len(daily_close) - 1)
            for i in range(1, look + 1):
                pp = float(daily_close.iloc[-i - 1])
                cp = float(daily_close.iloc[-i])
                pe = float(ema200.iloc[-i - 1])
                ce = float(ema200.iloc[-i])
                if pp <= pe and cp > ce:
                    ema200_cross = 'Bullish Cross'
                    break
                elif pp >= pe and cp < ce:
                    ema200_cross = 'Bearish Cross'
                    break

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
            
            _, upper_bb, lower_bb = _bollinger_bands(close)
            vwma = _vwma(close, sig_df.get('Volume', None))
            st_series, st_dir = _supertrend(sig_df['High'], sig_df['Low'], close)
            stoch_k, stoch_d = _stoch_rsi(close)

            macd_sig = signal_macd(hist)
            rsi_sig  = signal_rsi(rsi_series, buy_thresh=rsi_buy_thresh, sell_thresh=rsi_sell_thresh)
            ut_sig   = ('Buy'  if buy_sig.iloc[-1]  else
                        'Sell' if sell_sig.iloc[-1] else '---')
            ema_sig  = signal_ema_cross(close)
            bb_sig   = signal_bb(close, lower_bb, upper_bb)
            st_sig   = signal_supertrend(st_dir)
            stoch_sig= signal_stoch_rsi(stoch_k, stoch_d)

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

            # ── Consensus (7 signals) ──
            consensus, strength = _consensus(macd_sig, rsi_sig, ut_sig, ema_sig, bb_sig, st_sig, stoch_sig)

            return {
                'Stock':           stock,
                'Consensus':       consensus,
                'Strength':        strength,
                'Trend':           trend,
                'Trend Score':     trend_score,
                'Trend Reasons':   trend_reasons_str,
                'Trend Warnings':  trend_warnings_str,
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
                'Bollinger':    bb_sig,
                'Supertrend':   st_sig,
                'StochRSI':     stoch_sig,
                'VWAP':         safe_float(vwma.dropna().iloc[-1], 2) if not vwma.dropna().empty else None,
                '200 EMA Cross': ema200_cross,
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
    roe    = safe_float(info.get('returnOnEquity'))
    roa    = safe_float(info.get('returnOnAssets'))
    pb     = safe_float(info.get('priceToBook'))
    eg     = safe_float(info.get('earningsGrowth'))
    dy     = safe_float(info.get('dividendYield'))
    sector = info.get('sector') or 'Unknown'

    return {
        'sym':    sym,
        'close':  close,
        'info':   info,
        'qfin':   qfin,
        'pe':     pe,
        'dte':    dte,
        'roe':    roe,
        'roa':    roa,
        'pb':     pb,
        'eg':     eg,
        'dy':     dy,
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
            roa    = item['roa']
            pb     = item.get('pb')
            eg     = item.get('eg')
            dy     = item.get('dy')
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

            is_financial = sector in ['Financial Services', 'Banks']

            if dte is not None:
                if is_financial:
                    # High debt is normal for financials
                    fscore += 1 if dte < 500 else (0.5 if dte < 1000 else 0)
                else:
                    fscore += 1 if dte < 50 else (0.5 if dte < 100 else 0)

            if roe is not None:
                fscore += 1 if roe >= 0.15 else (0.5 if roe > 0.08 else 0)
            elif roa is not None:
                # ROA is typically lower than ROE, adjust thresholds
                fscore += 1 if roa >= 0.05 else (0.5 if roa > 0.02 else 0)

            # P/B ratio — value indicator (lower = more undervalued)
            if pb is not None and pb > 0:
                if is_financial:
                    fscore += 1 if pb < 2.0 else (0.5 if pb < 4.0 else 0)
                else:
                    fscore += 1 if pb < 1.5 else (0.5 if pb < 3.0 else 0)
                    
            # Dividend yield
            if dy is not None:
                fscore += 1 if dy > 0.02 else (0.5 if dy > 0.01 else 0)

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
            final  = round(0.6 * min(fscore / 8.0, 1.0) + 0.4 * min(tscore / 4.5, 1.0), 3)
            rec    = 'Sell'
            if fscore >= 2.5:
                rec = 'Buy' if final >= 0.65 else ('Hold' if final >= 0.45 else 'Sell')

            roe_display = safe_float(roe * 100, 1) if roe is not None else None
            roa_display = safe_float(roa * 100, 1) if roa is not None else None
            dy_display  = safe_float(dy * 100, 2) if dy is not None else None

            results.append({
                'Symbol':           sym,
                'Sector':           sector,
                'Price':            price,
                'P/E':              safe_float(pe, 1),
                'Sector Med P/E':   safe_float(sector_med_pe, 1),
                'D/E':              safe_float(dte, 1),
                'ROE':              roe_display,
                'ROA':              roa_display,
                'P/B':              safe_float(pb, 2),
                'Div Yield':        dy_display,
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
