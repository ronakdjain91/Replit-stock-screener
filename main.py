import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance


def generate_tradingview_link(symbol):
    symbol = symbol.replace(".NS", "")
    return f"https://www.tradingview.com/symbols/{symbol}"


def buySellSignalMACD(df):
    if 'MACDh_12_26_9' not in df.columns:
        return '---'
    if (df['MACDh_12_26_9'].iloc[-1] > 0 and df['MACDh_12_26_9'].iloc[-2] < 0):
        return 'Buy'
    elif (df['MACDh_12_26_9'].iloc[-2] > 0
          and df['MACDh_12_26_9'].iloc[-1] < 0):
        return 'Sell'
    else:
        return '---'


def buySellSignalRSI(df):
    if 'RSI_14' not in df.columns:
        return '---'
    rsi_current = df['RSI_14'].iloc[-1]
    rsi_previous = df['RSI_14'].iloc[-2]
    if rsi_current <= 35 and rsi_previous > 35:
        return 'Buy'
    elif rsi_current >= 55 and rsi_previous < 55:
        return 'Sell'
    else:
        return '---'


def calculate_ut_bot(df, a=1, c=10):
    # Calculate ATR
    atr = df.ta.atr(length=c)
    nLoss = a * atr

    # Initialize trailing stop
    trailing_stop = pd.Series(0.0, index=df.index)

    # Calculate trailing stop
    for i in range(1, len(df)):
        prev_stop = trailing_stop.iloc[i - 1]
        src = df['Close'].iloc[i]
        src_prev = df['Close'].iloc[i - 1]

        if src > prev_stop and src_prev > prev_stop:
            trailing_stop.iloc[i] = max(prev_stop, src - nLoss.iloc[i])
        elif src < prev_stop and src_prev < prev_stop:
            trailing_stop.iloc[i] = min(prev_stop, src + nLoss.iloc[i])
        else:
            trailing_stop.iloc[i] = src - nLoss.iloc[
                i] if src > prev_stop else src + nLoss.iloc[i]

    # Calculate position direction
    pos = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if src_prev < trailing_stop.iloc[i -
                                         1] and src > trailing_stop.iloc[i -
                                                                         1]:
            pos.iloc[i] = 1  # Long
        elif src_prev > trailing_stop.iloc[i -
                                           1] and src < trailing_stop.iloc[i -
                                                                           1]:
            pos.iloc[i] = -1  # Short
        else:
            pos.iloc[i] = pos.iloc[i - 1]

    # Generate buy/sell signals
    buy = (df['Close'] > trailing_stop) & (df['Close'].shift(1)
                                           <= trailing_stop.shift(1))
    sell = (df['Close'] < trailing_stop) & (df['Close'].shift(1)
                                            >= trailing_stop.shift(1))

    return trailing_stop, pos, buy, sell


def buySellSignalUTBot(buy, sell):
    if buy.iloc[-1]:
        return 'Buy'
    elif sell.iloc[-1]:
        return 'Sell'
    else:
        return '---'


nifty_100_stocks = [
    "HDFCBANK.NS", "RELIANCE.NS", "INFY.NS", "TCS.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "ITC.NS", "SBIN.NS", "HCLTECH.NS",
    "ASIANPAINT.NS", "AXISBANK.NS", "LT.NS", "M&M.NS", "BHARTIARTL.NS",
    "MARUTI.NS", "NESTLEIND.NS", "POWERGRID.NS", "SUNPHARMA.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "VEDL.NS", "WIPRO.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS",
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
    "CUB.NS", "DEEPAKNTR.NS", "ABBOTINDIA.NS","APOLLOTYRE.NS"
]

data = []

for stock in nifty_100_stocks:
    try:
        # Download data with auto_adjust=False to avoid MultiIndex
        df = yfinance.download(stock,
                               period='1y',
                               progress=False,
                               interval='1wk',
                               auto_adjust=False)
        if df.empty:
            print(f"No data for {stock}")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in df.columns.get_level_values(0)
            ]
        #print(f"Columns for {stock}: {list(df.columns)}"
              #)  # Debug: Verify columns

        # Ensure 'Close' column exists
        if 'Close' not in df.columns:
            print(
                f"No 'Close' column for {stock}. Columns: {list(df.columns)}")
            continue

        # Calculate MACD, RSI, and UT Bot
        macd = df.ta.macd()
        rsi = df.ta.rsi(length=14)
        trailing_stop, pos, buy, sell = calculate_ut_bot(df, a=1, c=10)

        # Concatenate indicators
        df = pd.concat([
            df, macd, rsi,
            trailing_stop.rename('TrailingStop'),
            pos.rename('Position')
        ],
                       axis=1)
        print(f"Processing {stock}")

        # Check for MACD, RSI, or UT Bot signals
        macd_signal = buySellSignalMACD(df)
        rsi_signal = buySellSignalRSI(df)
        ut_bot_signal = buySellSignalUTBot(buy, sell)
        if macd_signal != '---' or rsi_signal != '---' or ut_bot_signal != '---':
            data.append({
                'Stock':
                stock,
                'Last Day % Movement':
                ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) /
                 df['Close'].iloc[-2]) * 100,
                'Current Price':
                df['Close'].iloc[-1],
                'TradingView Link':
                generate_tradingview_link(stock),
                'MACD_7':
                df['MACDh_12_26_9'].iloc[-7] if len(df) >= 7 else None,
                'MACD_6':
                df['MACDh_12_26_9'].iloc[-6] if len(df) >= 6 else None,
                'MACD_5':
                df['MACDh_12_26_9'].iloc[-5] if len(df) >= 5 else None,
                'MACD_4':
                df['MACDh_12_26_9'].iloc[-4] if len(df) >= 4 else None,
                'MACD_3':
                df['MACDh_12_26_9'].iloc[-3] if len(df) >= 3 else None,
                'MACD_2':
                df['MACDh_12_26_9'].iloc[-2],
                'MACD_1':
                df['MACDh_12_26_9'].iloc[-1],
                'RSI_7':
                df['RSI_14'].iloc[-7] if len(df) >= 7 else None,
                'RSI_6':
                df['RSI_14'].iloc[-6] if len(df) >= 6 else None,
                'RSI_5':
                df['RSI_14'].iloc[-5] if len(df) >= 5 else None,
                'RSI_4':
                df['RSI_14'].iloc[-4] if len(df) >= 4 else None,
                'RSI_3':
                df['RSI_14'].iloc[-3] if len(df) >= 3 else None,
                'RSI_2':
                df['RSI_14'].iloc[-2],
                'RSI_1':
                df['RSI_14'].iloc[-1],
                'TrailingStop_7':
                df['TrailingStop'].iloc[-7] if len(df) >= 7 else None,
                'TrailingStop_6':
                df['TrailingStop'].iloc[-6] if len(df) >= 6 else None,
                'TrailingStop_5':
                df['TrailingStop'].iloc[-5] if len(df) >= 5 else None,
                'TrailingStop_4':
                df['TrailingStop'].iloc[-4] if len(df) >= 4 else None,
                'TrailingStop_3':
                df['TrailingStop'].iloc[-3] if len(df) >= 3 else None,
                'TrailingStop_2':
                df['TrailingStop'].iloc[-2],
                'TrailingStop_1':
                df['TrailingStop'].iloc[-1],
                'MACD Signal':
                macd_signal,
                'RSI Signal':
                rsi_signal,
                'UT Bot Signal':
                ut_bot_signal
            })
    except Exception as e:
        print(f"Error processing {stock}: {e}")
        continue

df_output = pd.DataFrame(data)
if not df_output.empty:
    df_output = df_output.sort_values(by='MACD_7')
    df_output.to_csv('nifty50_signals.csv', index=False)
print('Execution completed')
