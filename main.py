import pandas as pd
from numpy import nan as NaN
import pandas_ta as ta
import yfinance


def generate_tradingview_link(symbol):
  symbol = symbol.replace(".NS", "")
  return f"https://www.tradingview.com/symbols/{symbol}"


def buySellSignal(df):
  if (df['MACDh_12_26_9'].iloc[-1] > 0 and df['MACDh_12_26_9'].iloc[-2] < 0):
    return 'Buy'
  elif (df['MACDh_12_26_9'].iloc[-2] > 0 and df['MACDh_12_26_9'].iloc[-1] < 0):
    return 'Sell'
  else:
    return '---'


nifty_100_stocks = ["HDFCBANK.NS", "RELIANCE.NS", "INFY.NS", "TCS.NS", "HINDUNILVR.NS",
  "ICICIBANK.NS", "KOTAKBANK.NS", "ITC.NS", "SBIN.NS", "HCLTECH.NS",
  "ASIANPAINT.NS", "AXISBANK.NS", "LT.NS", "M&M.NS", "BHARTIARTL.NS",
  "MARUTI.NS", "NESTLEIND.NS", "POWERGRID.NS", "SUNPHARMA.NS", "TITAN.NS",
  "ULTRACEMCO.NS", "VEDL.NS", "WIPRO.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS",
  "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS",
  "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GAIL.NS",
  "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "IOC.NS", "INDUSINDBK.NS",
  "JSWSTEEL.NS", "NTPC.NS", "ONGC.NS", "SHREECEM.NS", "TATAMOTORS.NS",
  "TATASTEEL.NS", "TECHM.NS", "UPL.NS", "TATAELXSI.NS",
  # Next 48 stocks for Nifty 100 excluding HDFC.NS
  "WOCKPHARMA.NS", "ZEEL.NS", "ADANIGREEN.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
  "BAJAJHLDNG.NS", "BANDHANBNK.NS", "BERGEPAINT.NS", "COLPAL.NS",
  "DABUR.NS", "DLF.NS", "GODREJCP.NS", "HDFCLIFE.NS", "HINDPETRO.NS",
  "ICICIPRULI.NS", "IDEA.NS", "IGL.NS", "INDIGO.NS", "IOC.NS", "LUPIN.NS", "MANAPPURAM.NS", "MARICO.NS",
  "NHPC.NS", "NMDC.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS",
  "PNB.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", "SAIL.NS",
  "SBILIFE.NS", "SIEMENS.NS", "TATACHEM.NS", "TATACONSUM.NS",
  "UBL.NS", "ICICIGI.NS", "GLENMARK.NS", "SUNTV.NS",
  "PNBHOUSING.NS", "ABCAPITAL.NS", "INDIAMART.NS", "CUB.NS", "DEEPAKNTR.NS",
  "ABBOTINDIA.NS", "PEL.NS", "APOLLOTYRE.NS"
]

data = []

for stock in nifty_100_stocks:
  df = yfinance.download(stock, period='1y', progress=False, interval='1wk')
 # df.to_csv('newfile.csv', index=False)
  macd= df.ta.macd()
  df= pd.concat([df, macd], axis=1)
  #df.to_csv('newfile.csv', index=False)
  print(stock)
  if buySellSignal(df) !='---':
    data.append({
    'Stock':
    stock,
    'Last Day % Movement':
    ((df['Close'].iloc[-1] -  df['Close'].iloc[-2]) / df['Close'].iloc[-2]),
    'Current Price':
    df['Close'].iloc[-1],
    #'Volume': volume,
    'TradingView Link':
    generate_tradingview_link(stock),
    'MACD_7':
    df['MACDh_12_26_9'].iloc[-7],
    'MACD_6':
    df['MACDh_12_26_9'].iloc[-6],
    'MACD_5':
    df['MACDh_12_26_9'].iloc[-5],
    'MACD_4':
    df['MACDh_12_26_9'].iloc[-4],
    'MACD_3':
    df['MACDh_12_26_9'].iloc[-3],
    'MACD_2':
    df['MACDh_12_26_9'].iloc[-2],
    'MACD_1':
    df['MACDh_12_26_9'].iloc[-1],
    'Buy/Sell':
    buySellSignal(df)
  })
df_output = pd.DataFrame(data)
df_output =  df_output.sort_values(by='MACD_7')
df_output.to_csv('nifty50_signals.csv', index=False)
print('execution completed')
#print(macd)
'''macd.insert(0, 'day#', range(1, 249))
df = pd.concat([df, macd], axis=1)
df.to_csv('newfile.csv', index=False)
print(macd['day#'].iloc[0])
print(get_macd_signal(df))'''
