import pandas as pd
import os
from backtesting import Backtest
from strategies.stg_MA_1 import MA_1
from strategies.stg_MA_2 import MA_2
from strategies.stg_MA_3 import MA_3
from strategies.stg_BB_1 import BB_1
from strategies.stg_BB_MA_1 import BB_MA_1
from strategies.stg_GPT_1 import GPT_1
from strategies.SAVE.stg_ZRM_1 import ZRM_1


from datetime import datetime
from dateutil.relativedelta import relativedelta

# === Configuration ===
SYMBOL_INDEX = 1
TIMEFRAME = "m"    # 'm' = minute, 'h' = hour
cash = 1_000_000
commission = 0.00
months_ago = 12

symbols = ["BBAI", "EOSE", "MARA", "RCAT", "RKLB", "SMR", "TMC"]
symbol = symbols[SYMBOL_INDEX - 1] if 1 <= SYMBOL_INDEX <= len(symbols) else symbols[0]
print(f"✅ Selected symbol: {symbol} ({'Hourly' if TIMEFRAME=='h' else 'Minute'} candles)")

# === Load Data ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "Data", f"{symbol}.csv")
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Datetime'])
df.set_index('Date', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# === Resample if Hourly ===
if TIMEFRAME == "h":
    df = df.resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    df.dropna(inplace=True)

# === Filter by Date ===
end_date = df.index.max().date()
start_date = (end_date - relativedelta(months=months_ago))
df = df.loc[start_date:end_date]
print(f"📊 Test Range: {start_date} ~ {end_date}")

# === Run Backtest ===
bt = Backtest(
    df,
    ZRM_1,
    cash=1_000_000,
    commission=0.0,              # or your broker-equivalent
    trade_on_close=True,         # enter/exit at signal bar's close
    exclusive_orders=True,       # don't overlap orders
    hedging=False                # long-only or short-only behavior
)
result = bt.run()
bt.plot()

# === Trade Statistics ===
trades = result['_trades']
trades['holding_time_hr'] = (trades['ExitTime'] - trades['EntryTime']).dt.total_seconds() / 3600
avg_holding_hr = trades['holding_time_hr'].mean()
total_holding_hr = trades['holding_time_hr'].sum()
win_trades = trades[trades['PnL'] > 0]
loss_trades = trades[trades['PnL'] <= 0]
win_loss_ratio = len(win_trades) / len(loss_trades) if len(loss_trades) > 0 else float('inf')
avg_profit_per_trade = trades['ReturnPct'].mean()

# === Print Summary ===
print("\n--- Strategy Performance Summary ---")
print(f"Total return: {result['Return [%]']:.2f}%")
print(f"Profit factor: {result['Profit Factor']:.2f}")
print(f"Expectancy: {result['Expectancy [%]']:.2f}%")
print(f"Average holding time per trade: {avg_holding_hr:.2f} hours")
print(f"Total holding time: {total_holding_hr:.2f} hours")
print(f"Win/Loss ratio: {win_loss_ratio:.2f}")
print(f"Average profit per trade: {avg_profit_per_trade:.2f}%")
print(f"Total number of trades: {len(trades)}")
