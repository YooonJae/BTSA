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
from strategies.stg_TEST_1 import TEST_1

from datetime import datetime
from dateutil.relativedelta import relativedelta

# === Configuration ===
STRATEGY = TEST_1
SYMBOL_INDEX = 7
TIMEFRAME = "m"

# Year–Month selection (day defaults to 1)
START_YEAR  = 2024
START_MONTH = 1
DURATION    = 12   # in months

# Build START_DATE automatically
START_DATE = f"{START_YEAR:04d}-{START_MONTH:02d}-01"



symbols = ["BBAI", "EOSE", "MARA", "RCAT", "RKLB", "SMR", "TMC"]
#symbols = ["AAPL", "AMD", "GOOG", "MSFT", "NVDA", "PYPL", "TSLA"]

symbol = symbols[SYMBOL_INDEX - 1] if 1 <= SYMBOL_INDEX <= len(symbols) else symbols[0]
print(f"✅ Selected symbol: {symbol} ({'Hourly' if TIMEFRAME=='h' else 'Minute'} candles)")

# === Load Data ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "Data", "TransData", "S_cap", f"{symbol}.csv")
#data_path = os.path.join(BASE_DIR, "Data", "TransData", "L_cap", f"{symbol}.csv")

df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Datetime"])
df.set_index("Date", inplace=True)
df = df[["Open", "High", "Low", "Close", "Volume"]]

# === Resample if Hourly ===
if TIMEFRAME == "h":
    df = df.resample("1H").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

# === Helper for date filtering ===
def _parse_duration_months(dur):
    """Return relativedelta object. Accepts int or str ('3M', '10D', etc)."""
    if dur is None:
        return None
    if isinstance(dur, (int, float)):
        return relativedelta(months=int(dur))
    dur = str(dur).strip().upper()
    num = int("".join(ch for ch in dur if ch.isdigit()))
    unit = "".join(ch for ch in dur if ch.isalpha()) or "M"
    if unit == "D":
        return relativedelta(days=num)
    if unit == "W":
        return relativedelta(weeks=num)
    if unit == "M":
        return relativedelta(months=num)
    if unit == "Y":
        return relativedelta(years=num)
    raise ValueError(f"Unsupported duration format: {dur}")

# === Date Filtering ===
if START_DATE:
    start_dt = pd.Timestamp(START_DATE)
    end_dt = start_dt + _parse_duration_months(DURATION)
else:
    end_dt = df.index.max()

# Clip range within available data
start_dt = max(start_dt, df.index.min())
end_dt = min(end_dt, df.index.max())

df = df.loc[start_dt:end_dt]
print(f"📊 Test Range: {start_dt:%Y-%m-%d %H:%M} ~ {end_dt:%Y-%m-%d %H:%M}")

# === Run Backtest ===
bt = Backtest(
    df,
    STRATEGY,
    cash=10_000,
    commission=0.000,
    trade_on_close=True,
    exclusive_orders=True,
    hedging=False
)
result = bt.run()
bt.plot()

# === Trade Statistics ===
trades = result["_trades"].copy()
trades["holding_time_hr"] = (trades["ExitTime"] - trades["EntryTime"]).dt.total_seconds() / 3600
avg_holding_hr = trades["holding_time_hr"].mean()
total_holding_hr = trades["holding_time_hr"].sum()
win_trades = trades[trades["PnL"] > 0]
loss_trades = trades[trades["PnL"] <= 0]
win_loss_ratio = len(win_trades) / len(loss_trades) if len(loss_trades) > 0 else float("inf")
avg_profit_per_trade = trades["ReturnPct"].mean()

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
