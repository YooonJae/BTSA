import os
import re
import pandas as pd
from backtesting import Backtest
from strategies.stg_TEST_1 import TEST_1
from dateutil.relativedelta import relativedelta

# === Configuration ===
STRATEGY    = TEST_1
SYMBOL_KEY  = "s7"     # examples: "s1", "l3", "mv2", "AAPL" (raw ticker)
TIMEFRAME   = 1        # minutes (1, 3, 5, 15, 30, 60, ...)
START_YEAR  = 2024
START_MONTH = 12
DURATION    = 1        # months, or strings like "10D", "2W", "3M", "1Y"
START_DATE  = f"{START_YEAR:04d}-{START_MONTH:02d}-01"

# === Universes (freely expandable) ===
# Keys are arbitrary alphabetic prefixes you'll use in SYMBOL_KEY.
UNIVERSES: dict[str, list[str]] = {
    "s":  ["BBAI", "EOSE", "MARA", "RCAT", "RKLB", "SMR", "TMC"],           # small cap
    "l":  ["AAPL", "AMD", "GOOG", "MSFT", "NVDA", "PYPL", "TSLA"],          # large cap
    # Add anything you like:
    # "mv": ["SOFI", "PLTR", "NIO", ...],            # mega-volume
    # "etf": ["SPY", "QQQ", "IWM", ...],             # ETFs
    # "kr": ["005930.KS", "000660.KS", ...],         # KOSPI tickers, etc.
}

def _parse_duration(dur):
    """Return relativedelta from int months or strings like '3M', '10D', '2W', '1Y'."""
    if dur is None:
        return None
    if isinstance(dur, (int, float)):
        return relativedelta(months=int(dur))
    s = str(dur).strip().upper()
    num = int("".join(ch for ch in s if ch.isdigit()))
    unit = "".join(ch for ch in s if ch.isalpha()) or "M"
    return {
        "D": relativedelta(days=num),
        "W": relativedelta(weeks=num),
        "M": relativedelta(months=num),
        "Y": relativedelta(years=num),
    }.get(unit, relativedelta(months=num))

def resolve_symbol(symbol_key: str) -> tuple[str, str]:
    """
    Resolve SYMBOL_KEY to (ticker, universe_key).
    SYMBOL_KEY can be:
      - '<letters><digits>' like 's1', 'l7', 'mv3', case-insensitive
      - a raw ticker like 'TSLA' or '005930.KS'
    """
    key = symbol_key.strip()
    m = re.match(r"^([A-Za-z]+)(\d+)$", key)
    if m:
        uni_prefix = m.group(1).lower()
        idx = int(m.group(2)) - 1  # 1-based -> 0-based
        if uni_prefix not in UNIVERSES:
            raise KeyError(f"Unknown universe prefix '{uni_prefix}'. Add it to UNIVERSES.")
        universe = UNIVERSES[uni_prefix]
        if not (0 <= idx < len(universe)):
            raise IndexError(f"Index {idx+1} out of range for universe '{uni_prefix}' (size {len(universe)}).")
        return universe[idx], uni_prefix

    # Otherwise treat as raw ticker
    return key.upper(), "<direct>"

# === Resolve selection ===
ticker, uni = resolve_symbol(SYMBOL_KEY)
print(f"✅ Selected: {ticker} ({TIMEFRAME}-min) [{uni}]")

# === Load Data (all in Data/) ===
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "Data","transData", f"{ticker}.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"CSV not found: {data_path}\n"
        f"Place '{ticker}.csv' under 'Data/' (no subfolders)."
    )

df = pd.read_csv(data_path)

# Normalize datetime column name
if "Datetime" in df.columns:
    df["Date"] = pd.to_datetime(df["Datetime"])
elif "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
else:
    raise ValueError("Expected a 'Datetime' or 'Date' column in the CSV.")

df.set_index("Date", inplace=True)

# Normalize OHLCV case; keep only needed columns
expected = {"open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"}
cols_map = {c: expected[c.lower()] for c in df.columns if c.lower() in expected}
df = df.rename(columns=cols_map)[["Open","High","Low","Close","Volume"]].sort_index()

# Resample if needed
if TIMEFRAME > 1:
    df = df.resample(f"{TIMEFRAME}T").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()

# Date filtering
if START_DATE:
    start_dt = pd.Timestamp(START_DATE)
    end_dt   = start_dt + _parse_duration(DURATION)
else:
    start_dt = df.index.min()
    end_dt   = df.index.max()

start_dt = max(start_dt, df.index.min())
end_dt   = min(end_dt,   df.index.max())
df = df.loc[start_dt:end_dt]
print(f"📊 Test Range: {start_dt:%Y-%m-%d %H:%M} ~ {end_dt:%Y-%m-%d %H:%M}")

# === Run Backtest ===
bt = Backtest(
    df,
    STRATEGY,
    cash=100_000,
    commission=0.00,
    exclusive_orders=False,
    trade_on_close=True,
    hedging=False
)
result = bt.run()
bt.plot()

# === Trade Statistics ===
trades = result["_trades"].copy()
if len(trades) == 0:
    print("No closed trades.")
else:
    appt_pct = trades["ReturnPct"].mean() * 100
    exp_pct  = result["Expectancy [%]"]
    trades["holding_time_hr"] = (trades["ExitTime"] - trades["EntryTime"]).dt.total_seconds() / 3600
    avg_holding_hr   = trades["holding_time_hr"].mean()
    total_holding_hr = trades["holding_time_hr"].sum()
    win_trades  = trades[trades["PnL"] > 0]
    loss_trades = trades[trades["PnL"] <= 0]
    win_loss_ratio = (len(win_trades) / len(loss_trades)) if len(loss_trades) else float("inf")
    win_r = len(win_trades) / len(trades)
    avg_w = win_trades["ReturnPct"].mean() if len(win_trades) else 0.0
    avg_l = loss_trades["ReturnPct"].mean() if len(loss_trades) else 0.0
    exp_from_trades_pct = (win_r * avg_w + (1 - win_r) * avg_l) * 100
    avg_win_pct  = win_trades["ReturnPct"].mean() * 100 if len(win_trades) else float("nan")
    avg_loss_pct = loss_trades["ReturnPct"].mean() * 100 if len(loss_trades) else float("nan")
    win_rate = (len(win_trades) / len(trades) * 100) if len(trades) else float("nan")

    print("\n--- Strategy Performance Summary ---")
    print(f"Total return: {result['Return [%]']:.2f}%")
    print(f"Profit factor: {result['Profit Factor']:.2f}")
    print(f"Expectancy (result): {exp_pct:.2f}%")
    print(f"Expectancy (recomputed): {exp_from_trades_pct:.2f}%")
    print(f"Average profit per trade: {appt_pct:.2f}%")
    print(f"Average holding time per trade: {avg_holding_hr:.2f} hours")
    print(f"Total holding time: {total_holding_hr:.2f} hours")
    print(f"Win/Loss ratio: {win_loss_ratio:.2f}")
    print(f"Average win profit: {avg_win_pct:.2f}%")
    print(f"Average loss profit: {avg_loss_pct:.2f}%")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total number of trades: {len(trades)}")