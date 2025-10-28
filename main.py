import os
import re
import pandas as pd
from backtesting import Backtest
from strategies.stg_TEST_1 import TEST_1
from dateutil.relativedelta import relativedelta

# === Configuration ===
STRATEGY    = TEST_1
SYMBOL_KEY  = "l1"     # examples: "s1", "l3", "mv2", "AAPL" (raw ticker)
TIMEFRAME   = 1        # minutes (1, 3, 5, 15, 30, 60, ...)
START_YEAR  = 2025
START_MONTH = 1
DURATION    = 1

# NEW: stitch toggle
STITCH_UNIVERSE = False   # if True and SYMBOL_KEY is like "l3"/"s2", chain the whole "l"/"s" universe

# === Universes (freely expandable) ===
UNIVERSES: dict[str, list[str]] = {
    "s":  ["BBAI", "EOSE", "MARA", "RCAT", "RKLB", "SMR", "TMC"],
    "l":  ["AAPL", "AMD", "GOOG", "MSFT", "NVDA", "PYPL", "TSLA"],
    "x":  ["X:BTCUSD", "X:ETHUSD", "X:LTCUSD", "X:XRPUSD", "X:BCHUSD"],
}

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "Data", "transData")

def _parse_duration(dur):
    if dur is None:
        return None
    if isinstance(dur, (int, float)):
        return relativedelta(months=int(dur))
    s = str(dur).strip().upper()
    num = int("".join(ch for ch in s if ch.isdigit()))
    unit = "".join(ch for ch in s if ch.isalpha()) or "M"
    return {"D": relativedelta(days=num),
            "W": relativedelta(weeks=num),
            "M": relativedelta(months=num),
            "Y": relativedelta(years=num)}.get(unit, relativedelta(months=num))

def resolve_symbol(symbol_key: str) -> tuple[str, str]:
    key = symbol_key.strip()
    m = re.match(r"^([A-Za-z]+)(\d+)$", key)
    if m:
        uni_prefix = m.group(1).lower()
        idx = int(m.group(2)) - 1
        if uni_prefix not in UNIVERSES:
            raise KeyError(f"Unknown universe prefix '{uni_prefix}'. Add it to UNIVERSES.")
        universe = UNIVERSES[uni_prefix]
        if not (0 <= idx < len(universe)):
            raise IndexError(f"Index {idx+1} out of range for universe '{uni_prefix}' (size {len(universe)}).")
        return universe[idx], uni_prefix
    return key.upper(), "<direct>"

def _load_one_csv(ticker: str) -> pd.DataFrame:
    p = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"CSV not found: {p}\nPlace '{ticker}.csv' under 'Data/transData/'."
        )
    df = pd.read_csv(p)

    # normalize datetime column
    if "Datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["Datetime"])
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        raise ValueError(f"{ticker}: expected a 'Datetime' or 'Date' column.")
    df.set_index("Date", inplace=True)

    # normalize OHLCV case; keep only needed columns
    expected = {"open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"}
    cols_map = {c: expected[c.lower()] for c in df.columns if c.lower() in expected}
    df = df.rename(columns=cols_map)[["Open","High","Low","Close","Volume"]].sort_index()

    # optional resample
    if TIMEFRAME > 1:
        df = df.resample(f"{TIMEFRAME}T").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

    return df

def _retime_segment(seg: pd.DataFrame, start_ts: pd.Timestamp, freq_min: int) -> pd.DataFrame:
    """Assign a fresh, strictly increasing DateTimeIndex to the segment."""
    new_idx = pd.date_range(start=start_ts, periods=len(seg), freq=f"{freq_min}T")
    out = seg.copy()
    out.index = new_idx
    return out

def build_stitched_universe(prefix: str) -> pd.DataFrame:
    """Chain the entire universe, scaling each next ticker to last Close of the prior one."""
    tickers = UNIVERSES[prefix]
    stitched = []
    scale_to = None   # prior last Close in stitched price space
    t0 = None         # next segment start time
    freq_min = max(1, int(TIMEFRAME))

    print(f"🔗 Stitched sequence for '{prefix}': {', '.join(tickers)}")

    for i, tk in enumerate(tickers, 1):
        seg = _load_one_csv(tk)
        if seg.empty:
            continue

        # set initial timeline start (first segment keeps its first timestamp)
        if t0 is None:
            t0 = seg.index[0]

        # scale prices to connect
        if scale_to is not None:
            first_close = seg["Close"].iloc[0]
            if first_close and pd.notna(first_close) and first_close != 0:
                factor = scale_to / first_close
            else:
                factor = 1.0
            seg = seg.copy()
            seg[["Open","High","Low","Close"]] *= factor
        # retime to be strictly increasing and continuous
        seg = _retime_segment(seg, start_ts=t0, freq_min=freq_min)

        stitched.append(seg)

        # prep for next segment start/scale
        scale_to = seg["Close"].iloc[-1]
        t0 = seg.index[-1] + pd.Timedelta(minutes=freq_min)

        print(f"  • {tk}: {len(seg)} bars, end Close={seg['Close'].iloc[-1]:.4f}")

    if not stitched:
        raise ValueError(f"No data found to stitch for universe '{prefix}'.")
    out = pd.concat(stitched)
    return out

# === Resolve selection ===
ticker, uni = resolve_symbol(SYMBOL_KEY)

if STITCH_UNIVERSE and uni in UNIVERSES:
    # Build the synthetic continuous series for the whole universe
    df = build_stitched_universe(uni)
    print(f"✅ Selected: {uni.upper()}-universe stitched ({TIMEFRAME}-min) | Bars: {len(df)}")
    print("📊 Test Range: FULL stitched span (ignores START_YEAR/MONTH/DURATION)")
else:
    # Single ticker as before
    print(f"✅ Selected: {ticker} ({TIMEFRAME}-min) [{uni}]")
    df = _load_one_csv(ticker)

    # Date filtering (single ticker only)
    START_DATE = f"{START_YEAR:04d}-{START_MONTH:02d}-01" if START_YEAR and START_MONTH else None
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
    commission=lambda price, size: 0.00 * abs(size),  # 0.0035
    exclusive_orders=True,
    trade_on_close=True,
    hedging=False,
)

result = bt.run()
bt.plot()











# === Trade Statistics ===
# Get equity start/end safely
eq = result.get("_equity_curve", None)
if eq is not None and len(eq) > 0:
    start_equity = float(eq["Equity"].iloc[0])
    end_equity   = float(eq["Equity"].iloc[-1])
else:
    start_equity = float(result.get("Equity Initial [$]", 0.0))
    end_equity   = float(result.get("Equity Final [$]", start_equity))
net_pnl_usd = end_equity - start_equity

trades = result.get("_trades", None)
if trades is None or len(trades) == 0:
    print("No closed trades.")
    print(f"Net P&L: ${net_pnl_usd:,.2f} (from ${start_equity:,.2f} to ${end_equity:,.2f})")
else:
    trades = trades.copy()

    appt_pct = trades["ReturnPct"].mean() * 100
    exp_pct  = result["Expectancy [%]"]

    # Holding time metrics
    trades["holding_time_hr"] = (trades["ExitTime"] - trades["EntryTime"]).dt.total_seconds() / 3600
    avg_holding_hr   = trades["holding_time_hr"].mean()
    total_holding_hr = trades["holding_time_hr"].sum()

    # Win/loss breakdown
    win_trades  = trades[trades["PnL"] > 0]
    loss_trades = trades[trades["PnL"] <= 0]
    win_loss_ratio = (len(win_trades) / len(loss_trades)) if len(loss_trades) else float("inf")
    win_r = len(win_trades) / len(trades)
    avg_w = win_trades["ReturnPct"].mean() if len(win_trades) else 0.0
    avg_l = loss_trades["ReturnPct"].mean() if len(loss_trades) else 0.0
    exp_from_trades_pct = (win_r * avg_w + (1 - win_r) * avg_l) * 100
    avg_win_pct  = win_trades["ReturnPct"].mean() * 100 if len(win_trades) else float("nan")
    avg_loss_pct = loss_trades["ReturnPct"].mean() * 100 if len(loss_trades) else float("nan")

    # Profit per hour metrics
    if total_holding_hr > 0:
        pct_per_hour = result['Return [%]'] / total_holding_hr
        profit_per_hour = trades['PnL'].sum() / total_holding_hr
    else:
        pct_per_hour = float("nan")
        profit_per_hour = float("nan")

    # --- Summary ---
    print("\n--- Strategy Performance Summary ---")
    print(f"Total number of trades: {len(trades)}")
    print(f"Total return: {result['Return [%]']:.2f}%")
    # print(f"Net P&L: ${net_pnl_usd:,.2f} (from ${start_equity:,.2f} to ${end_equity:,.2f})")
    print(f"---")
    print(f"Profit factor: {result['Profit Factor']:.2f}")
    print(f"Expectancy (result): {exp_pct:.2f}%")
    # print(f"Expectancy (recomputed): {exp_from_trades_pct:.2f}%")
    # print(f"Average profit per trade: {appt_pct:.2f}%")
    print(f"---")
    print(f"Average win profit: {avg_win_pct:.2f}%")
    print(f"Average loss profit: {avg_loss_pct:.2f}%")
    print(f"Profit per hour (%): {pct_per_hour:.4f}%/hr")
    print(f"Win/Loss ratio: {win_loss_ratio:.2f}")
    print(f"---")
    print(f"Average holding time per trade: {avg_holding_hr:.2f} hours")
    print(f"Total holding time: {total_holding_hr:.2f} hours")

    # print("\n--- Trade History ---")
    # print(trades[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']].head())