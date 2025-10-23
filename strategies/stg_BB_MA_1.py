# strategies/stg_bb1.py
import numpy as np
import pandas as pd
from backtesting import Strategy


def _bollinger_bands(series, n=390, n_std=2):
    """Return (middle, upper, lower) Bollinger Bands."""
    ma = pd.Series(series).rolling(n, min_periods=n).mean()
    std = pd.Series(series).rolling(n, min_periods=n).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma.values, upper.values, lower.values


def _sma(series, n=390):
    """Simple Moving Average."""
    return pd.Series(series).rolling(n, min_periods=n).mean().values


class BB_MA_1(Strategy):
    """
    Bollinger Band + MA Trend Filter (390-candle window)
    - Buy when Close crosses above the lower band AND MA slope > 0
    - Sell when:
        1) Close crosses below the upper band, OR
        2) Close falls below MA (trend breakdown)
    - Use all available equity (integer shares only)
    """
    n = 390
    n_std = 2
    n_ma = 390 * 5  # 5-day MA (on 1-min candles)

    def init(self):
        close = self.data.Close
        self.ma, self.upper, self.lower = self.I(_bollinger_bands, close, self.n, self.n_std)
        self.trend_ma = self.I(_sma, close, self.n_ma)

    def next(self):
        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        upper, lower = self.upper[-1], self.lower[-1]
        prev_upper, prev_lower = self.upper[-2], self.lower[-2]
        ma_value = self.trend_ma[-1]

        # Compute MA slope (current - previous)
        ma_slope = self.trend_ma[-1] - self.trend_ma[-2]

        # --- Buy when crossing above lower band and trend is upward ---
        if prev_price < prev_lower and price > lower and not self.position:
            if ma_slope > 0:  # ✅ only buy when MA slope is positive
                size = int(self.equity / price)
                if size > 0:
                    self.buy(size=size)

        # --- Sell when crossing below upper band OR price below MA ---
        elif self.position:
            if (prev_price > prev_upper and price < upper) or (price < ma_value):
                self.position.close()
