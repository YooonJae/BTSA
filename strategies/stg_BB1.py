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


class BB1(Strategy):
    """
    Bollinger Band strategy (390-candle window)
    - Buy when Close crosses above the lower band
    - Sell when Close crosses below the upper band
    - Use all available equity (integer shares only)
    """
    n = 390
    n_std = 2

    def init(self):
        close = self.data.Close
        self.ma, self.upper, self.lower = self.I(_bollinger_bands, close, self.n, self.n_std)

    def next(self):
        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        upper, lower = self.upper[-1], self.lower[-1]
        prev_upper, prev_lower = self.upper[-2], self.lower[-2]

        # --- Buy when crossing above lower band ---
        if prev_price < prev_lower and price > lower and not self.position:
            size = int(self.equity / price)  # ✅ ensure whole shares
            if size > 0:
                self.buy(size=size)

        # --- Sell when crossing below upper band ---
        elif prev_price > prev_upper and price < upper and self.position:
            self.position.close()
