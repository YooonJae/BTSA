import numpy as np
import pandas as pd
from backtesting import Strategy

def _bollinger_bands(series, n=30, n_std=2):
    """Return (middle, upper, lower) Bollinger Bands."""
    ma = pd.Series(series).rolling(n).mean()
    std = pd.Series(series).rolling(n).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma.values, upper.values, lower.values

def crossover(s1, s2): return s1[-2] < s2[-2] and s1[-1] > s2[-1]
def crossunder(s1, s2): return s1[-2] > s2[-2] and s1[-1] < s2[-1]


class BB_1(Strategy):
    n = 30
    n_std = 2

    def init(self):
        close = self.data.Close
        self.ma, self.upper, self.lower = self.I(_bollinger_bands, close, self.n, self.n_std)

    def next(self):
        o, h, l, c = self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1]
        current_time = self.data.index[-1]

        # Skip until enough data
        if np.isnan(self.lower[-1]) or np.isnan(self.upper[-1]):
            return

        if not isinstance(current_time, pd.Timestamp):
            return

        # --- Force close before market close ---
        if current_time.hour == 15 and current_time.minute == 59:
            if self.position:
                self.position.close()
            return

        # --- Skip before 9AM ---
        if current_time.hour < 9:
            return

        # --- ENTRY CONDITION ---
        # Trigger if any of OHLC crosses above lower band
        entered_lower_band = (
            (l <= self.lower[-1] <= h) or
            (crossover([self.data.Close[-2], o], [self.lower[-2], self.lower[-1]]))
        )

        if entered_lower_band and not self.position:
            size = int(self.equity / c)
            if size > 0:
                self.buy(size=size)

        # --- EXIT CONDITION ---
        # Trigger if any of OHLC crosses below upper band
        exited_upper_band = (
            (l <= self.upper[-1] <= h) or
            (crossunder([self.data.Close[-2], o], [self.upper[-2], self.upper[-1]]))
        )

        if exited_upper_band and self.position:
            self.position.close()
