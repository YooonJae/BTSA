from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np

class ZRM_1(Strategy):
    """
    Strategy using Z-score vs rolling mean to detect high/low regimes.
    When price is significantly above the rolling mean -> go short
    When price is significantly below the rolling mean -> go long
    """
    # user-tunable parameters
    window = 50        # rolling window length
    z_entry = 1.5      # threshold for entry
    z_exit = 0.3       # threshold for exit

    def init(self):
        # compute rolling mean and std
        price = self.data.Close
        self.mean = self.I(lambda x: pd.Series(x).rolling(self.window).mean(), price)
        self.std = self.I(lambda x: pd.Series(x).rolling(self.window).std(), price)
        # compute z-score
        self.z = self.I(lambda x, m, s: (x - m) / s, price, self.mean, self.std)

    def next(self):
        z_now = self.z[-1]
        price = self.data.Close[-1]

        # entry signals
        if not self.position:
            if z_now <= -self.z_entry:
                self.buy()
            elif z_now >= self.z_entry:
                self.sell()

        # exit signals (when normalized deviation returns near zero)
        elif self.position.is_long and z_now > -self.z_exit:
            self.position.close()
        elif self.position.is_short and z_now < self.z_exit:
            self.position.close()
