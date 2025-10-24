from backtesting import Strategy
import numpy as np
import pandas as pd

class GPT_1(Strategy):
    """
    Regime Shift Detection Strategy using CUSUM filter.
    Detects persistent upward/downward shifts in the signal.
    Buys when an upward shift is detected and sells when a downward shift is detected.
    """

    # Parameters (tunable)
    k = 0.5     # Drift parameter (smaller = more sensitive)
    h = 5.0     # Threshold for regime shift detection
    lookback = 100  # Rolling window for mean estimation

    def init(self):
        # Initialize storage for CUSUM values
        self.S_pos = 0
        self.S_neg = 0
        self.last_regime = 0  # 1 = high, -1 = low, 0 = neutral

    def next(self):
        # Current data point (e.g. price)
        x_t = self.data.Close[-1]

        # Estimate local mean (baseline)
        if len(self.data.Close) < self.lookback:
            mu_0 = np.mean(self.data.Close)
        else:
            mu_0 = np.mean(self.data.Close[-self.lookback:])

        # Update CUSUM statistics
        self.S_pos = max(0, self.S_pos + x_t - mu_0 - self.k)
        self.S_neg = max(0, self.S_neg - (x_t - mu_0) - self.k)

        # Check for regime shifts
        if self.S_pos > self.h:
            # Upward shift → go long
            if self.last_regime != 1:
                self.position.close()
                self.buy()
                self.last_regime = 1
            self.S_pos = 0
            self.S_neg = 0

        elif self.S_neg > self.h:
            # Downward shift → go short
            if self.last_regime != -1:
                self.position.close()
                self.sell()
                self.last_regime = -1
            self.S_pos = 0
            self.S_neg = 0
