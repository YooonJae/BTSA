from backtesting import Strategy
import numpy as np
import pandas as pd


class TEST_1(Strategy):
    """
    Detrended Rolling Percentile Strategy (long-only)
    with optional MA level gate.

    Entry:
      - residual percentile < low_threshold
      - AND (if use_ma_filter)  price_now < MA(ma_n)

    Exit:
      - residual percentile > high_threshold
    """

    # --- Core params ---
    window = 390
    low_threshold = 0.1
    high_threshold = 0.9

    # --- MA params ---
    ma_n = 390                 # e.g., 1 trading day of 1-min bars
    use_ma_filter = False      # require price < MA?

    def init(self):
        price = self.data.Close

        # Detrended rolling percentile
        self.percentile = self.I(self._compute_detrended_percentile, price, self.window)

        # Optional MA (for gating)
        self.ma = self.I(self._rolling_mean, price, self.ma_n)

    @staticmethod
    def _rolling_mean(x: np.ndarray, n: int):
        """Simple rolling mean (same length as input)."""
        return pd.Series(x).rolling(n, min_periods=n).mean().values

    @staticmethod
    def _compute_detrended_percentile(price: np.ndarray, window: int):
        """Compute rolling percentile of detrended residuals."""
        n = len(price)
        pct = np.full(n, np.nan)
        t = np.arange(window)

        for i in range(window - 1, n):
            y = price[i - window + 1:i + 1]
            b, a = np.polyfit(t, y, 1)  # linear detrend
            residuals = y - (a + b * t)
            now_resid = residuals[-1]
            pct[i] = np.mean(residuals <= now_resid)
        return pct

    def _entry_gate_ok(self, price_now: float) -> bool:
        """Check MA gate if enabled."""
        if not self.use_ma_filter:
            return True
        if np.isnan(self.ma[-1]):
            return False
        # Gate condition: price < MA
        return price_now < self.ma[-1]

    def next(self):
        p = self.percentile[-1]
        if np.isnan(p):
            return

        price_now = self.data.Close[-1]

        # Entry rule
        if (p < self.low_threshold) and not self.position and self._entry_gate_ok(price_now):
            self.buy()

        # Exit rule
        elif (p > self.high_threshold) and self.position.is_long:
            self.position.close()