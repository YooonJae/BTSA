# strategies/stg_ma_cross_any.py
import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


def crossunder(series1, series2):
    """Return True where series1 crosses *below* series2."""
    s1, s2 = np.asarray(series1), np.asarray(series2)
    return (s1[-2] > s2[-2]) and (s1[-1] < s2[-1])


def _sma(series, n):
    """Simple Moving Average"""
    return pd.Series(series).rolling(n, min_periods=n).mean().values


class MA_1(Strategy):
    """
    Strategy:
      - MAs: 30, 90, 270 (on 1-min candles)
      - BUY (ALL IN): price crosses ANY MA upward (only if n_short > n_long)
      - SELL (ALL OUT): price crosses ANY MA downward
      - SELL priority to avoid flip-flop on same bar
    """

    n_1   = 30
    n_2 = 90
    n_3  = 270

    def init(self):
        close = self.data.Close
        self.ma_1   = self.I(_sma, close, self.n_1)
        self.ma_2 = self.I(_sma, close, self.n_2)
        self.ma_3  = self.I(_sma, close, self.n_3)

    def next(self):
        if len(self.data.Close) < 2 or np.isnan(self.ma_3[-1]):
            return

        mas = (self.ma_1, self.ma_2, self.ma_3)

        cross_up   = any(crossover(self.data.Close, ma)  for ma in mas)
        cross_down = any(crossunder(self.data.Close, ma) for ma in mas)

        # --- Priority: SELL first ---
        if self.position and cross_down:
            self.position.close()
            return

        # --- BUY condition: only if short MA > long MA ---
        if (not self.position 
            and cross_up ):

            price = float(self.data.Close[-1])
            size = int(self.equity // price)
            if size > 0:
                self.buy(size=size)
