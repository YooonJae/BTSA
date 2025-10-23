# strategies/stg_MA2.py
import numpy as np
import pandas as pd
from backtesting import Strategy


def _sma(series, n):
    return pd.Series(series).rolling(n, min_periods=n).mean().values


class MA_2(Strategy):
    """
    1분봉 기준:
      - 1일 MA: 390
      - 단기 MA: n_short
      - 30일 MA: n_long
    조건:
      - 모든 MA의 기울기 > 0 → 진입 (BUY)
      - 하나라도 기울기 <= 0 → 청산 (SELL)
    """
    n_day   = 195 # 390*1
    n_short = 195 # 390*1
    n_long  = 195 # 390*1

    def init(self):
        close = self.data.Close
        self.ma_day   = self.I(_sma, close, self.n_day)
        self.ma_short = self.I(_sma, close, self.n_short)
        self.ma_long  = self.I(_sma, close, self.n_long)

        # ---- Compute slope conditions ----
        slope_day   = np.r_[np.nan, (self.ma_day[1:]   > self.ma_day[:-1]).astype(int)]
        slope_short = np.r_[np.nan, (self.ma_short[1:] > self.ma_short[:-1]).astype(int)]
        slope_long  = np.r_[np.nan, (self.ma_long[1:]  > self.ma_long[:-1]).astype(int)]

        # ---- Combine all into one multi-column indicator subplot ----
        def conds():
            return pd.DataFrame({
                'Day MA slope>0 (green)':   slope_day,
                'Short MA slope>0 (blue)':  slope_short,
                'Long MA slope>0 (red)':    slope_long
            })

        self.I(conds, name='MA_Slope_Conditions')

    def next(self):
        # Current & previous MAs
        md, md_prev = self.ma_day[-1], self.ma_day[-2]
        ms, ms_prev = self.ma_short[-1], self.ma_short[-2]
        ml, ml_prev = self.ma_long[-1], self.ma_long[-2]
        price = self.data.Close[-1]

        if any(np.isnan(x) for x in (md, md_prev, ms, ms_prev, ml, ml_prev)):
            return

        # ---- Slopes ----
        cond_day_slope   = md > md_prev
        cond_short_slope = ms > ms_prev
        cond_long_slope  = ml > ml_prev

        # ---- Buy when all slopes positive ----
        all_slopes_positive = cond_day_slope and cond_short_slope and cond_long_slope
        any_slope_negative  = not all_slopes_positive

        if all_slopes_positive:
            if not self.position:
                size = int(self.equity / price)
                if size > 0:
                    self.buy(size=size)
        elif any_slope_negative:
            if self.position:
                self.position.close()
