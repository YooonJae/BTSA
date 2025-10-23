# strategies/stg_rsi.py
import numpy as np
import pandas as pd
from backtesting import Strategy


def _sma(series, n):
    return pd.Series(series).rolling(n, min_periods=n).mean().values


class MA_3(Strategy):
    """
    1분봉 기준:
      - 1일 MA: 390 * 5
      - 단기 MA: 390 * 15
      - 장기 MA: 390 * 30

    조건:
      - 모든 MA 기울기 > 0
      - 종가가 모든 MA 위
      -> 진입 시 '한 번에' 풀매수
      - 어느 하나라도 기울기 <= 0 -> 전량 청산
    """

    n_day   = 30
    n_short = 60
    n_long  = 90

    def init(self):
        close = self.data.Close
        self.ma_day   = self.I(_sma, close, self.n_day)
        self.ma_short = self.I(_sma, close, self.n_short)
        self.ma_long  = self.I(_sma, close, self.n_long)

        # ---- Compute all condition arrays ----
        slope_day   = np.r_[np.nan, (self.ma_day[1:] > self.ma_day[:-1]).astype(int)]
        slope_short = np.r_[np.nan, (self.ma_short[1:] > self.ma_short[:-1]).astype(int)]
        slope_long  = np.r_[np.nan, (self.ma_long[1:] > self.ma_long[:-1]).astype(int)]

        price_above_day   = (close > self.ma_day).astype(int)
        price_above_short = (close > self.ma_short).astype(int)
        price_above_long  = (close > self.ma_long).astype(int)

        # ---- Combine into one multi-column indicator (overlapping subplot) ----
        def conds():
            return pd.DataFrame({
                'Slope_Day (cyan)': slope_day,
                'Slope_Short (purple)': slope_short,
                'Slope_Long (green)': slope_long,
                'Price>DayMA (orange)': price_above_day,
                'Price>ShortMA (blue)': price_above_short,
                'Price>LongMA (red)': price_above_long
            })

        self.I(conds, name='MA_Conditions')

    def next(self):
        # Current and previous MA values
        md, md_prev = self.ma_day[-1], self.ma_day[-2]
        ms, ms_prev = self.ma_short[-1], self.ma_short[-2]
        ml, ml_prev = self.ma_long[-1], self.ma_long[-2]
        price = self.data.Close[-1]

        if any(np.isnan(x) for x in (md, md_prev, ms, ms_prev, ml, ml_prev)):
            return

        # ---- Slope & price conditions ----
        slope_day_pos   = md > md_prev
        slope_short_pos = ms > ms_prev
        slope_long_pos  = ml > ml_prev

        all_slopes_pos  = slope_day_pos 
        any_slope_neg   = not all_slopes_pos

        cond_price_above_day   = price > md
        cond_price_above_short = price > ms
        cond_price_above_long  = price > ml

        all_prices_above = (
            cond_price_above_day and
            cond_price_above_short and
            cond_price_above_long
        )

        # ---- Entry condition: all slopes positive & all prices above ----
        long_signal = all_slopes_pos and all_prices_above

        # ---- Exit condition: any slope negative ----
        exit_signal = any_slope_neg

        # ---- Trading logic ----
        if long_signal:
            if not self.position:
                size = int(self.equity / price)
                if size > 0:
                    self.buy(size=size)
        elif exit_signal:
            if self.position:
                self.position.close()
