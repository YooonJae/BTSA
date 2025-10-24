# strategies/stg_EMA2.py
import numpy as np
import pandas as pd
from backtesting import Strategy


def _ema(series, n):
    """Exponential Moving Average"""
    return pd.Series(series).ewm(span=n, adjust=False, min_periods=n).mean().values


class MA_2(Strategy):
    """
    1분봉 기준:
      - EMA 1: 30
      - EMA 2: 60
      - EMA 3: 90
    조건:
      - 모든 EMA의 기울기 > 0 → 진입 (BUY)
      - 하나라도 기울기 <= 0 → 청산 (SELL)
      - 오후 3시 59분 (15:59) → 보유 포지션 전부 청산
      - 오전 9시 30분 이전에는 진입하지 않음
    """
    n_1 = 30
    n_2 = 60
    n_3 = 90

    def init(self):
        close = self.data.Close
        self.ema_1 = self.I(_ema, close, self.n_1)
        self.ema_2 = self.I(_ema, close, self.n_2)
        self.ema_3 = self.I(_ema, close, self.n_3)

        slope_1 = np.r_[np.nan, (self.ema_1[1:] > self.ema_1[:-1]).astype(int)]
        slope_2 = np.r_[np.nan, (self.ema_2[1:] > self.ema_2[:-1]).astype(int)]
        slope_3 = np.r_[np.nan, (self.ema_3[1:] > self.ema_3[:-1]).astype(int)]

        def conds():
            return pd.DataFrame({
                '1 EMA slope>0 (green)': slope_1,
                '2 EMA slope>0 (blue)': slope_2,
                '3 EMA slope>0 (red)': slope_3
            })

        self.I(conds, name='EMA_Slope_Conditions')

    def next(self):
        # Current & previous EMAs
        e1, e1_prev = self.ema_1[-1], self.ema_1[-2]
        e2, e2_prev = self.ema_2[-1], self.ema_2[-2]
        e3, e3_prev = self.ema_3[-1], self.ema_3[-2]
        price = self.data.Close[-1]

        if any(np.isnan(x) for x in (e1, e1_prev, e2, e2_prev, e3, e3_prev)):
            return

        # ---- Time-based rules ----
        current_time = self.data.index[-1]

        # Always close at 15:59
        if isinstance(current_time, pd.Timestamp):
            if current_time.hour == 15 and current_time.minute == 59:
                if self.position:
                    self.position.close()
                return

        # ---- Slope conditions ----
        cond_e1_slope = e1 > e1_prev
        cond_e2_slope = e2 > e2_prev
        cond_e3_slope = e3 > e3_prev
        all_slopes_positive = cond_e1_slope and cond_e2_slope and cond_e3_slope

        # ---- Entry & exit logic ----
        # ❌ Don't enter before 9:30
        if all_slopes_positive:
            if not self.position:
                if current_time.hour > 9 or (current_time.hour == 9 and current_time.minute >= 59):
                    size = int(self.equity / price)
                    if size > 0:
                        self.buy(size=size)
        else:
            if self.position:
                self.position.close()
