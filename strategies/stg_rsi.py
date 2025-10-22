# strategies/stg_rsi.py
import numpy as np
import pandas as pd
from backtesting import Strategy

def _sma(series, n):
    return pd.Series(series).rolling(n, min_periods=n).mean().values

class STA(Strategy):
    """
    1분봉 기준:
      - 1일 MA: 390
      - (기존) 단기 MA: n_short
      - 30일 MA: 11,700
    조건:
      - 모든 MA 기울기 > 0
      - 종가가 모든 MA 위
      -> 진입 시 '한 번에' 풀매수, 해제 시 '한 번에' 전량 청산
    """
    n_day   = 390*5      # 추가: 1일선
    n_short = 390*10    # 기존 값 유지
    n_long  = 11700

    def init(self):
        close = self.data.Close
        self.ma_day   = self.I(_sma, close, self.n_day)    # 추가
        self.ma_short = self.I(_sma, close, self.n_short)
        self.ma_long  = self.I(_sma, close, self.n_long)

    def next(self):
        # ---- 데이터 준비/검사 ----
        if len(self.ma_day) < 2 or len(self.ma_short) < 2 or len(self.ma_long) < 2:
            return

        md, md_prev = self.ma_day[-1], self.ma_day[-2]
        ms, ms_prev = self.ma_short[-1], self.ma_short[-2]
        ml, ml_prev = self.ma_long[-1], self.ma_long[-2]

        if any(np.isnan(x) for x in (md, md_prev, ms, ms_prev, ml, ml_prev)):
            return

        price = self.data.Close[-1]
        long_signal = (
            (md > md_prev) and (ms > ms_prev) and (ml > ml_prev) and
            (price > md) and (price > ms) and (price > ml)
        )

        # ---- 진입/청산: 단발성 ----
        if long_signal:
            if not self.position:
                # 포지션 없을 때 equity == 현금
                size = int(self.equity / price)
                if size > 0:
                    self.buy(size=size)
        else:
            if self.position:
                self.position.close()
