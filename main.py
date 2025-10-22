import pandas as pd
import os
from backtesting import Backtest
from strategies.stg_rsi import STA
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 원하는 종목코드를 여기서 선택
symbol = "BBAI"
cash = 1_000_000
commission = 0.00
months_ago = 13


# 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "Data", f"{symbol}.csv")
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Datetime'])
df.set_index('Date', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]


# 날짜 필터링
end_date = df.index.max().date()
start_date = (end_date - relativedelta(months=months_ago))
df = df.loc[start_date:end_date]

print(f"📊 백테스트 기간: {start_date} ~ {end_date}")

# 백테스트 실행
bt = Backtest(df, STA, cash=cash, commission=commission)
result = bt.run()
bt.plot()
print(result['_trades'])  # 🟢 반드시 확인!

