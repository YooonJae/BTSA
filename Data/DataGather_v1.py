from polygon import RESTClient
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
import os

API_KEY = "Y990EDZYLDzDApD8ze2fbLPDslIs0LCG"

def main(ticker: str, start_date_str: str, end_date_str: str, save_folder: str):
    client = RESTClient(API_KEY)

    # DATE RANGE SETUP
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    current_date = start_date
    all_data = []

    print(f"📊 [{ticker}] 데이터 수집 시작: {start_date_str} ~ {end_date_str}")

    while current_date <= end_date:
        from_date = current_date
        to_date = (from_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        if to_date > end_date:
            to_date = end_date

        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")
        print(f"\n📅 {from_str} ~ {to_str} 요청 중...")

        attempt = 0
        success = False

        while not success and attempt < 5:
            try:
                resp = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="minute",
                    from_=from_str,
                    to=to_str,
                    limit=50000
                )

                monthly_data = [{
                    "timestamp": pd.to_datetime(r.timestamp, unit="ms", utc=True),
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume
                } for r in resp]

                print(f"✅ 수신 완료: {len(monthly_data)}개 캔들")
                all_data.extend(monthly_data)
                success = True
                time.sleep(6)

            except Exception as e:
                attempt += 1
                print(f"⚠️ {attempt}회차 재시도... (오류: {e})")
                if attempt >= 5:
                    print(f"\n❌ 5회 재시도 실패 - 중단됨 (기간: {from_str} ~ {to_str})")
                    exit(1)
                time.sleep(20)

        current_date = (to_date + timedelta(days=1))

    # DataFrame GENERATOION
    df = pd.DataFrame(all_data)
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df = df[(df['timestamp'].dt.time >= pd.to_datetime("09:30").time()) &
            (df['timestamp'].dt.time <= pd.to_datetime("15:59").time())]

    df.sort_values("timestamp", inplace=True)

    # SAVE PATH
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{ticker.upper()}.csv")
    df.to_csv(save_path, index=False)

    print(f"\n✅ 저장 완료: {save_path}")



if __name__ == "__main__":
    main(
        ticker="BURU",
        start_date_str="2024-01-01",
        end_date_str="2025-10-24",
        save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "RawData", "SmallP_LargePV")
    )
