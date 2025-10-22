import pandas as pd
import os
import pandas_market_calendars as mcal

from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime


MINUTES_PER_DAY = 390  # 정규장 1일당 분봉 수


def count_trading_days_nyse(start_str: str, end_str: str) -> int:
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=start_str, end_date=end_str)
    return len(schedule), schedule.index  # 두 번째 값은 실제 거래일 리스트



def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert("America/New_York")
    df.set_index("timestamp", inplace=True)
    df = df.between_time("09:30", "15:59")  # 정규장 필터링
    return df


def fill_missing_minutes(df: pd.DataFrame) -> pd.DataFrame:
    result = []

    for day in df.index.normalize().unique():
        full_range = pd.date_range(
            start=day + pd.Timedelta(hours=9, minutes=30),
            end=day + pd.Timedelta(hours=15, minutes=59),
            freq='T',
            tz='America/New_York'
        )

        day_data = df[df.index.normalize() == day]
        day_data = day_data.reindex(full_range)

        day_data["close"] = day_data["close"].fillna(method="ffill")
        day_data["open"] = day_data["open"].fillna(day_data["close"])
        day_data["high"] = day_data["high"].fillna(day_data["close"])
        day_data["low"] = day_data["low"].fillna(day_data["close"])
        day_data["volume"] = day_data["volume"].fillna(0)

        result.append(day_data)

    return pd.concat(result)


def finalize_for_backtesting(df: pd.DataFrame) -> pd.DataFrame:
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    df.rename(columns={
        "timestamp": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }, inplace=True)

    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    return df


def main(
    ticker: str,
    input_folder: str,
    output_folder: str,
    start_date_str: str = None,
    end_date_str: str = None
):
    file_name = f"{ticker.upper()}.csv"
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    if not os.path.exists(input_path):
        print(f"❌ 입력 파일이 없습니다: {input_path}")
        return

    print(f"📂 불러오는 파일: {input_path}")

    # 데이터 로딩 및 정리
    raw_df = load_and_clean_data(input_path)

    # ✅ 자동 날짜 추출
    inferred_start = raw_df.index.min().date().isoformat()
    inferred_end = raw_df.index.max().date().isoformat()

    print(f"⏱ 자동 감지된 날짜 범위: {inferred_start} ~ {inferred_end}")

    filled_df = fill_missing_minutes(raw_df)
    final_df = finalize_for_backtesting(filled_df)

    os.makedirs(output_folder, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ 정리된 데이터 저장 완료: {output_path}")

    # ✅ NYSE 기준 거래일 수 계산
    expected_days, valid_trading_days = count_trading_days_nyse(inferred_start, inferred_end)
    expected_rows = expected_days * MINUTES_PER_DAY
    actual_rows = len(final_df)

    print(f"\n📊 검증 결과")
    print(f" - 거래일 수: {expected_days}일")
    print(f" - 예상 분봉 수: {expected_rows}개")
    print(f" - 실제 분봉 수: {actual_rows}개")
    print(f" - ✅ 일치 여부: {'✅ 일치합니다!' if expected_rows == actual_rows else '❌ 불일치!'}")

    # 누락된 거래일 확인
    actual_days = pd.to_datetime(final_df["Datetime"]).dt.normalize().unique()
    missing_days = [pd.Timestamp(day).date() for day in valid_trading_days if pd.Timestamp(day).normalize() not in actual_days]

    if missing_days:
        print(f"❌ 누락된 거래일 수: {len(missing_days)}일")
        for day in missing_days:
            print(f"   - {day}")
    else:
        print("✅ 모든 거래일 데이터가 존재합니다.")






# ▶️ 실행 예시
if __name__ == "__main__":
    main(
        ticker="BBAI",
        input_folder=r"C:\Users\AWK\Desktop\USSystemTrade\Data\RawData",
        output_folder=r"C:\Users\AWK\Desktop\USSystemTrade\Data"
    )
