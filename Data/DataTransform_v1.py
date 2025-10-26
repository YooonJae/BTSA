import os
from datetime import datetime
import pandas as pd
import pandas_market_calendars as mcal

from pandas.tseries.holiday import USFederalHolidayCalendar  # (unused, but kept if you need it)

MINUTES_PER_DAY = 390  # 정규장 1일당 분봉 수


def count_trading_days_nyse(start_str: str, end_str: str) -> int:
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=start_str, end_date=end_str)
    return len(schedule), schedule.index  # 두 번째 값은 실제 거래일 리스트


def load_and_clean_data(path: str, input_tz: str = "UTC") -> pd.DataFrame:
    """
    CSV must contain 'timestamp' and OHLCV.
    - Converts from input_tz to America/New_York
    - Keeps only 09:30–15:59 (regular session)
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])

    if input_tz.upper() == "UTC":
        # timestamps are UTC; convert to NY
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    else:
        # timestamps are in some local tz (e.g., 'America/New_York')
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(input_tz).dt.tz_convert("America/New_York")

    df.set_index("timestamp", inplace=True)
    df = df.between_time("09:30", "15:59")
    return df


def fill_missing_minutes(df: pd.DataFrame, trading_days) -> pd.DataFrame:
    """
    trading_days: iterable of date-like (e.g., numpy.datetime64[D] or datetime.date)
    """
    result = []
    prior_close = None  # used when a whole day is missing

    tz = "America/New_York"

    for day in trading_days:
        # day as a date
        day_date = pd.Timestamp(day).date()

        # Full intraday 1-minute range in local NY time
        day_start = pd.Timestamp.combine(pd.Timestamp(day_date), pd.Timestamp("09:30").time()).tz_localize(tz)
        day_end   = pd.Timestamp.combine(pd.Timestamp(day_date), pd.Timestamp("15:59").time()).tz_localize(tz)
        full_range = pd.date_range(start=day_start, end=day_end, freq="1min")

        # Slice by date (avoids tz-aware vs tz-naive mismatch)
        day_data = df.loc[df.index.date == day_date].reindex(full_range)

        # If the whole day is empty, try to seed close from the last known close before 09:30
        if day_data["close"].isna().all():
            prev = df.loc[df.index < day_start, "close"].dropna()
            if not prev.empty:
                prior_close = prev.iloc[-1]

        # Fill 'close': seed with prior_close if present, then forward fill within the day
        if day_data["close"].isna().all() and prior_close is not None:
            day_data["close"] = prior_close

        day_data["close"] = day_data["close"].ffill()

        # Fill O/H/L/Volume
        day_data["open"]   = day_data["open"].fillna(day_data["close"])
        day_data["high"]   = day_data["high"].fillna(day_data["close"])
        day_data["low"]    = day_data["low"].fillna(day_data["close"])
        day_data["volume"] = day_data["volume"].fillna(0)

        # Update prior_close for the next day
        if not day_data["close"].isna().all():
            prior_close = day_data["close"].iloc[-1]

        result.append(day_data)

    return pd.concat(result)


def finalize_for_backtesting(df: pd.DataFrame) -> pd.DataFrame:
    """
    backtesting.py 호환 포맷으로 컬럼/타임존 정리
    """
    df = df.copy()
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    df.rename(
        columns={
            "timestamp": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

    # ✅ Clean up any remaining missing values
    df = df.fillna(method="ffill").fillna(method="bfill")  # forward/backward fill
    df = df.dropna(subset=["Open", "High", "Low", "Close"])  # ensure no NaN remains

    return df



def main(
    ticker: str,
    input_folder: str,
    output_folder: str,
    start_date_str: str = None,
    end_date_str: str = None,
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

    # ✅ 자동 날짜 추출 (CSV 내 실제 존재 구간)
    inferred_start = raw_df.index.min().date().isoformat()
    inferred_end = raw_df.index.max().date().isoformat()

    # 사용자가 명시한 기간이 있으면 그걸 우선
    start_for_calendar = start_date_str or inferred_start
    end_for_calendar = end_date_str or inferred_end

    print(f"⏱ 자동 감지된 날짜 범위: {inferred_start} ~ {inferred_end}")

    # ✅ 예상 거래일(캘린더) 생성
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=start_for_calendar, end_date=end_for_calendar)
    trading_days = schedule.index.date  # date 객체 배열

    # (선택) 원본 CSV에서 통째로 빠진 거래일 진단
    expected_days_idx = pd.Index(pd.to_datetime(schedule.index).normalize().date)
    present_days_idx = pd.Index(raw_df.index.normalize().date)
    truly_missing = expected_days_idx.difference(present_days_idx)
    if len(truly_missing) > 0:
        print("⚠️ 원본 CSV에서 통째로 빠진 거래일:")
        for d in truly_missing:
            print(f"   - {d}")

    # ✅ 분단위 누락 채우기(거래일 기준)
    filled_df = fill_missing_minutes(raw_df, trading_days)

    # ✅ backtesting.py 형식으로 최종 정리
    final_df = finalize_for_backtesting(filled_df)

    # 저장
    os.makedirs(output_folder, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ 정리된 데이터 저장 완료: {output_path}")

    # ✅ NYSE 기준 거래일 수 및 분봉 수 검증
    expected_days, valid_trading_days = count_trading_days_nyse(start_for_calendar, end_for_calendar)
    expected_rows = expected_days * MINUTES_PER_DAY
    actual_rows = len(final_df)

    print(f"\n📊 검증 결과")
    print(f" - 거래일 수: {expected_days}일")
    print(f" - 예상 분봉 수: {expected_rows}개")
    print(f" - 실제 분봉 수: {actual_rows}개")
    print(f" - ✅ 일치 여부: {'✅ 일치합니다!' if expected_rows == actual_rows else '❌ 불일치!'}")

    # 누락된 거래일(최종 결과 기준) 확인
    actual_days = pd.to_datetime(final_df["Datetime"]).dt.normalize().unique()
    missing_days = [
        pd.Timestamp(day).date()
        for day in valid_trading_days
        if pd.Timestamp(day).normalize() not in actual_days
    ]

    if missing_days:
        print(f"❌ 누락된 거래일 수: {len(missing_days)}일")
        for day in missing_days:
            print(f"   - {day}")
    else:
        print("✅ 모든 거래일 데이터가 존재합니다.")


# ▶️ 실행 예시
if __name__ == "__main__":
    main(
        ticker="BURU",
        input_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "RawData", "SmallP_LargePV"),
        output_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "TransData", "SmallP_LargePV"),
    )
