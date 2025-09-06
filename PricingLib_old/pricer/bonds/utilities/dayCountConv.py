from datetime import datetime, timedelta
from enum import Enum
from typing import TypeVar

Tnumber = TypeVar("Tnumber", int, float)


class FrequencyType(Enum):
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    CONTINUOUS = 99
    ZERO = -1


def frequency_to_period(frequency: FrequencyType) -> float:
    return {
        FrequencyType.ANNUAL: 1.0,
        FrequencyType.SEMI_ANNUAL: 2.0,
        FrequencyType.QUARTERLY: 4.0,
        FrequencyType.MONTHLY: 12.0,
        FrequencyType.CONTINUOUS: -1.0,
        FrequencyType.ZERO: 1.0,
    }.get(frequency, 1.0)


class DayCountTypes(Enum):
    ACT_360 = 1
    ACT_365 = 2
    ACT_ACT_ISDA = 3
    ACT_ACT_ICMA = 4
    THIRTY_360 = 5
    THIRTY_E_360_ISDA = 6


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def is_last_day_feb(dates: datetime) -> bool:
    return (dates.month == 2 and is_leap_year(dates.year) and dates.day == 29) or (
        dates.month == 2 and is_leap_year(dates.year) is False and dates.day == 28
    )


def thirty_360_factor(
    datetime_object_end: datetime, datetime_object_start: datetime
) -> float:
    return (
        360 * (datetime_object_end.year - datetime_object_start.year)
        + 30 * (datetime_object_end.month - datetime_object_start.month)
        + (datetime_object_end.day - datetime_object_start.day)
    )


def act_act_isda(start_date: str, end_date: str) -> tuple | Tnumber:
    datetime_object_start: datetime = datetime.strptime(start_date, "%Y%m%d")
    datetime_object_end: datetime = datetime.strptime(end_date, "%Y%m%d")
    time_delta: int = (datetime_object_end - datetime_object_start).days

    if datetime_object_start == datetime_object_end:
        return tuple([0, 0])

    denominator1: int = 366 if is_leap_year(datetime_object_start.year) else 365
    denominator2: int = 366 if is_leap_year(datetime_object_end.year) else 365

    if datetime_object_start.year == datetime_object_end.year:
        factor = time_delta / denominator1
        return factor, time_delta, denominator1

    else:
        days1: int = (
            datetime(datetime_object_start.year + 1, 1, 1) - datetime_object_start
        ).days
        days2: int = (
            datetime_object_end - datetime(datetime_object_end.year, 1, 1)
        ).days
        factor1 = days1 / denominator1
        factor2 = days2 / denominator2
        factor = factor1 + factor2
        return factor, days1, days2, denominator1, denominator2


def thirty_360(start_date: str, end_date: str) -> tuple | Tnumber:
    datetime_object_start: datetime = datetime.strptime(start_date, "%Y%m%d")
    datetime_object_end: datetime = datetime.strptime(end_date, "%Y%m%d")
    if datetime_object_start.day == 31:
        datetime_object_start = datetime_object_start.replace(day=30)
    if datetime_object_end.day == 31 and datetime_object_start.day == 30:
        datetime_object_end = datetime_object_end.replace(day=30)
    numerator = thirty_360_factor(datetime_object_end, datetime_object_start)
    denominator = 360
    factor = numerator / denominator
    return factor, numerator, denominator


def act_act_icma(
    start_date: str, end_date: str, settlement_date: str, frequency: FrequencyType
) -> tuple | Tnumber:
    datetime_object_start: datetime = datetime.strptime(start_date, "%Y%m%d")
    datetime_object_settlement: datetime = datetime.strptime(settlement_date, "%Y%m%d")
    datetime_object_end: datetime = datetime.strptime(end_date, "%Y%m%d")

    freq: float = frequency_to_period(frequency)
    if freq is None or settlement_date is None:
        raise ValueError("Frequency or Settlement Date is required for ACT_ACT_ICMA")
    numerator: timedelta = datetime_object_end - datetime_object_start
    denominator: timedelta = freq * (datetime_object_settlement - datetime_object_start)
    factor = numerator / denominator
    return factor, numerator.days, denominator.days


def thirty_e_360_isda(start_date: str, end_date: str) -> tuple | Tnumber:
    datetime_object_start: datetime = datetime.strptime(start_date, "%Y%m%d")
    datetime_object_end: datetime = datetime.strptime(end_date, "%Y%m%d")
    if datetime_object_start.day == 31:
        datetime_object_start = datetime_object_start.replace(day=30)
    start_last_day_feb = is_last_day_feb(datetime_object_start)
    if start_last_day_feb is True:
        datetime_object_start = datetime_object_start.replace(day=30)
    if datetime_object_end.day == 31:
        datetime_object_end = datetime_object_end.replace(day=30)
    end_last_day_feb = is_last_day_feb(datetime_object_end)
    if end_last_day_feb is True:
        datetime_object_end = datetime_object_end.replace(day=30)
    numerator = thirty_360_factor(datetime_object_end, datetime_object_start)
    denominator = 360
    factor = numerator / denominator
    return factor, numerator, denominator


class DayCountConv:
    """
    Class to calculate day count fraction and factor for different day count types:
    - ACT_360
    - ACT_365
    - ACT_ACT_ISDA
    - ACT_ACT_ICMA
    - THIRTY_360
    - THIRTY_E_360_ISDA
    """

    def __init__(self, dctype: DayCountTypes):
        if dctype not in DayCountTypes:
            raise ValueError("Day Count Type not supported")
        self.day_count_type = dctype

    def day_count_fraction(
        self,
        start_date: str,
        end_date: str,
        settlement_date: str,
        frequency: FrequencyType,
    ) -> tuple | Tnumber:
        datetime_object_start: datetime = datetime.strptime(start_date, "%Y%m%d")
        datetime_object_end: datetime = datetime.strptime(end_date, "%Y%m%d")
        time_delta: int = (datetime_object_end - datetime_object_start).days

        match self.day_count_type:
            case DayCountTypes.ACT_360:
                factor = time_delta / 360
                return factor, time_delta

            case DayCountTypes.ACT_365:
                factor = time_delta / 365
                return factor, time_delta

            case DayCountTypes.ACT_ACT_ISDA:
                return act_act_isda(start_date, end_date)

            case DayCountTypes.ACT_ACT_ICMA:
                return act_act_icma(start_date, end_date, settlement_date, frequency)

            case DayCountTypes.THIRTY_360:
                return thirty_360(start_date, end_date)

            case DayCountTypes.THIRTY_E_360_ISDA:
                return thirty_e_360_isda(start_date, end_date)


# print(
#     DayCountConv(DayCountTypes.THIRTY_E_360_ISDA).day_count_fraction(
#         "20210101", "20220101", "20210101", FrequencyType.ANNUAL
#     )
# )
