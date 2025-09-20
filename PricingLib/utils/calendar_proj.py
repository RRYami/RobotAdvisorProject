import calendar as cal
from datetime import datetime, timedelta
from enum import Enum

import holidays


class CalendarType(Enum):
    NONE = 1
    WEEKEND = 2
    FRANCE = 3
    GERMANY = 4
    UNITED_KINGDOM = 5
    UNITED_STATES = 6
    JAPAN = 7


class BusinessDayAdjustment(Enum):
    UNADJUSTED = 1
    FOLLOWING = 2
    PRECEDING = 3
    MODIFIED_FOLLOWING = 4
    MODIFIED_PRECEDING = 5


class DateGenerationRule(Enum):
    BACKWARD = 1
    FORWARD = 2


def add_days(date: datetime, days: int) -> datetime:
    """
    Add days to a date.
    :param date: Date
    :param days: Number of days to add
    :return: Date with days added
    """
    if isinstance(date, datetime):
        final_date = date + timedelta(days=days)
    return final_date


def is_end_of_month(date: datetime) -> bool:
    """
    Check if date is the end of the month
    :param date: Date
    :return: True if date is the end of the month, False otherwise
    """
    if isinstance(date, datetime):
        return True if date.day == cal.monthrange(date.year, date.month)[1] else False


def end_of_month(date: datetime) -> datetime:
    """
    Return the end of the month for a given date
    :param date: Date
    :return: End of month for the given date
    """
    if isinstance(date, datetime):
        return date + timedelta(
            days=cal.monthrange(date.year, date.month)[1] - date.day
        )


class CalendarHandler:
    """
    Class to manage payment day as holidays according to country specific calendar
    """

    def __init__(self, calendar_type: CalendarType):
        if calendar_type not in CalendarType:
            raise ValueError("Calendar Type not supported")
        self.calendar_type = calendar_type

    def is_holiday(self, date: datetime | str) -> bool | str:
        """
        Check if date is a holiday
        :param self: CalendarType
        :param date: Date
        :return: True if date is a holiday, False otherwise
        """
        if isinstance(date, str):
            new_date = datetime.strptime(date, "%Y%m%d")
        else:
            if isinstance(date, datetime):
                new_date = date

        match self.calendar_type:
            case CalendarType.FRANCE:
                return True if new_date in holidays.country_holidays("FR") else False
            case CalendarType.GERMANY:
                return True if new_date in holidays.country_holidays("DE") else False
            case CalendarType.UNITED_KINGDOM:
                return True if new_date in holidays.country_holidays("UK") else False
            case CalendarType.UNITED_STATES:
                return True if new_date in holidays.country_holidays("US") else False
            case CalendarType.JAPAN:
                return True if new_date in holidays.country_holidays("JP") else False
            case _:
                return "No holiday for this calendar type"

    def is_business_day(self, date: datetime | str) -> bool | str | datetime:
        """
        Check if date is a business day
        :param date: Date
        :return: True if date is a business day, False otherwise
        """
        if isinstance(date, str):
            new_date = datetime.strptime(date, "%Y%m%d")
        else:
            if isinstance(date, datetime):
                new_date = date

        match self.calendar_type:
            case CalendarType.FRANCE:
                return (
                    True
                    if cal.weekday(new_date.year, new_date.month, new_date.day) < 5
                    and date not in holidays.country_holidays("FR")
                    else False
                )
            case CalendarType.GERMANY:
                return (
                    True
                    if cal.weekday(new_date.year, new_date.month, new_date.day) < 5
                    and date not in holidays.country_holidays("DE")
                    else False
                )
            case CalendarType.UNITED_KINGDOM:
                return (
                    True
                    if cal.weekday(new_date.year, new_date.month, new_date.day) < 5
                    and date not in holidays.country_holidays("UK")
                    else False
                )
            case CalendarType.UNITED_STATES:
                return (
                    True
                    if cal.weekday(new_date.year, new_date.month, new_date.day) < 5
                    and date not in holidays.country_holidays("US")
                    else False
                )
            case CalendarType.JAPAN:
                return (
                    True
                    if cal.weekday(new_date.year, new_date.month, new_date.day) < 5
                    and date not in holidays.country_holidays("JP")
                    else False
                )
            case _:
                return "No holiday for this calendar type"

    def adjust(
        self,
        date: datetime,
        business_day_adjustment: BusinessDayAdjustment,
    ) -> datetime:
        """
        Adjust the payment date if it falls on a holiday according to business day adjustment type
        :param date: Date
        :param business_day_adjustment: Business Day Adjustment type
        :return:  Datetime | New date adjusted according to business day adjustment
        """

        if type(business_day_adjustment) is not BusinessDayAdjustment:
            raise ValueError("Business Day Adjustment Type not supported")

        # Convert date to datetime if it's a string
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y%m%d")

        assert isinstance(date, datetime)

        match business_day_adjustment:
            case BusinessDayAdjustment.UNADJUSTED:
                pass
            case BusinessDayAdjustment.FOLLOWING:
                while self.is_business_day(date) is False:
                    date = add_days(date, 1)
                while self.is_holiday(date) is True:
                    date = add_days(date, 1)
            case BusinessDayAdjustment.PRECEDING:
                while self.is_business_day(date) is False:
                    date = add_days(date, -1)
                while self.is_holiday(date) is True:
                    date = add_days(date, -1)
            case BusinessDayAdjustment.MODIFIED_FOLLOWING:
                adjusted_date = date
                while (
                    self.is_holiday(adjusted_date) is True
                    or self.is_business_day(adjusted_date) is False
                ):
                    adjusted_date = add_days(adjusted_date, 1)

                if adjusted_date.month != date.month:
                    adjusted_date = add_days(date, -1)
                else:
                    pass
                date = adjusted_date
            case BusinessDayAdjustment.MODIFIED_PRECEDING:
                adjusted_date = date
                while (
                    self.is_holiday(adjusted_date) is True
                    or self.is_business_day(adjusted_date) is False
                ):
                    adjusted_date = add_days(adjusted_date, -1)

                if adjusted_date.month != date.month:
                    adjusted_date = add_days(date, 1)
                else:
                    pass
                date = adjusted_date
        return date


# print(CalendarHandler(CalendarType.FRANCE).is_holiday("20231226"))
# print(
#     CalendarHandler(CalendarType.UNITED_KINGDOM).adjust(
#         datetime(2024, 1, 1), BusinessDayAdjustment.MODIFIED_PRECEDING
#     )
# )
# print(CalendarHandler(CalendarType.UNITED_KINGDOM).adjust(datetime(2023, 12, 26), BusinessDayAdjustment.PRECEDING))
