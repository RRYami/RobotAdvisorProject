from datetime import datetime

import pandas as pd
from calendar_proj import CalendarHandler, CalendarType, add_days
from frequency import FrequencyTypes  # annual_frequency


class Schedule:
    """
    Class to manage schedule of coupon payments. Set of dates generated according to ISDA standard rule.
    which starts on the next date after the effective date and runs up to a termination date.
    Dates are adjusted to a provided calendar.
    """

    def __init__(
        self,
        effective_date: datetime,
        termination_date: datetime,
        frequency_type: FrequencyTypes = FrequencyTypes.ANNUAL,
        calendar_type: CalendarType = CalendarType.UNITED_STATES,
    ):
        if effective_date >= termination_date:
            raise ValueError("Effective date must be before termination date")

        self._effective_date = effective_date
        self._termination_date: datetime = termination_date

        self._freq_type = frequency_type
        self._calendar_type = calendar_type

        self._generate()

    def _generate(self):
        """Generate schedule of dates according to specified date generation
        rules and also adjust these dates for holidays according to the
        specified business day convention and the specified calendar."""

        calendar = CalendarHandler(self._calendar_type)
        # frequency = annual_frequency(self._freq_type)

        dates = pd.date_range(
            start=self._effective_date, end=self._termination_date, freq="M"
        )

        schedule = []
        for i in dates:
            if calendar.is_holiday(i) is False and calendar.is_business_day(i) is True:
                schedule.append(i)
            elif (
                calendar.is_holiday(i) is False and calendar.is_business_day(i) is False
            ):
                schedule.append(add_days(i, 1))
            elif calendar.is_holiday(i) is True and calendar.is_business_day(i) is True:
                schedule.append(add_days(i, 1))
            elif (
                calendar.is_holiday(i) is True and calendar.is_business_day(i) is False
            ):
                schedule.append(add_days(i, 2))
        return schedule


print(
    Schedule(
        datetime(2021, 1, 1),
        datetime(2024, 12, 31),
    )._generate()
)
