import unittest

from calendar_proj import CalendarHandler, CalendarType


class TestCalendar(unittest.TestCase):
    def test_is_holiday(self):
        calendar = CalendarHandler(CalendarType.FRANCE)
        date = "20211225"
        self.assertTrue(calendar.is_holiday(date))

    def test_is_business_day(self):
        calendar = CalendarHandler(CalendarType.FRANCE)
        date = "20211225"
        self.assertFalse(calendar.is_business_day(date))
