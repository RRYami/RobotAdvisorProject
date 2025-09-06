import time
import warnings

import numpy as np
import pandas as pd
import QuantLib as ql
import sqlalchemy as db
from scipy import interpolate

warnings.filterwarnings("ignore")

start = time.perf_counter()


def db_get_associated_yield_curve(security_id, position_date, model_date):
    engine = db.create_engine(
        "mssql+pyodbc:///?odbc_connect=Driver={SQL Server};Server=192.168.11.102;Database=IDPM;Trusted_Connection=True;"
    )

    conn = engine.connect()
    cursor = conn.connection.cursor()

    select_users = f"""EXEC [Helper].[YieldCurve_GetSecurityMapping]
            @SecurityID = {security_id},
            @PositionDate = '{position_date}',
            @ModelDate = '{model_date}'"""

    result_proc = cursor.execute(select_users)
    result = []
    columns_fin = []
    result = [cursor.fetchall()]
    columns = [column[0] for column in cursor.description]

    if result_proc.nextset():
        result1 = [cursor.fetchall()]
        columns2 = [column[0] for column in cursor.description]

    if result_proc.nextset():
        result2 = [cursor.fetchall()]
        columns3 = [column[0] for column in cursor.description]

    columns_fin.append(columns)
    columns_fin.append(columns2)
    columns_fin.append(columns3)
    result.append(result1)
    result.append(result2)

    return result[1][0][0][0]


def db_get_bond_information(security_id):
    engine = db.create_engine(
        "mssql+pyodbc:///?odbc_connect=Driver={SQL Server};Server=192.168.11.102;Database=IDPM;Trusted_Connection=True;"
    )
    query = f"""SELECT SecurityID
    , s.BondCoupon
    , s.ParValue
    , s.IssueDate
    , s.BondRedemptionDate
    , s.DayCountConventionID
    , s.BondCouponFrequency
    , s.SettlementDays
    , isnull(s.DayCountConventionID,3) AS DayCountConventionID
    , isnull(dcc.Code, 'Actual/360') As DayCountConvention
    FROM tblSecurity s
    LEFT JOIN tblDefnDayCountConvention dcc
        ON dcc.ID = s.DayCountConventionID
    WHERE SecurityID = {security_id}"""

    data = pd.read_sql(
        db.text(query),
        engine.connect(),
    )
    return data


def db_get_yield_curve_data(yield_curve_id, market_date):
    engine = db.create_engine(
        "mssql+pyodbc:///?odbc_connect=Driver={SQL Server};Server=192.168.11.102;Database=IDPM;Trusted_Connection=True;"
    )
    query = f"""select * from YieldCurvePoint yp
    join YieldCurveData yd
        on yd.YieldCurvePointID = yp.YieldCurvePointID
    where yp.YieldCurveID = {yield_curve_id}
    and yd.MarketDate = '{market_date}'"""

    data = pd.read_sql(
        db.text(query),
        engine.connect(),
    )
    return data


def db_get_schedule(security_id):
    engine = db.create_engine(
        "mssql+pyodbc:///?odbc_connect=Driver={SQL Server};Server=192.168.11.102;Database=IDPM;Trusted_Connection=True;"
    )
    query = f"""SELECT SecurityID, ScheduleValue, StartDate, EndDate FROM tblSecuritySchedule WHERE SecurityID = {security_id}"""

    data = pd.read_sql(
        db.text(query),
        engine.connect(),
    )
    return data


def get_forward_rate(data: pd.DataFrame):
    maturity_arr = np.array(data["DaysToMaturity"] / 366)
    yield_arr = np.array(data["Yield"])
    forward_arr = np.zeros(len(yield_arr))
    forward_arr[0] = yield_arr[0]
    for i in range(1, len(yield_arr)):
        forward_arr[i] = (
            yield_arr[i] * maturity_arr[i] - yield_arr[i - 1] * maturity_arr[i - 1]
        ) / (maturity_arr[i] - maturity_arr[i - 1])
    return (forward_arr / 100, maturity_arr)


def interpolate_yield_curve(rate: np.ndarray, tenor: np.ndarray):
    new_tenor = np.linspace(0, np.ceil(tenor[-1]), len(tenor))
    dxdy = np.gradient(rate, tenor)
    forward_coef = interpolate.CubicHermiteSpline(tenor, rate, dxdy)
    forward_fit = forward_coef(new_tenor)
    return (forward_fit, new_tenor)


# Load bond information
yield_curve_id = db_get_associated_yield_curve(820175, "20240110", "20240110")
print("yield_curve_id:", yield_curve_id)
bond_information = db_get_bond_information(820175)
schedule_raw = db_get_schedule(820175)
yield_curve_data = db_get_yield_curve_data(yield_curve_id, "20230908")
forward_rate, tenor = get_forward_rate(yield_curve_data)
interpolated_yield, new_tenor = interpolate_yield_curve(forward_rate, tenor)

# tranform the date to QuantLib format
matruity_dates_ql = [
    ql.Date(yield_curve_data["MarketDate"][0], "%Y-%m-%d") + int(d)
    for d in yield_curve_data["DaysToMaturity"]
]
maturity_array = np.array(matruity_dates_ql)


class Bond:
    def __init__(self) -> None:
        self.security_id = bond_information["SecurityID"][0]
        self.bond_coupon = bond_information["BondCoupon"][0] / 100
        self.par_value = bond_information["ParValue"][0]
        self.issue_date = ql.Date(
            bond_information["IssueDate"][0].strftime("%Y-%m-%d"), "%Y-%m-%d"
        )
        self.redemption_date = ql.Date(
            bond_information["BondRedemptionDate"][0].strftime("%Y-%m-%d"), "%Y-%m-%d"
        )
        self.coupon_frequency = bond_information["BondCouponFrequency"][0]
        self.day_count_convention = bond_information["DayCountConvention"][0]
        self.settlement_days = bond_information["SettlementDays"][0]
        self.accrual_daycount = self.day_count_convention


class BondPricing(Bond):
    """
    {
    'SimpleDayCounter': ql.SimpleDayCounter(),
    'Thirty360': ql.Thirty360(ql.Thirty360.ISDA),
    'Actual360': ql.Actual360(),
    'Actual365Fixed': ql.Actual365Fixed(),
    'Actual365Fixed(Canadian)': ql.Actual365Fixed(ql.Actual365Fixed.Canadian),
    'Actual365FixedNoLeap': ql.Actual365Fixed(ql.Actual365Fixed.NoLeap),
    'ActualActual': ql.ActualActual(ql.ActualActual.ISDA),
    'Business252': ql.Business252()
    }
    {
    Argentina : [`Merval`]
    Brazil : [`Exchange`, `Settlement`]
    Canada : [`Settlement`, `TSX`]
    China : [`IB`, `SSE`]
    CzechRepublic : [`PSE`]
    France : [`Exchange`, `Settlement`]
    Germany : [`Eurex`, `FrankfurtStockExchange`, `Settlement`, `Xetra`]
    HongKong : [`HKEx`]
    Iceland : [`ICEX`]
    India : [`NSE`]
    Indonesia : [`BEJ`, `JSX`]
    Israel : [`Settlement`, `TASE`]
    Italy : [`Exchange`, `Settlement`]
    Mexico : [`BMV`]
    Russia : [`MOEX`, `Settlement`]
    SaudiArabia : [`Tadawul`]
    Singapore : [`SGX`]
    Slovakia : [`BSSE`]
    SouthKorea : [`KRX`, `Settlement`]
    Taiwan : [`TSEC`]
    Ukraine : [`USE`]
    UnitedKingdom : [`Exchange`, `Metals`, `Settlement`]
    UnitedStates : [`FederalReserve`, `GovernmentBond`, `LiborImpact`, `NERC`, `NYSE`, `Settlement`]
    }
    """

    def __init__(
        self,
        calculation_date,
        day_count_convention,
        calendar,
        date_generation=1,
    ) -> None:

        super().__init__()
        self.yield_curve_id = yield_curve_id
        self.forward_rate = interpolated_yield
        self.maturity_dates = maturity_array
        self.calculation_date = ql.Date(calculation_date, "%Y%m%d")
        self.day_count_convention = day_count_convention
        self.calendar = calendar
        self.acrrual_convention = ql.Unadjusted
        self.date_generation = date_generation

        ql.Settings.instance().evaluationDate = self.calculation_date

    # Forward curve builder
    def helper_forward_curve(self):
        ts = ql.ForwardCurve(
            self.maturity_dates, self.forward_rate, self.day_count_convention
        )
        ts_handle = ql.YieldTermStructureHandle(ts)
        return ts_handle

    # Callability schedule
    def helper_schedule_builer(self):
        callability_schedule = ql.CallabilitySchedule()
        call_price_t = list(schedule_raw["ScheduleValue"])
        call_date_t = []
        for i in schedule_raw["EndDate"]:
            call_date_t.append(ql.Date(i.day, i.month, i.year))

        for price, date in zip(call_price_t, call_date_t):
            callability_schedule.append(
                ql.Callability(
                    ql.BondPrice(price, ql.BondPrice.Clean), ql.Callability.Call, date
                )
            )
        return callability_schedule

    # Cash flow schedule
    def cf_bond_schedule(self):
        match self.coupon_frequency:
            case 1:
                self.coupon_frequency = ql.Annual
            case 2:
                self.coupon_frequency = ql.Semiannual
            case 4:
                self.coupon_frequency = ql.Quarterly
            case 12:
                self.coupon_frequency = ql.Monthly
            case _:
                raise ValueError("Invalid coupon frequency")

        self.date_generation = (
            ql.DateGeneration.Forward
            if self.date_generation
            else ql.DateGeneration.Backward
        )
        schedule = ql.Schedule(
            self.issue_date,
            self.redemption_date,
            ql.Period(self.coupon_frequency),
            self.calendar,
            self.acrrual_convention,
            self.acrrual_convention,
            self.date_generation,
            False,
        )
        return schedule

    def bond_helper(self):
        schedule = self.cf_bond_schedule()
        call_schedule = self.helper_schedule_builer()
        b = ql.CallableFixedRateBond(
            int(self.settlement_days),
            self.par_value,
            schedule,
            [self.bond_coupon],
            self.day_count_convention,
            ql.Following,
            self.par_value,
            self.issue_date,
            call_schedule,
        )
        return b

    # Bond pricing with Hull White model fixed parameters a and s
    def bond_pricing(self, a, s, grid_points):
        ts_handle = self.helper_forward_curve()
        bond = self.bond_helper()
        model = ql.HullWhite(ts_handle, a, s)
        engine = ql.TreeCallableFixedRateBondEngine(model, grid_points)
        bond.setPricingEngine(engine)
        return bond


x = BondPricing(
    "202401109", ql.Actual360(), ql.UnitedStates(ql.UnitedStates.GovernmentBond)
)
print(x.bond_pricing(0.101, 0.0225, 400).cleanPrice())
end = time.perf_counter()
print(f"Finished in {end-start} seconds")
