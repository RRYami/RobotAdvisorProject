import time
import warnings

import numpy as np
import pandas as pd
import QuantLib as ql
import sqlalchemy as db
from scipy import interpolate

warnings.filterwarnings("ignore")

start = time.perf_counter()


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


class ModelCalibrator:
    def __init__(self, endCriteria):
        self.endCriteria = endCriteria
        self.helpers = []

    def AddCalibrationHelper(self, helper):
        self.helpers.append(helper)

    def Calibrate(self, model, engine, curve, fixedParameters):
        # assign pricing engine to all calibration helpers
        for i in range(len(self.helpers)):
            self.helpers[i].setPricingEngine(engine)
        method = ql.LevenbergMarquardt()
        if len(fixedParameters) == 0:
            model.calibrate(self.helpers, method, self.endCriteria)
        else:
            model.calibrate(
                self.helpers,
                method,
                self.endCriteria,
                ql.NoConstraint(),
                [],
                fixedParameters,
            )


class HullWhiteCalibration(ModelCalibrator):
    def __init__(
        self,
        tradeDate: ql.Date,
        maturity_array: np.ndarray,
        forward_rate: np.ndarray,
        swaption_iv_list: np.ndarray,
    ) -> None:
        super().__init__(endCriteria=ql.EndCriteria(10000, 1000, 1e-8, 1e-8, 1e-8))
        self.maturity_array = maturity_array
        self.forward_rate = forward_rate
        self.tradeDate = tradeDate
        self.swaption_iv_list = swaption_iv_list
        self.calendar = ql.TARGET()
        self.dayCounter = ql.Actual360()
        ql.Settings.instance().evaluationDate = self.tradeDate

    def curve_builder(self):
        ts = ql.ForwardCurve(self.maturity_array, self.forward_rate, ql.Actual360())
        ts.enableExtrapolation()
        ts_handle = ql.YieldTermStructureHandle(ts)
        return ts_handle

    def swaption_helper(
        self,
    ):
        calibrator = ModelCalibrator(self.endCriteria)
        for i in range(len(self.swaption_iv_list)):
            t = i + 1
            tenor = len(self.swaption_iv_list) - i
            helper = ql.SwaptionHelper(
                ql.Period(t, ql.Years),
                ql.Period(tenor, ql.Years),
                ql.QuoteHandle(ql.SimpleQuote(self.swaption_iv_list[i])),
                ql.USDLibor(
                    ql.Period(3, ql.Months),
                    self.curve_builder(),
                ),
                ql.Period(1, ql.Years),
                self.dayCounter,
                self.dayCounter,
                self.curve_builder(),
            )
            calibrator.AddCalibrationHelper(helper)
        return calibrator

    def get_params(self):
        model = ql.HullWhite(self.curve_builder())
        engine = ql.JamshidianSwaptionEngine(model)
        calibrator = self.swaption_helper()
        curve = self.curve_builder()
        calibrator.Calibrate(model, engine, curve, [])
        return model.params()


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


def CreateSwaptionVolatilityList(vol: np.ndarray):
    vol_list = []
    for item in vol:
        vol_list.append(item)
    return vol_list


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

# Swaption implied volatility list
swaption_iv = pd.read_csv(
    r"C:\Users\rrenard\OneDrive - Arkus\Desktop\ModelDev\HullWhite\swapIV.csv",
    index_col=0,
)
swaption_iv_arr = np.array(swaption_iv)
iv_diag = swaption_iv_arr.diagonal() / 100

y = HullWhiteCalibration(
    ql.Date(8, ql.September, 2023), maturity_array, forward_rate, iv_diag
)
params = y.get_params()

x = BondPricing(
    "202401109", ql.Actual360(), ql.UnitedStates(ql.UnitedStates.GovernmentBond)
)

print(x.bond_pricing(params[0], params[1], 400).cleanPrice())

end = time.perf_counter()
print(f"Finished in {end-start} seconds")
