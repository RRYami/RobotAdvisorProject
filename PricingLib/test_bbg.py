import numpy as np
import QuantLib as ql

calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
bussiness_convention = ql.ModifiedFollowing
settlement_days = 0
day_count = ql.Actual360()

interest_rate = 0.0533
calc_date = ql.Date(10, 4, 2024)
yield_curve = ql.FlatForward(
    calc_date, interest_rate, day_count, ql.Compounded, ql.Continuous
)
ql.Settings.instance().evaluationDate = calc_date
option_maturity_date = ql.Date(24, 5, 2024)
strike = 101.75
spot = 101.53125  # futures price
volatility = 2.1990069128464625 / 100
flavor = ql.Option.Put

discount = yield_curve.discount(option_maturity_date)
strikepayoff = ql.PlainVanillaPayoff(flavor, strike)
T = yield_curve.dayCounter().yearFraction(calc_date, option_maturity_date)
stddev = volatility * np.sqrt(T)

black = ql.BlackCalculator(strikepayoff, spot, stddev, discount)
print("Option Price: " + str(black.value()))
print("Delta: " + str(black.delta(spot)))
print("Gamma: " + str(black.gamma(spot)))
print("Theta: " + str(black.theta(spot, T)))
print("Vega: " + str(black.vega(T)))
