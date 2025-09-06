import numpy as np
import scipy.stats as stats
from scipy.special import ndtr

from pricer.options.cl_option import Option


class BlackScholes(Option):
    """
    This is a class for Options contract for pricing European options on stocks/index without dividends.

    Attributes:
        spot          : int or float
        strike        : int or float
        rate          : float
        dte           : int or float [days to expiration in number of days]
        volatility    : float
        call_price     : int or float [default None]
        put_price      : int or float [default None]
    """

    def __init__(
        self,
        spot,
        strike,
        rate,
        time_to_maturity,
        volatility,
        dividend=None,
        call_price=None,
        put_price=None,
    ):
        super().__init__(spot, strike, rate, time_to_maturity, volatility)
        # Utility
        self.dte = time_to_maturity / 365
        self._a_ = self.volatility * self.dte**0.5
        self.call_price = call_price
        self.put_price = put_price
        self.dividend = dividend

        if self.volatility <= 0 or self.dte <= 0:
            raise ZeroDivisionError("The volatility and/or days to expiration cannot be below or equal to zero")

        if self.strike <= 0:
            raise ZeroDivisionError("The strike price cannot be below or equal to zero")
        else:
            if self.dividend is None:
                self._d1_ = (
                    np.log(self.spot / self.strike) + (self.rate + (self.volatility**2) / 2) * self.dte
                ) / self._a_
            else:
                self._d1_ = (
                    np.log(self.spot / self.strike) + (self.rate - self.dividend + (self.volatility**2) / 2) * self.dte
                ) / self._a_

        self._d2_ = self._d1_ - self._a_

        self._b_ = np.exp(-(self.rate * self.dte))
        """
        Contains all the attributes defined for the object itself. It maps the attribute name to its value.
        """
        for i in [
            "call_price",
            "put_price",
            "call_delta",
            "put_delta",
            "call_theta",
            "put_theta",
            "call_rho",
            "put_rho",
            "vega",
            "gamma",
            "implied_vol",
        ]:
            self.__dict__[i] = None

        [self.call_price, self.put_price] = self._price()
        [self.call_delta, self.put_delta] = self._delta()
        [self.call_theta, self.put_theta] = self._theta()
        [self.call_rho, self.put_rho] = self._rho()
        self.vega = self._vega()
        self.gamma = self._gamma()

    def __repr__(self) -> str:
        return f"""BlackScholes Class ->
    BlackScholes({self.spot}, {self.strike}, {self.rate}, {self.time_to_maturity}, {self.volatility}, {self.dividend}, {self.call_price}, {self.put_price})"""

    def __str__(self) -> str:
        return f"""BlackScholes Class ->
    BlackScholes({self.spot}, {self.strike}, {self.rate}, {self.time_to_maturity}, {self.volatility}, {self.dividend}, {self.call_price}, {self.put_price})"""

    # Option Price
    def _price(self):
        """Returns the option price: [Call price, Put price]"""

        if self.volatility == 0 or self.dte == 0:
            call = np.maximum(0.0, self.spot - self.strike)
            put = np.maximum(0.0, self.strike - self.spot)
        if self.dividend is None or self.dividend == 0:
            call = self.spot * ndtr(self._d1_) - self.strike * np.exp(-self.rate * self.dte) * ndtr(self._d2_)

            put = self.strike * np.e ** (-self.rate * self.dte) * ndtr(-self._d2_) - self.spot * ndtr(-self._d1_)
        else:
            time_value = np.exp(-self.rate * self.dte)
            forward_price = self.spot * np.exp((self.rate - self.dividend) * self.dte)
            call = time_value * (forward_price * ndtr(self._d1_) - self.strike * ndtr(self._d2_))
            put = time_value * (self.strike * ndtr(-self._d2_) - forward_price * ndtr(-self._d1_))
        return (call, put)

    # Option Delta
    def _delta(self):
        """Returns the option delta: [Call delta, Put delta]"""
        call = ndtr(self._d1_)
        put = -ndtr(-self._d1_)
        return (call, put)

    # Option Gamma
    def _gamma(self):
        """Returns the option gamma"""
        return stats.norm.pdf(self._d1_) / (self.spot * self._a_)

    # Option Vega
    def _vega(self):
        """Returns the option vega"""
        if self.volatility == 0 or self.dte == 0:
            return 0.0
        else:
            return self.spot * stats.norm.pdf(self._d1_) * self.dte**0.5 / 100

    # Option Theta
    def _theta(self):
        """Returns the option theta: [Call theta, Put theta]"""
        call = -self.spot * stats.norm.pdf(self._d1_) * self.volatility / (
            2 * self.dte**0.5
        ) - self.rate * self.strike * self._b_ * ndtr(self._d2_)

        put = -self.spot * stats.norm.pdf(self._d1_) * self.volatility / (
            2 * self.dte**0.5
        ) + self.rate * self.strike * self._b_ * ndtr(-self._d2_)
        return (call / 365, put / 365)

    # Option Rho
    def _rho(self):
        """Returns the option rho: [Call rho, Put rho]"""
        call = self.strike * self.dte * self._b_ * ndtr(self._d2_) / 100
        put = -self.strike * self.dte * self._b_ * ndtr(-self._d2_) / 100
        return (call, put)
