# import time
import numpy as np
import scipy.stats as stats
from scipy.special import ndtr

from pricer.options.cl_option import Option

_ITERATION_MAX_ERROR = 0.0001
_DELTA_S = 0.001
_DELTA_V = 0.00001
_DELTA_T = 1 / 365
_DELTA_R = 0.0001


class BaroneAdesiWhaley(Option):
    """
    This is a class for Options contract for pricing American options on stocks/index without dividends.
    """

    def __init__(self, spot, strike, rate, time_to_maturity, volatility, cost_of_carry=0.0):
        super().__init__(spot, strike, rate, time_to_maturity, volatility)
        self.cost_of_carry = cost_of_carry
        self.dte = time_to_maturity / 365

        if self.strike == 0:
            raise ZeroDivisionError("The strike price cannot be zero")

    def _Kc(self):

        N = 2 * self.cost_of_carry / self.volatility**2
        m = 2 * self.rate / self.volatility**2
        q2u = (-1 * (N - 1) + np.sqrt((N - 1) ** 2 + 4 * m)) / 2
        su = self.strike / (1 - (1 / q2u))
        h2 = (
            -1
            * (self.cost_of_carry * self.dte + 2 * self.volatility * np.sqrt(self.dte))
            * (self.strike / (su - self.strike))
        )

        Si = self.strike + (su - self.strike) * (1 - np.exp(h2))

        k = 2 * self.rate / (self.volatility**2 * (1 - np.exp(-1 * self.rate * self.dte)))
        d1 = (np.log(Si / self.strike) + (self.cost_of_carry + self.volatility**2 / 2) * self.dte) / (
            self.volatility * np.sqrt(self.dte)
        )
        Q2 = (-1 * (N - 1) + np.sqrt((N - 1) ** 2 + 4 * k)) / 2
        LHS = Si - self.strike
        RHS = (
            self.european_price_op(
                "Call",
                Si,
                self.strike,
                self.dte,
                self.rate,
                self.volatility,
                self.cost_of_carry,
            )
            + (1 - np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(d1)) * Si / Q2
        )
        bi = (
            np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(d1) * (1 - 1 / Q2)
            + (
                1
                - np.exp((self.cost_of_carry - self.rate) * self.dte)
                * stats.norm.pdf(d1)
                / (self.volatility * np.sqrt(self.dte))
            )
            / Q2
        )

        E = _ITERATION_MAX_ERROR

        while np.abs(LHS - RHS) / self.strike > E:
            Si = (self.strike + RHS - bi * Si) / (1 - bi)
            d1 = (np.log(Si / self.strike) + (self.cost_of_carry + self.volatility**2 / 2) * self.dte) / (
                self.volatility * np.sqrt(self.dte)
            )
            LHS = Si - self.strike
            RHS = (
                self.european_price_op(
                    "Call",
                    Si,
                    self.strike,
                    self.dte,
                    self.rate,
                    self.volatility,
                    self.cost_of_carry,
                )
                + (1 - np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(d1)) * Si / Q2
            )
            bi = (
                np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(d1) * (1 - 1 / Q2)
                + (
                    1
                    - np.exp((self.cost_of_carry - self.rate) * self.dte)
                    * ndtr(d1)
                    / (self.volatility * np.sqrt(self.dte))
                )
                / Q2
            )
        return Si

    def _Kp(self):

        N = 2 * self.cost_of_carry / self.volatility**2
        m = 2 * self.rate / self.volatility**2
        q1u = (-1 * (N - 1) - np.sqrt((N - 1) ** 2 + 4 * m)) / 2
        su = self.strike / (1 - 1 / q1u)
        h1 = (
            (self.cost_of_carry * self.dte - 2 * self.volatility * np.sqrt(self.dte)) * self.strike / (self.strike - su)
        )
        Si = su + (self.strike - su) * np.exp(h1)

        k = 2 * self.rate / (self.volatility**2 * (1 - np.exp(-1 * self.rate * self.dte)))
        d1 = (np.log(Si / self.strike) + (self.cost_of_carry + self.volatility**2 / 2) * self.dte) / (
            self.volatility * np.sqrt(self.dte)
        )
        Q1 = (-1 * (N - 1) - np.sqrt((N - 1) ** 2 + 4 * k)) / 2
        LHS = self.strike - Si
        RHS = (
            self.european_price_op("Put", Si, self.strike, self.dte, self.rate, self.volatility, self.cost_of_carry)
            - (1 - np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(-d1)) * Si / Q1
        )
        bi = (
            -1 * np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(-d1) * (1 - 1 / Q1)
            - (
                1
                + np.exp((self.cost_of_carry - self.rate) * self.dte)
                * stats.norm.pdf(-d1)
                / (self.volatility * np.sqrt(self.dte))
            )
            / Q1
        )

        E = _ITERATION_MAX_ERROR

        while np.abs(LHS - RHS) / self.strike > E:
            Si = (self.strike - RHS + bi * Si) / (1 + bi)
            d1 = (np.log(Si / self.strike) + (self.cost_of_carry + self.volatility**2 / 2) * self.dte) / (
                self.volatility * np.sqrt(self.dte)
            )
            LHS = self.strike - Si
            RHS = (
                self.european_price_op("Put", Si, self.strike, self.dte, self.rate, self.volatility, self.cost_of_carry)
                - (1 - np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(-d1)) * Si / Q1
            )
            bi = (
                -np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(-d1) * (1 - 1 / Q1)
                - (
                    1
                    + np.exp((self.cost_of_carry - self.rate) * self.dte)
                    * stats.norm.pdf(-d1)
                    / (self.volatility * np.sqrt(self.dte))
                )
                / Q1
            )
        return Si

    @staticmethod
    def european_price_op(opt_tpye, spot, strike, time_to_maturity, rate, volatility, cost_of_carry):
        """
        This method calculates the European option price using BSM formula.
        """
        d1 = (np.log(spot / strike) + (cost_of_carry + volatility**2 / 2) * time_to_maturity) / (
            volatility * np.sqrt(time_to_maturity)
        )
        d2 = d1 - volatility * time_to_maturity**0.5
        if opt_tpye == "Call":
            price = spot * np.exp((cost_of_carry - rate) * time_to_maturity) * ndtr(d1) - strike * np.exp(
                -rate * time_to_maturity
            ) * ndtr(d2)
        else:
            price = strike * np.exp(-rate * time_to_maturity) * ndtr(-d2) - spot * np.exp(
                (cost_of_carry - rate) * time_to_maturity
            ) * ndtr(-d1)
        return price

    def _approximation_american_call(self):
        """
        This method calculates the approximation American option price using Barone-Adesi and Whaley .
        """
        if self.cost_of_carry >= self.rate:
            return self.european_price_op(
                "Call", self.spot, self.strike, self.dte, self.rate, self.volatility, self.cost_of_carry
            )
        else:
            sk = self._Kc()
            N = 2 * self.cost_of_carry / self.volatility**2
            k = 2 * self.rate / (self.volatility**2 * (1 - np.exp(-1 * self.rate * self.dte)))
            d1 = (np.log(sk / self.strike) + (self.cost_of_carry + (self.volatility**2) / 2) * self.dte) / (
                self.volatility * np.sqrt(self.dte)
            )
            Q2 = (-1 * (N - 1) + np.sqrt((N - 1) ** 2 + 4 * k)) / 2
            a2 = (sk / Q2) * (1 - np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(d1))
            if self.spot < sk:
                return (
                    self.european_price_op(
                        "Call",
                        self.spot,
                        self.strike,
                        self.dte,
                        self.rate,
                        self.volatility,
                        self.cost_of_carry,
                    )
                    + a2 * (self.spot / sk) ** Q2
                )
            else:
                return self.spot - self.strike

    def _approximation_american_put(self):
        sk = self._Kp()
        N = 2 * self.cost_of_carry / self.volatility**2
        k = 2 * self.rate / (self.volatility**2 * (1 - np.exp(-1 * self.rate * self.dte)))
        d1 = (np.log(sk / self.strike) + (self.cost_of_carry + (self.volatility**2) / 2) * self.dte) / (
            self.volatility * np.sqrt(self.dte)
        )
        Q1 = (-1 * (N - 1) - np.sqrt((N - 1) ** 2 + 4 * k)) / 2
        a1 = -1 * (sk / Q1) * (1 - np.exp((self.cost_of_carry - self.rate) * self.dte) * ndtr(-d1))
        if self.spot > sk:
            return (
                self.european_price_op(
                    "Put", self.spot, self.strike, self.dte, self.rate, self.volatility, self.cost_of_carry
                )
                + a1 * (self.spot / sk) ** Q1
            )
        else:
            return self.strike - self.spot

    def american_price(self):
        """
        This method calculates the American option price using Barone-Adesi and Whaley approximation.
        """
        call_approx = self._approximation_american_call()
        put_approx = self._approximation_american_put()
        return (call_approx, put_approx)

    def delta(self):
        """
        This method calculates the option delta using Barone-Adesi and Whaley approximation.
        """
        self.spot += _DELTA_S
        c_up_price = self._approximation_american_call()
        p_up_price = self._approximation_american_put()
        self.spot -= 2 * _DELTA_S
        c_down_price = self._approximation_american_call()
        p_down_price = self._approximation_american_put()

        call_delta = (c_up_price - c_down_price) / (2 * _DELTA_S)
        put_delta = (p_up_price - p_down_price) / (2 * _DELTA_S)

        return (call_delta, put_delta)

    def gamma(self):
        """
        This method calculates the option gamma using Barone-Adesi and Whaley approximation.
        """
        original_price = self.american_price()
        self.spot += _DELTA_S
        c_up_price = self._approximation_american_call()
        p_up_price = self._approximation_american_put()
        self.spot -= 2 * _DELTA_S
        c_down_price = self._approximation_american_call()
        p_down_price = self._approximation_american_put()

        call_gamma = (c_up_price - 2 * original_price[0] + c_down_price) / _DELTA_S**2
        put_gamma = (p_up_price - 2 * original_price[1] + p_down_price) / _DELTA_S**2

        return (call_gamma, put_gamma)

    def vega(self):
        """
        This method calculates the option vega using Barone-Adesi and Whaley approximation.
        """
        self.volatility += _DELTA_V
        self.spot += _DELTA_S
        c_up_up_price = self._approximation_american_call()
        p_up_up_price = self._approximation_american_put()
        self.volatility -= 2 * _DELTA_V
        c_up_down_price = self._approximation_american_call()
        p_up_down_price = self._approximation_american_put()

        call_vega = (c_up_up_price - c_up_down_price) / 2
        put_vega = (p_up_up_price - p_up_down_price) / 2

        return (call_vega, put_vega)

    def theta(self):
        """
        This method calculates the option theta using Barone-Adesi and Whaley approximation.
        """
        self.spot += _DELTA_S
        self.dte -= _DELTA_T
        c_up_price = self._approximation_american_call()
        p_up_price = self._approximation_american_put()
        self.dte += _DELTA_T
        c_up_down_price = self._approximation_american_call()
        p_up_down_price = self._approximation_american_put()

        call_theta = c_up_down_price - c_up_price
        put_theta = p_up_down_price - p_up_price

        return (call_theta, put_theta)

    def rho(self):
        """
        This method calculates the option rho using Barone-Adesi and Whaley approximation.
        """
        self.spot += _DELTA_S
        self.rate += _DELTA_R
        c_up_price = self._approximation_american_call()
        p_up_price = self._approximation_american_put()
        self.rate -= 2 * _DELTA_R
        c_down_price = self._approximation_american_call()
        p_down_price = self._approximation_american_put()

        call_rho = c_up_price - c_down_price
        put_rho = p_up_price - p_down_price

        return (call_rho, put_rho)
