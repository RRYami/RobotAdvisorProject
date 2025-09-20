import numpy as np
import scipy.stats as stats

from cl_option import OptionFuture


class Black76(OptionFuture):
    """
    Black76 model

    Attributes:
        future_price: future's price
        strike: strike price
        volatility: volatility
        bit_t: time to maturity
        prompt_t: time to prompt date
        rate: interest rate
    """

    def __init__(self, future_price, strike, volatility, bit_t, prompt_t, rate):
        # Future's price
        self.future_price = future_price

        # Strike price
        self.strike = strike

        # Volatility
        self.volatility = volatility

        # Time to maturity of the Option
        self.bit_t = bit_t / 365

        # Time to prompt date (Future's maturity date)
        self.prompt_t = prompt_t / 365

        # Interest rate
        self.rate = rate

        # Utility
        self._d1_ = (np.log(self.future_price / self.strike) + (self.volatility**2 / 2) * self.bit_t) / (
            self.volatility * self.bit_t**0.5
        )
        self._d2_ = self._d1_ - self.volatility * self.bit_t**0.5
        self._discount_factor_ = np.exp(-(np.log(1 + self.rate)) * self.prompt_t)

        for i in ["call_price", "put_price", "delta", "gamma", "theta", "vega", "rho"]:
            self.__dict__[i] = None

        [self.call_price, self.put_price] = self._price()
        [self.call_delta, self.put_delta] = self._delta()
        [self.call_theta, self.put_theta] = self._theta()
        [self.call_rho, self.put_rho] = self._rho()
        self.gamma = self._gamma()
        self.vega = self._vega()

    def _price(self):
        """Calculate the price of the option using the Black76 model"""
        if self.volatility == 0 or self.bit_t == 0:
            call = np.maximum(0, self.future_price - self.strike)
            put = np.maximum(0, self.strike - self.future_price)
        else:
            call = self._discount_factor_ * (
                self.future_price * stats.norm.cdf(self._d1_) - self.strike * stats.norm.cdf(self._d2_)
            )
            put = self._discount_factor_ * (
                self.strike * stats.norm.cdf(-self._d2_) - self.future_price * stats.norm.cdf(-self._d1_)
            )
        return [call, put]

    def _delta(self):
        """Calculate the delta of the option using the Black76 model"""
        call = np.exp(-self.rate * self.bit_t) * stats.norm.cdf(self._d1_)
        put = np.exp(-self.rate * self.bit_t) * (stats.norm.cdf(self._d1_) - 1)
        return [call, put]

    def _gamma(self):
        """Calculate the gamma of the option using the Black76 model"""
        return (
            np.exp(-self.rate * self.bit_t)
            * stats.norm.pdf(self._d1_)
            / (self.future_price * self.volatility * self.bit_t**0.5)
        )

    def _theta(self):
        """Calculate the theta of the option using the Black76 model"""
        call = (
            -(
                (self.future_price * np.exp(-self.rate * self.bit_t) * stats.norm.pdf(self._d1_) * self.volatility)
                / (2 * self.bit_t**0.5)
            )
            + self.rate * self.future_price * np.exp(-self.rate * self.bit_t) * stats.norm.cdf(self._d1_)
            + self.rate * self.strike * np.exp(-self.rate * self.bit_t) * stats.norm.cdf(self._d2_)
        )
        put = (
            -(
                (self.future_price * np.exp(-self.rate * self.bit_t) * stats.norm.pdf(self._d1_) * self.volatility)
                / (2 * self.bit_t**0.5)
            )
            - self.rate * self.future_price * np.exp(-self.rate * self.bit_t) * stats.norm.cdf(-self._d1_)
            + self.rate * self.strike * np.exp(-self.rate * self.bit_t) * stats.norm.cdf(-self._d2_)
        )
        return [call, put]

    def _vega(self):
        """Calculate the vega of the option using the Black76 model"""
        return self.future_price * np.exp(-self.rate * self.bit_t) * stats.norm.pdf(self._d1_) * self.bit_t**0.5

    def _rho(self):
        call = -self.bit_t * self.call_price
        put = -self.bit_t * self.put_price
        return [call, put]
