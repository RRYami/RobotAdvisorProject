import numpy as np
import scipy.stats as stats


class BinomialMethod:
    """
    This is a class for Options contract for pricing European options on stocks/index without dividends.
    Attributes:
        spot          : int or float
        strike        : int or float
        rate          : float
        volatility    : float
        big_t         : int or float [days to expiration in number of years]
        n_steps       : int [number of steps]
    """

    def __init__(
        self,
        spot,
        strike,
        rate,
        volatility,
        big_t,
        n_steps,
        callprice=None,
        putprice=None,
    ):
        # Spot Price
        self.spot = spot

        # Option Strike
        self.strike = strike

        # Interest Rate
        self.rate = rate

        # Volatility
        self.volatility = volatility

        # Days To Expiration
        self.big_t = big_t

        # Number of Steps
        self.n_steps = n_steps

        # Call price
        self.callprice = callprice

        # Put price
        self.putprice = putprice

        # Utility
        self._dt_ = self.big_t / self.n_steps
        self._u_ = np.exp(self.volatility * np.sqrt(self._dt_))
        self._d_ = 1 / self._u_
        self._p_ = (np.exp(self.rate * self._dt_) - self._d_) / (self._u_ - self._d_)
        self._d1_ = (np.log(self.spot / self.strike) + (self.rate + 0.5 * self.volatility**2) * self.big_t) / (
            self.volatility * np.sqrt(self.big_t)
        )
        self._d2_ = self._d1_ - self.volatility * np.sqrt(self.big_t)

        if self.strike == 0:
            raise ZeroDivisionError("The strike price cannot be zero")
        else:
            pass

        for i in [
            "call_price",
            "put_price",
            "call_delta",
            "put_delta",
            "gamma",
            "vega",
            "call_theta",
            "put_theta",
        ]:
            self.__dict__[i] = None

        [self.call_price, self.put_price] = self._price()
        [self.call_delta, self.put_delta] = self._delta()
        self.gamma = self._gamma()
        self.vega = self._vega()
        [self.call_theta, self.put_theta] = self._theta()

    def _price(self) -> tuple[float, float]:
        """
        This function calculates the price of the option using the binomial method.
        """
        # Initialize the stock price tree
        stock_price = np.zeros((self.n_steps + 1, self.n_steps + 1))
        stock_price[0, 0] = self.spot
        for i in range(1, self.n_steps + 1):
            stock_price[i, 0] = stock_price[i - 1, 0] * self._u_
            for j in range(1, i + 1):
                stock_price[i, j] = stock_price[i - 1, j - 1] * self._d_

        # Initialize the option value tree
        call = np.zeros((self.n_steps + 1, self.n_steps + 1))
        put = np.zeros((self.n_steps + 1, self.n_steps + 1))
        for j in range(self.n_steps + 1):
            call[self.n_steps, j] = max(stock_price[self.n_steps, j] - self.strike, 0)
            put[self.n_steps, j] = max(self.strike - stock_price[self.n_steps, j], 0)

        # Calculate the option value at each node
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                call[i, j] = np.exp(-self.rate * self._dt_) * (
                    self._p_ * call[i + 1, j] + (1 - self._p_) * call[i + 1, j + 1]
                )
                put[i, j] = np.exp(-self.rate * self._dt_) * (
                    self._p_ * put[i + 1, j] + (1 - self._p_) * put[i + 1, j + 1]
                )
        return (float(call[0, 0]), float(put[0, 0]))

    def _delta(self) -> tuple[float, float]:
        """Calculates the delta of the option.

        Returns:
            tuple: (call_delta, put_delta)
        """
        call_delta = float(stats.norm.cdf(self._d1_))
        put_delta = float(stats.norm.cdf(self._d1_) - 1)
        return (call_delta, put_delta)

    def _gamma(self) -> float:
        gamma: float = stats.norm.pdf(self._d1_) / (self.spot * self.volatility * np.sqrt(self.big_t))
        return gamma

    def _vega(self) -> float:
        """Calculates the vega of the option.

        Returns:
            float: vega
        """
        vega: float = self.spot * stats.norm.pdf(self._d1_) * np.sqrt(self.big_t)
        return vega

    def _theta(self) -> tuple[float, float]:
        """Calculates the theta of the option.

        Returns:
            tuple: (call_theta, put_theta)
        """
        call_theta: float = -(self.spot * stats.norm.pdf(self._d1_) * self.volatility) / (
            2 * np.sqrt(self.big_t)
        ) - self.rate * self.strike * np.exp(-self.rate * self.big_t) * stats.norm.cdf(self._d2_)
        put_theta: float = -(self.spot * stats.norm.pdf(self._d1_) * self.volatility) / (
            2 * np.sqrt(self.big_t)
        ) + self.rate * self.strike * np.exp(-self.rate * self.big_t) * stats.norm.cdf(-self._d2_)
        return (call_theta, put_theta)
