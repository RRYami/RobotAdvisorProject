import numpy as np
from cl_option import Option
from sklearn.tree import DecisionTreeRegressor


class MonteCarloEuropean(Option):
    def __init__(
        self, spot, strike, rate, time_to_maturity, volatility, opt_type, cost_carry=0, dividend=None
    ):
        super().__init__(spot, strike, rate, time_to_maturity, volatility, cost_carry, dividend)
        self.num_simulations = 1000000
        self.opt_type = opt_type
        self.dte = self.time_to_maturity / 365

    def simulate_stock_price(self) -> np.ndarray:
        """Simulate stock prices

        Returns:
            np.ndarray: Stock prices at time t
        """
        Z = np.random.standard_normal(self.num_simulations)
        if self.dividend is None or self.dividend == 0:
            stock_t = self.spot * np.exp(
                ((self.rate) - 0.5 * self.volatility**2) * self.dte
                + self.volatility * np.sqrt(self.dte) * Z
            )
        else:
            stock_t = self.spot * np.exp(
                (self.rate - self.dividend)
                - 0.5 * self.volatility**2 * self.dte
                + self.volatility * np.sqrt(self.dte) * Z
            )
        return stock_t

    def calculate_payoff(self) -> np.ndarray:
        if self.opt_type == "Call":
            return np.maximum(self.simulate_stock_price() - self.strike, 0)
        else:
            return np.maximum(self.strike - self.simulate_stock_price(), 0)

    def option_price(self) -> float:
        payoff = self.calculate_payoff()
        option_price = np.exp(-self.rate * self.dte) * np.mean(payoff)
        return option_price


class MonteCaroloAmerican(Option):
    def __init__(
        self, spot, strike, rate, time_to_maturity, volatility, opt_type, dividend=0
    ):
        super().__init__(spot, strike, rate, time_to_maturity, volatility, dividend)
        self.num_simulations = 100000
        self.opt_type = opt_type
        self.num_steps = 200

    def price_with_decision_tree(self):
        dt = self.time_to_maturity / self.num_steps
        discount_factor = np.exp(-self.rate * dt)
        S = np.zeros((self.num_simulations, self.num_steps + 1))
        S[:, 0] = self.spot
        for t in range(1, self.num_steps + 1):
            Z = np.random.standard_normal(self.num_simulations)
            S[:, t] = S[:, t - 1] * np.exp(
                (self.rate - self.dividend - 0.5 * self.volatility**2) * dt
                + self.volatility * np.sqrt(dt) * Z
            )

        if self.opt_type == "Put":
            payoff = np.maximum(self.strike - S, 0)
        elif self.opt_type == "Call":
            payoff = np.maximum(S - self.strike, 0)
        else:
            raise ValueError("self.opt_type must be 'Call' or 'Put'")

        cash_flow = payoff[:, -1]

        for t in range(self.num_steps - 1, 0, -1):
            in_the_money = payoff[:, t] > 0
            X = S[in_the_money, t].reshape(-1, 1)
            Y = cash_flow[in_the_money] * discount_factor
            if len(X) == 0:
                continue

            tree = DecisionTreeRegressor(max_leaf_nodes=10)
            tree.fit(X, Y)
            continuation_value = tree.predict(X)

            exercise_value = payoff[in_the_money, t]
            exercise = exercise_value > continuation_value
            cash_flow[in_the_money] = np.where(
                exercise, exercise_value, cash_flow[in_the_money] * discount_factor
            )

        option_price = np.mean(cash_flow * discount_factor)

        return option_price


print(MonteCarloEuropean(101.15, 98.01, 0.01, 60, 0.00991, "Call").option_price())
