class Option:
    def __init__(self, spot, strike, rate, time_to_maturity, volatility, cost_carry=0, dividend=None):
        self.spot = spot
        self.strike = strike
        self.rate = rate
        self.cost_carry = cost_carry
        self.dividend = dividend
        self.time_to_maturity = time_to_maturity  # ! Days
        self.volatility = volatility

    def __repr__(self) -> str:
        return f"Option Class -> Option({self.spot}, {self.strike}, {self.rate}, {self.time_to_maturity}, {self.volatility}, {self.dividend})"

    def __str__(self) -> str:
        return f"Option Class -> Option({self.spot}, {self.strike}, {self.rate}, {self.time_to_maturity}, {self.volatility}, {self.dividend})"


# ! TO DO
class OptionFuture:
    def __init__(self, future, strike, rate, time_to_maturity_op, time_to_maturirity_fut, volatility):
        self.future = future
        self.strike = strike
        self.rate = rate
        self.time_to_maturity_op = time_to_maturity_op  # ! Days
        self.time_to_maturirity_fut = time_to_maturirity_fut  # ! Days
        self.volatility = volatility

    def __repr__(self) -> str:
        return f"""Option Class -> Option({self.future}, {self.strike}, {self.rate},
                                    {self.time_to_maturity_op}, {self.time_to_maturirity_fut}, {self.volatility})"""

    def __str__(self) -> str:
        return f"""Option Class -> Option({self.future}, {self.strike}, {self.rate},
                                    {self.time_to_maturity_op}, {self.time_to_maturirity_fut}, {self.volatility})"""
