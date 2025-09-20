from dataclasses import dataclass
from typing import Optional


@dataclass
class Option:
    spot: float
    strike: float
    rate: float
    time_to_maturity: float  # ! Days
    volatility: float
    dividend: Optional[float] = 0
    cost_carry: Optional[float] = 0

    def __repr__(self) -> str:
        return f"Option Class -> Option({self.spot}, {self.strike}, {self.rate}, {self.time_to_maturity}, {self.volatility}, {self.dividend})"

    def __str__(self) -> str:
        return f"Option Class -> Option({self.spot}, {self.strike}, {self.rate}, {self.time_to_maturity}, {self.volatility}, {self.dividend})"


@dataclass
class OptionFuture:
    future: float
    strike: float
    rate: float
    time_to_maturity_op: float  # ! Days
    time_to_maturirity_fut: float  # ! Days
    volatility: float

    def __repr__(self) -> str:
        return f"""Option Class -> Option({self.future}, {self.strike}, {self.rate},
                                    {self.time_to_maturity_op}, {self.time_to_maturirity_fut}, {self.volatility})"""

    def __str__(self) -> str:
        return f"""Option Class -> Option({self.future}, {self.strike}, {self.rate},
                                    {self.time_to_maturity_op}, {self.time_to_maturirity_fut}, {self.volatility})"""
