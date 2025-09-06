import numpy as np
import pandas as pd
from pandas import Timestamp
import matplotlib.pyplot as plt
import datetime as dt
from dataclasses import dataclass
from typing import Callable
from Exotics.monte_carlo import SimulationConfig, simulate_geometric_brownian_motion

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)


@dataclass(frozen=True)
class OutperformanceCertificateParams:
    nominal: float
    participation_rate: float
    cap_rate: float
    risk_free_rate: float
    initial_index_value: float
    valuation_date: pd.Timestamp
    maturity_date: pd.Timestamp
    max_amount_factor: float


@dataclass(frozen=True)
class ReverseBarrierConvertibleCertificateParams:
    nominal: float
    strike_rate: float
    coupon_rate: float
    risk_free_rate: float
    barrier_rate: float
    valuation_date: pd.Timestamp
    observation_dates: list[pd.Timestamp]
    initial_index_value: float
    valuation_date: pd.Timestamp
    maturity_date: pd.Timestamp


def calculate_maturity(valuation_date: pd.Timestamp, maturity_date: pd.Timestamp) -> float:
    return (maturity_date - valuation_date).days / 365.25


def simulate_paths(config: SimulationConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    return simulate_geometric_brownian_motion(config=config, rng=rng)


def plot_paths(outperformance_certificate: OutperformanceCertificateParams, paths: np.ndarray):

    cap_level = outperformance_certificate.initial_index_value * outperformance_certificate.cap_rate

    plt.figure(figsize=(10, 6))

    for path_idx, path in enumerate(paths[:1]):  # Plot first path for clarity
        plt.plot(path, alpha=0.6, label=f'Path {path_idx + 1}' if path_idx == 0 else "")

        # Detect first breach only
        cap_hit = np.where((path[:-1] <= cap_level) & (path[1:] > cap_level))[0] + 1
        if len(cap_hit) > 0:
            plt.scatter(
                cap_hit, path[cap_hit], color='red',
                label='Cap Breach' if path_idx == 0 else "", s=5
            )

    # Add cap levels
    plt.axhline(y=cap_level, color='r', linestyle='--', label='Cap Level')

    # Finalize plot
    plt.xlabel('Time (days)')
    plt.ylabel('Index Level')
    plt.title('Simulated Index Paths with Cap Breaches')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_payoff(opc: OutperformanceCertificateParams, paths: np.ndarray) -> np.ndarray:
    """
    Vectorized payoff for an Outperformance Certificate.

    Scenarios:
    1) final <= initial  -> nominal * (final / initial)
    2) initial < final <= cap_level -> nominal * (1 + participation_rate * performance)
    3) final > cap_level -> nominal * (1 + participation_rate * cap_rate)

    Returns: discounted payoff per simulation (np.ndarray shape (n_sims,))
    """
    # Validate shapes: accept (n_sims, n_steps) or (n_sims, n_steps, 1)
    if paths.ndim == 3:
        if paths.shape[2] != 1:
            raise ValueError("OutperformancePayoffStrategy expects single-underlying paths (last dim == 1).")
        final_prices = paths[:, -1, 0]
    elif paths.ndim == 2:
        final_prices = paths[:, -1]
    else:
        raise ValueError("Unexpected paths array shape for OutperformancePayoffStrategy.")

    # Cast scalars
    initial = float(opc.initial_index_value)
    if initial == 0:
        raise ValueError("Initial index value must be non-zero.")

    nominal = opc.nominal
    participation = opc.participation_rate
    cap_rate = opc.cap_rate
    cap_level = initial * cap_rate
    maturity_time = calculate_maturity(opc.valuation_date, opc.maturity_date)
    r = opc.risk_free_rate
    max_amount_rate = opc.max_amount_factor

    # Vectorized performance
    performance = (final_prices - initial) / initial

    # Prepare payoff array
    payoff = np.empty_like(final_prices, dtype=float)

    # Case 1: final <= initial
    mask1 = final_prices <= initial
    payoff[mask1] = nominal * (final_prices[mask1] / initial)

    # Case 2: initial < final <= cap_level
    mask2 = (final_prices > initial) & (final_prices <= cap_level)
    payoff[mask2] = nominal * (1.0 + participation * performance[mask2])

    # Case 3: final > cap_level (cap the performance at cap_rate)
    mask3 = final_prices > cap_level
    payoff[mask3] = nominal * max_amount_rate

    # Discount to present value
    discounted_payoff = payoff * np.exp(-r * maturity_time)

    return discounted_payoff

    # ###########################################################
    # # TEST 2 outperformance certificate
    # ##########################################################


if __name__ == "__main__":
    opc_params = OutperformanceCertificateParams(
        nominal=100,
        participation_rate=1.75,
        cap_rate=1.25,
        risk_free_rate=0.04,
        initial_index_value=88.64,
        valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
        maturity_date=pd.Timestamp(dt.date(2024, 1, 4)),
        max_amount_factor=1.4375
    )

    config = SimulationConfig(
        initial_index_value=opc_params.initial_index_value,
        mu=0.05,
        volatility=0.34,
        maturity=calculate_maturity(opc_params.valuation_date, opc_params.maturity_date),
        nb_simulations=10000
    )

    opc_paths = simulate_paths(config, rng=None)
    opc_discounted_payoffs = calculate_payoff(opc_params, opc_paths)

    print("\n")
    print("-----------------TEST Outperformance Certificate-----------------")
    print("-----------------------------------------------------------------")
    print("Payout Paths Discounted:", opc_discounted_payoffs)
    print("Average OPC Price Discounted:", np.mean(opc_discounted_payoffs))
    plot_paths(opc_params, opc_paths)
