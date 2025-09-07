import numpy as np
import pandas as pd
from pandas import Timestamp
import matplotlib.pyplot as plt
import datetime as dt
from dataclasses import dataclass
from typing import Callable
from monte_carlo import SingleSimulationConfig, simulate_geometric_brownian_motion, MultiSimulationConfig, simulate_multi_geometric_brownian_motion, simulate_multi_geometric_brownian_motion_robust

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)

type Exotics = OutperformanceCertificateParams | ReverseBarrierConvertibleCertificateParams | MultiBarrierConvertibleCertificateParams
type SimulationConfig = SingleSimulationConfig | MultiSimulationConfig


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


@dataclass(frozen=True)
class MultiBarrierConvertibleCertificateParams:
    nominal: float
    strike_rates: list[float]
    coupon_rate: float
    risk_free_rate: float
    barrier_rates: list[float]
    valuation_date: pd.Timestamp
    observation_dates: list[pd.Timestamp]
    initial_index_values: list[float]
    maturity_date: pd.Timestamp


def calculate_maturity(valuation_date: pd.Timestamp, maturity_date: pd.Timestamp) -> float:
    return (maturity_date - valuation_date).days / 365.25


def simulate_paths(config: SimulationConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    if isinstance(config, SingleSimulationConfig):
        return simulate_geometric_brownian_motion(config=config, rng=rng)
    elif isinstance(config, MultiSimulationConfig):
        return simulate_multi_geometric_brownian_motion(config=config, rng=rng)
    else:
        raise ValueError("Unsupported config type for path simulation.")


def plot_paths(outperformance_certificate: Exotics, paths: np.ndarray):
    if isinstance(outperformance_certificate, OutperformanceCertificateParams):
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

    elif isinstance(outperformance_certificate, ReverseBarrierConvertibleCertificateParams):

        barrier_value = outperformance_certificate.barrier_rate * outperformance_certificate.initial_index_value
        strike_value = outperformance_certificate.strike_rate * outperformance_certificate.initial_index_value
        observation_dates_days = [
            (date - outperformance_certificate.valuation_date).days for date in outperformance_certificate.observation_dates]

        plt.figure(figsize=(10, 6))
        for path_idx, path in enumerate(paths[:1]):  # Plot first path for clarity
            plt.plot(path, alpha=0.6, label=f'Path {path_idx + 1}' if path_idx == 0 else "")

            # Detect first breach only
            breach_indices = np.where((path[:-1] >= barrier_value) & (path[1:] < barrier_value))[0] + 1
            if len(breach_indices) > 0:
                plt.scatter(
                    breach_indices, path[breach_indices], color='red',
                    label='Barrier Breach' if path_idx == 0 else "", s=5
                )

        # Add barrier and strike levels
        plt.axhline(y=barrier_value, color='r', linestyle='--', label='Barrier Level')
        plt.axhline(y=strike_value, color='g', linestyle='-.', label='Strike Level')

        # Mark observation dates
        for obs_date in observation_dates_days:
            plt.axvline(x=obs_date, color='C2', alpha=0.5,
                        label='Observation Date' if obs_date == observation_dates_days[0] else "")

        # Finalize plot
        plt.xlabel('Time (days)')
        plt.ylabel('Index Level')
        plt.title('Simulated Index Paths with First Barrier Breaches')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif isinstance(outperformance_certificate, MultiBarrierConvertibleCertificateParams):
        plt.figure(figsize=(12, 8))
        num_underlyings = paths.shape[2]
        barrier_values = [rate * init for rate,
                          init in zip(outperformance_certificate.barrier_rates, outperformance_certificate.initial_index_values)]
        strike_value = [rate * init for rate,
                        init in zip(outperformance_certificate.strike_rates, outperformance_certificate.initial_index_values)]
        observation_dates_days = [
            (date - outperformance_certificate.valuation_date).days for date in outperformance_certificate.observation_dates]

        for underlying in range(num_underlyings):
            plt.subplot(num_underlyings, 1, underlying + 1)
            plt.title(f"Underlying {underlying + 1}")

            for path_idx, path in enumerate(paths[:10, :, underlying]):
                plt.plot(path, alpha=0.6)

                # Detect first breach
                breach_indices = np.where((path[:-1] >= barrier_values[underlying]) &
                                          (path[1:] < barrier_values[underlying]))[0] + 1
                if len(breach_indices) > 0:
                    plt.scatter(breach_indices, path[breach_indices], color='red', s=10)

            for observation_date in observation_dates_days:
                plt.axvline(x=observation_date, color='C1', linestyle='--', alpha=0.5,
                            label='Observation Date' if observation_date == observation_dates_days[0] else "")

            plt.axhline(y=barrier_values[underlying], color='red', linestyle='--', label='Barrier Level')
            plt.axhline(y=strike_value[underlying], color='green', linestyle='--', label='Strike Level')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Unsupported certificate type for plotting.")


def calculate_outperformance_payoff(opc: OutperformanceCertificateParams, paths: np.ndarray) -> np.ndarray:
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


def calculate_reverse_barrier_convertible_payoff(rbc: ReverseBarrierConvertibleCertificateParams, paths: np.ndarray, continuous: bool) -> np.ndarray:
    """
    Vectorized payoff for a Reverse Barrier Convertible Certificate.
    Scenarios:
    1) No barrier breach: nominal + coupons
    2) Barrier breach and final < strike: (final / strike) * nominal + coupons
    3) Barrier breach and final >= strike: nominal + coupons
    Returns: discounted payoff per simulation (np.ndarray shape (n_sims,))
    """
    nb_simulations = paths.shape[0]
    maturity_time = calculate_maturity(rbc.valuation_date, rbc.maturity_date)
    payoff = np.zeros(nb_simulations)
    barrier_value = rbc.initial_index_value * rbc.barrier_rate
    strike_value = rbc.initial_index_value * rbc.strike_rate
    observation_dates_days = [
        (date - rbc.valuation_date).days for date in rbc.observation_dates
    ]

    if not continuous:
        # Vectorized barrier breach check
        barrier_breached = np.any(paths < barrier_value, axis=1)
        final_prices = paths[:, -1]

        # Vectorized payoff calculation
        coupon_total = len(rbc.observation_dates) * rbc.coupon_rate * rbc.nominal
        full_payoff = rbc.nominal + coupon_total

        # No breach: full payoff
        payoff[~barrier_breached] = full_payoff

        # Breach and final < strike: proportional + coupon
        breach_and_low = barrier_breached & (final_prices < strike_value)
        payoff[breach_and_low] = (final_prices[breach_and_low] / strike_value) * rbc.nominal + coupon_total

        # Breach and final >= strike: full payoff
        breach_and_high = barrier_breached & (final_prices >= strike_value)
        payoff[breach_and_high] = full_payoff

        discounted_payoff = payoff * np.exp(-rbc.risk_free_rate * maturity_time)
        return discounted_payoff

    elif continuous:
        nb_simulations, n_steps = paths.shape
        payoff = np.zeros(nb_simulations)

        # Vectorized barrier breach check at observation dates
        observed_paths = paths[:, observation_dates_days]
        barrier_breached = np.any(observed_paths < barrier_value, axis=1)
        final_prices = paths[:, -1]

        # Vectorized payoff calculation
        coupon_total = len(rbc.observation_dates) * rbc.coupon_rate * rbc.nominal
        full_payoff = rbc.nominal + coupon_total

        # No breach: full payoff
        payoff[~barrier_breached] = full_payoff

        # Breach and final < strike: proportional + coupon
        breach_and_low = barrier_breached & (final_prices < strike_value)
        payoff[breach_and_low] = (final_prices[breach_and_low] / strike_value) * rbc.nominal + coupon_total

        # Breach and final >= strike: full payoff
        breach_and_high = barrier_breached & (final_prices >= strike_value)
        payoff[breach_and_high] = full_payoff

        maturity_time = observation_dates_days[-1] / 252
        discounted_payoff = payoff * np.exp(-rbc.risk_free_rate * maturity_time)
        return discounted_payoff


def calculate_multi_barrier_convertible_payoff(mbc: MultiBarrierConvertibleCertificateParams, paths: np.ndarray, time_grid: np.ndarray, continuous: bool):
    """
    Vectorized payoff for a Multi-Barrier Convertible Certificate.
    Scenarios:
    1) No barrier breach: nominal + coupons
    2) Barrier breach and final < strike: (final / strike) * nominal + coupons
    3) Barrier breach and final >= strike: nominal + coupons
    Returns: discounted payoff per simulation (np.ndarray shape (n_sims,))
    """
    n_sims, n_steps, _ = paths.shape
    nb_simulations = paths.shape[0]
    maturity_time = calculate_maturity(mbc.valuation_date, mbc.maturity_date)
    payoff = np.zeros(nb_simulations)
    barrier_values = [rate * init for rate, init in zip(mbc.barrier_rates, mbc.initial_index_values)]
    strike_values = [rate * init for rate, init in zip(mbc.strike_rates, mbc.initial_index_values)]
    observation_dates_days = [
        (date - mbc.valuation_date).days for date in mbc.observation_dates
    ]

    if not continuous:
        # --- Select observation indices ---
        if hasattr(mbc, "observation_dates") and mbc.observation_dates is not None:
            obs_indices = np.searchsorted(time_grid, observation_dates_days, side='right') - 1
        else:
            obs_indices = np.arange(n_steps)  # fallback: monitor all time steps

        # --- Barrier check (vectorized) ---
        # Shape: (n_sims, len(obs_indices), n_underlyings)
        obs_prices = paths[:, obs_indices, :]
        barrier_breached_per_sim = np.any(obs_prices < np.array(barrier_values), axis=1)
        barrier_breached = np.any(barrier_breached_per_sim, axis=1)

        # --- Final prices ---
        final_prices = paths[:, -1, :]  # (n_sims, n_underlyings)
        worst_final_price = np.min(final_prices / np.array(strike_values), axis=1)

        # --- Payoff logic ---
        coupon_total = len(mbc.observation_dates) * mbc.coupon_rate * mbc.nominal
        full_payoff = mbc.nominal + coupon_total
        payoff = np.full(n_sims, full_payoff)

        # Breach & worst < 1 → proportional + coupon
        breach_and_low = barrier_breached & (worst_final_price < 1)
        payoff[breach_and_low] = worst_final_price[breach_and_low] * mbc.nominal + coupon_total

        # Breach & worst ≥ 1 → full payoff (already default)
        # Breach and worst_final_price >= 1: full payoff
        breach_and_high = barrier_breached & (worst_final_price >= 1)
        payoff[breach_and_high] = full_payoff
        # --- Discounted payoff ---
        discounted_payoff = payoff * np.exp(-mbc.risk_free_rate * maturity_time)
        return discounted_payoff
    elif continuous:
        barrier_breached_per_sim = np.zeros((nb_simulations, len(barrier_values)), dtype=bool)
        barrier_breached_per_sim = np.any(paths < np.array(barrier_values), axis=1)
        barrier_breached = np.any(barrier_breached_per_sim, axis=1)
        # for u in range(len(rbc.barriers)):
        #     barrier_breached_per_sim[:, u] = np.any(paths[:, :, u] < rbc.barrier_values[u], axis=1)
        # barrier_breached = np.any(barrier_breached_per_sim, axis=1)

        # Final prices for all underlyings
        final_prices = paths[:, -1, :]  # Shape: (nb_simulations, num_underlyings)

        # Worst-of logic: min(final_prices / strike_values) per simulation
        worst_final_price = np.min(final_prices / np.array(strike_values), axis=1)

        # Vectorized payoff calculation
        coupon_total = len(observation_dates_days) * mbc.coupon_rate * mbc.nominal
        full_payoff = mbc.nominal + coupon_total

        # No breach: full payoff
        payoff[~barrier_breached] = full_payoff

        # Breach and worst_final_price < 1: proportional + coupon
        breach_and_low = barrier_breached & (worst_final_price < 1)
        payoff[breach_and_low] = worst_final_price[breach_and_low] * mbc.nominal + coupon_total

        # Breach and worst_final_price >= 1: full payoff
        breach_and_high = barrier_breached & (worst_final_price >= 1)
        payoff[breach_and_high] = full_payoff

        discounted_payoff = payoff * np.exp(-mbc.risk_free_rate * maturity_time)
        return discounted_payoff


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

    opc_sim_config = SingleSimulationConfig(
        initial_index_value=opc_params.initial_index_value,
        mu=0.05,
        volatility=0.34,
        maturity=calculate_maturity(opc_params.valuation_date, opc_params.maturity_date),
        nb_simulations=10000
    )

    opc_paths = simulate_paths(opc_sim_config, rng=None)
    opc_discounted_payoffs = calculate_outperformance_payoff(opc_params, opc_paths)

    print("\n")
    print("-----------------TEST Outperformance Certificate-----------------")
    print("-----------------------------------------------------------------")
    print("Payout Paths Discounted:", opc_discounted_payoffs)
    print("Average OPC Price Discounted:", np.mean(opc_discounted_payoffs))
    plot_paths(opc_params, opc_paths)

    rbc_params = ReverseBarrierConvertibleCertificateParams(
        nominal=100,
        strike_rate=1.0,
        coupon_rate=0.02080,
        risk_free_rate=0.03,
        barrier_rate=0.80,
        valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
        observation_dates=[
            pd.Timestamp(dt.date(2022, 1, 4)),
            pd.Timestamp(dt.date(2023, 1, 4)),
            pd.Timestamp(dt.date(2024, 1, 4))
        ],
        initial_index_value=88.64,
        maturity_date=pd.Timestamp(dt.date(2024, 1, 4))
    )

    rbc_sim_config = SingleSimulationConfig(
        initial_index_value=rbc_params.initial_index_value,
        mu=0.03,
        volatility=0.279,
        maturity=calculate_maturity(rbc_params.valuation_date, rbc_params.maturity_date),
        nb_simulations=10000
    )

    rbc_paths = simulate_paths(rbc_sim_config, rng=None)
    rbc_discounted_payoffs = calculate_reverse_barrier_convertible_payoff(rbc_params, rbc_paths, continuous=False)
    print("\n")
    print("-----------------TEST Reverse Barrier Certificate-----------------")
    print("-------------------------------------------------------------------")
    print("Payout Paths Discounted:", rbc_discounted_payoffs)
    print("Average RBC Price Discounted:", np.mean(rbc_discounted_payoffs))
    plot_paths(rbc_params, rbc_paths)

    mbc_params = MultiBarrierConvertibleCertificateParams(
        nominal=100,
        strike_rates=[1.0, 1.0, 1.0, 1.0],
        coupon_rate=0.00428,
        risk_free_rate=0.04,
        barrier_rates=[0.80, 0.80, 0.80, 0.80],
        valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
        observation_dates=[
            pd.Timestamp(dt.date(2022, 1, 4)),
            pd.Timestamp(dt.date(2023, 1, 4)),
            pd.Timestamp(dt.date(2024, 1, 4))
        ],
        initial_index_values=[5667.20, 11080.90, 12217.70, 4939.44],
        maturity_date=pd.Timestamp(dt.date(2024, 1, 4))
    )

    mbc_sim_config = MultiSimulationConfig(
        initial_index_values=np.array(mbc_params.initial_index_values),
        mu=np.array([0.10, 0.15, 0.16, 0.17]),
        volatility=np.array([0.10, 0.14, 0.15, 0.18]),
        correlation_matrix=np.array([
            [1.0, 0.51, 0.49, 0.57],
            [0.51, 1.0, 0.71, 0.87],
            [0.49, 0.71, 1.0, 0.81],
            [0.57, 0.87, 0.81, 1.0]]),
        maturity=calculate_maturity(mbc_params.valuation_date, mbc_params.maturity_date),
        nb_simulations=10000,
        time_step_per_year=252
    )
    mbc_paths = simulate_paths(mbc_sim_config, rng=None)
    time_grid = np.linspace(0, mbc_sim_config.maturity, int(
        mbc_sim_config.maturity * mbc_sim_config.time_step_per_year) + 1)
    mbc_discounted_payoffs = calculate_multi_barrier_convertible_payoff(
        mbc_params, mbc_paths, time_grid, continuous=False)
    print("\n")
    print("-----------------TEST Multi Barrier Certificate-----------------")
    print("-----------------------------------------------------------------")
    print("Payout Paths Discounted:", mbc_discounted_payoffs)
    print("Average MBC Price Discounted:", np.mean(mbc_discounted_payoffs))
    plot_paths(mbc_params, mbc_paths)
