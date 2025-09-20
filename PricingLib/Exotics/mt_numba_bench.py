import numpy as np
from numba import njit, prange
from typing import Optional
import time
from monte_carlo import MultiSimulationConfig, HypersphereDecomposition, SingleSimulationConfig


def simulate_geometric_brownian_motion(config: SingleSimulationConfig, rng: int | None = None) -> np.ndarray:
    if rng is not None:
        np.random.seed(rng)
    dt = 1 / config.time_step_per_year
    time_steps = int(config.maturity * config.time_step_per_year)
    paths = np.zeros((config.nb_simulations, time_steps + 1))
    paths[:, 0] = config.initial_index_value
    z = np.random.standard_normal((time_steps, config.nb_simulations))
    drift = (config.mu - 0.5 * config.volatility ** 2) * dt
    diffusion = config.volatility * np.sqrt(dt) * z
    log_returns = drift + diffusion
    print(log_returns)
    cum_log_returns = np.cumsum(log_returns, axis=0)
    paths[:, 1:] = config.initial_index_value * np.exp(cum_log_returns.T)
    return paths


@njit(fastmath=True, cache=True)
def simulate_geometric_brownian_motion_numba_core(
    nb_simulations: int,
    time_steps: int,
    initial_index_value: float,
    mu: float,
    volatility: float,
    dt: float,
    z: np.ndarray  # Pre-generated random numbers
) -> np.ndarray:
    # Pre-compute all constants
    drift = (mu - 0.5 * volatility * volatility) * dt
    vol_sqrt_dt = volatility * np.sqrt(dt)

    # Allocate arrays
    paths = np.empty((nb_simulations, time_steps + 1))
    paths[:, 0] = initial_index_value

    # Manual cumulative sum implementation
    cum_log_returns = np.empty((time_steps, nb_simulations))

    # Compute log returns and cumulative sum manually
    for t in prange(time_steps):
        log_returns = drift + vol_sqrt_dt * z[t, :]
        if t == 0:
            cum_log_returns[t, :] = log_returns
        else:
            cum_log_returns[t, :] = cum_log_returns[t-1, :] + log_returns

    # Apply exponential and scale by initial value
    paths[:, 1:] = initial_index_value * np.exp(cum_log_returns.T)

    return paths


# Wrapper that matches your exact interface
def simulate_geometric_brownian_motion_numba_batch(config: SingleSimulationConfig, rng: Optional[int] = None) -> np.ndarray:
    # Set seed exactly like your original function
    if rng is not None:
        np.random.seed(rng)

    dt = 1 / config.time_step_per_year
    time_steps = int(config.maturity * config.time_step_per_year)

    # Generate random numbers exactly like your original function
    z = np.random.standard_normal((time_steps, config.nb_simulations))

    return simulate_geometric_brownian_motion_numba_core(
        nb_simulations=config.nb_simulations,
        time_steps=time_steps,
        initial_index_value=config.initial_index_value,
        mu=config.mu,
        volatility=config.volatility,
        dt=dt,
        z=z
    )


@njit(parallel=True, fastmath=True, cache=True)
def simulate_multi_gbm_numba_cpu_blocked(
    initial_values: np.ndarray,
    mu: np.ndarray,
    volatility: np.ndarray,
    L_chol: np.ndarray,
    dt: float,
    time_steps: int,
    nb_simulations: int,
    random_numbers: np.ndarray,
    block_size: int = 1024
) -> np.ndarray:
    """
    Block-based multi-asset GBM simulation using Numba JIT compilation.
    Processes simulations in blocks for better cache locality and memory efficiency.
    """
    num_assets = len(initial_values)
    paths = np.zeros((nb_simulations, time_steps + 1, num_assets))

    # Pre-compute drift terms
    drift = np.empty(num_assets)
    vol_sqrt_dt = np.empty(num_assets)
    for j in prange(num_assets):
        drift[j] = (mu[j] - 0.5 * volatility[j] ** 2) * dt
        vol_sqrt_dt[j] = volatility[j] * np.sqrt(dt)

    # Calculate number of blocks
    num_blocks = (nb_simulations + block_size - 1) // block_size

    # Process simulations in blocks
    for block_idx in prange(num_blocks):
        start_sim = block_idx * block_size
        end_sim = min(start_sim + block_size, nb_simulations)

        # Set initial values for this block
        for i in prange(start_sim, end_sim):
            for j in prange(num_assets):
                paths[i, 0, j] = initial_values[j]

        # Process each time step for this block
        for t in prange(1, time_steps + 1):
            # Process all simulations in this block for current time step
            for i in prange(start_sim, end_sim):
                # Apply Cholesky correlation
                for j in prange(num_assets):
                    correlated_z = 0.0
                    for k in prange(num_assets):
                        correlated_z += L_chol[j, k] * random_numbers[i, t-1, k]

                    # Calculate log return and update price
                    log_return = drift[j] + vol_sqrt_dt[j] * correlated_z
                    paths[i, t, j] = paths[i, t-1, j] * np.exp(log_return)

    return paths


@njit(parallel=True, fastmath=True, cache=True)
def simulate_multi_gbm_numba_cpu(
    initial_values: np.ndarray,
    mu: np.ndarray,
    volatility: np.ndarray,
    L_chol: np.ndarray,
    dt: float,
    time_steps: int,
    nb_simulations: int,
    random_numbers: np.ndarray
) -> np.ndarray:
    """
    Standard implementation for comparison.
    """
    num_assets = len(initial_values)
    paths = np.zeros((nb_simulations, time_steps + 1, num_assets))

    # Set initial values in parallel
    for i in prange(nb_simulations):
        for j in range(num_assets):
            paths[i, 0, j] = initial_values[j]

    # Pre-compute drift terms
    drift = np.empty(num_assets)
    for j in range(num_assets):
        drift[j] = (mu[j] - 0.5 * volatility[j] ** 2) * dt

    sqrt_dt = np.sqrt(dt)

    # Main simulation loop with parallel execution over simulations
    for t in range(1, time_steps + 1):
        for i in prange(nb_simulations):
            # Apply Cholesky correlation
            for j in range(num_assets):
                correlated_z = 0.0
                for k in range(num_assets):
                    correlated_z += L_chol[j, k] * random_numbers[i, t-1, k]

                # Calculate log return and update price
                log_return = drift[j] + volatility[j] * sqrt_dt * correlated_z
                paths[i, t, j] = paths[i, t-1, j] * np.exp(log_return)

    return paths


def simulate_multi_geometric_brownian_motion_numba_cpu(
    config: MultiSimulationConfig,
    rng: Optional[np.random.Generator] = None,
    method: str = "sim_blocked",
    sim_block_size: int = 1024,
) -> np.ndarray:
    """
    Optimized CPU simulation wrapper with multiple blocking strategies.

    Parameters:
    -----------
    method : str
        - "standard": No blocking
        - "sim_blocked": Block by simulations only
        - "time_blocked": Block by time steps only
        - "2d_blocked": Block by both simulations and time (recommended)
    sim_block_size : int
        Number of simulations per block
    time_block_size : int
        Number of time steps per block
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = 1.0 / config.time_step_per_year
    time_steps = int(config.time_step_per_year * config.maturity)
    num_assets = len(config.initial_index_values)

    # Validate and fix correlation matrix
    if not HypersphereDecomposition.is_positive_definite(config.correlation_matrix):
        config.correlation_matrix = HypersphereDecomposition(config.correlation_matrix).optimization()

    # Pre-compute Cholesky decomposition
    L = np.linalg.cholesky(config.correlation_matrix)

    # Pre-generate all random numbers
    random_numbers = rng.standard_normal((config.nb_simulations, time_steps, num_assets))

    # Choose implementation based on method
    if method == "standard":
        return simulate_multi_gbm_numba_cpu(
            config.initial_index_values, config.mu, config.volatility, L,
            dt, time_steps, config.nb_simulations, random_numbers
        )
    elif method == "sim_blocked":
        return simulate_multi_gbm_numba_cpu_blocked(
            config.initial_index_values, config.mu, config.volatility, L,
            dt, time_steps, config.nb_simulations, random_numbers, sim_block_size
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def simulate_multi_geometric_brownian_motion_pure_numpy(
    config: MultiSimulationConfig,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Pure NumPy vectorized implementation for comparison.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = 1.0 / config.time_step_per_year
    time_steps = int(config.time_step_per_year * config.maturity)
    num_assets = len(config.initial_index_values)

    # Validate correlation matrix
    if not HypersphereDecomposition.is_positive_definite(config.correlation_matrix):
        config.correlation_matrix = HypersphereDecomposition(config.correlation_matrix).optimization()

    # Pre-compute constants
    L = np.linalg.cholesky(config.correlation_matrix)
    sqrt_dt = np.sqrt(dt)
    drift = (config.mu - 0.5 * config.volatility ** 2) * dt

    # Generate all random numbers at once
    z = rng.standard_normal((config.nb_simulations, time_steps, num_assets))

    # Apply correlation in one matrix operation
    correlated_z = np.einsum('ijk,kl->ijl', z, L.T)

    # Calculate log returns for all simulations and time steps
    log_returns = drift[np.newaxis, np.newaxis, :] + \
        config.volatility[np.newaxis, np.newaxis, :] * sqrt_dt * correlated_z

    # Cumulative sum to get cumulative log returns
    cum_log_returns = np.cumsum(log_returns, axis=1)

    # Initialize paths array
    paths = np.zeros((config.nb_simulations, time_steps + 1, num_assets))
    paths[:, 0, :] = config.initial_index_values

    # Calculate all prices at once using broadcasting
    paths[:, 1:, :] = config.initial_index_values[np.newaxis, np.newaxis, :] * \
        np.exp(cum_log_returns)

    return paths


def benchmark_cpu_implementations():
    """Benchmark CPU-only Monte Carlo implementations with different blocking strategies."""

    # Test configuration
    config = MultiSimulationConfig(
        initial_index_values=np.array([100.0, 110.0, 120.0, 90.0]),
        mu=np.array([0.05, 0.06, 0.04, 0.07]),
        volatility=np.array([0.2, 0.25, 0.18, 0.22]),
        correlation_matrix=np.array([
            [1.0, 0.3, 0.2, 0.1],
            [0.3, 1.0, 0.4, 0.2],
            [0.2, 0.4, 1.0, 0.3],
            [0.1, 0.2, 0.3, 1.0]
        ]),
        maturity=1.0,
        nb_simulations=10000,
        time_step_per_year=252
    )

    from monte_carlo import simulate_multi_geometric_brownian_motion

    implementations = [
        ("Original", lambda cfg, rng: simulate_multi_geometric_brownian_motion(cfg, rng)),
        ("Pure NumPy", simulate_multi_geometric_brownian_motion_pure_numpy),
        ("Numba Standard", lambda cfg, rng: simulate_multi_geometric_brownian_motion_numba_cpu(cfg, rng, "standard")),
        ("Numba Sim Blocked", lambda cfg, rng: simulate_multi_geometric_brownian_motion_numba_cpu(cfg, rng, "sim_blocked", 1024)),
    ]

    results = {}
    rng = np.random.default_rng(42)

    print(f"CPU Blocking Benchmark:")
    print(f"Simulations: {config.nb_simulations:,}")
    print(f"Time steps: {int(config.time_step_per_year * config.maturity)}")
    print(f"Assets: {len(config.initial_index_values)}")
    print("-" * 70)

    for name, func in implementations:
        print(f"Running {name}...")

        # Warm-up for JIT compilation
        if "Numba" in name:
            small_config = MultiSimulationConfig(
                initial_index_values=config.initial_index_values,
                mu=config.mu,
                volatility=config.volatility,
                correlation_matrix=config.correlation_matrix,
                maturity=config.maturity,
                nb_simulations=1000,
                time_step_per_year=50
            )
            try:
                _ = func(small_config, np.random.default_rng(42))
                print(f"  Warm-up completed")
            except Exception as e:
                print(f"  Warm-up failed: {e}")
                continue

        # Actual benchmark
        start_time = time.perf_counter()
        try:
            paths = func(config, np.random.default_rng(42))
            end_time = time.perf_counter()

            runtime = end_time - start_time
            results[name] = {
                'time': runtime,
                'paths': paths,
                'mean_final': np.mean(paths[:, -1, :], axis=0)
            }

            print(f"  Time: {runtime:.3f}s")
            print(f"  Final prices (mean): {results[name]['mean_final']}")

        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {'time': float('inf'), 'error': str(e)}

    # Performance comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)

    if results:
        valid_results = {k: v for k, v in results.items()
                         if 'time' in v and v['time'] != float('inf')}

        if valid_results:
            baseline_time = results['Original']['time'] if 'Original' in results else min(
                r['time'] for r in valid_results.values())

            for name, result in results.items():
                if 'time' in result and result['time'] != float('inf'):
                    speedup = baseline_time / result['time']
                    print(f"{name:20s}: {result['time']:8.3f}s  (speedup: {speedup:6.2f}x)")
                else:
                    print(f"{name:20s}: FAILED - {result.get('error', 'Unknown error')}")


def benchmark_numba_single_asset():
    """Benchmark single-asset GBM implementations."""

    # Test configuration
    config = SingleSimulationConfig(
        initial_index_value=100.0,
        mu=0.05,
        volatility=0.2,
        maturity=1.0,
        nb_simulations=10000,
        time_step_per_year=252
    )

    implementations = [
        ("Original", lambda cfg: simulate_geometric_brownian_motion(cfg, rng=42)),
        ("Numba Ultra Fast", lambda cfg: simulate_geometric_brownian_motion_numba_batch(cfg, rng=42)),
    ]

    results = {}

    print(f"\nSingle-Asset Benchmark:")
    print(f"Simulations: {config.nb_simulations:,}")
    print(f"Time steps: {int(config.time_step_per_year * config.maturity)}")
    print("-" * 70)

    for name, func in implementations:
        print(f"Running {name}...")

        # Warm-up for JIT compilation
        if "Numba" in name:
            small_config = SingleSimulationConfig(
                initial_index_value=100.0,
                mu=0.05,
                volatility=0.2,
                maturity=0.1)
            try:
                _ = func(small_config)
                print(f"  Warm-up completed")
            except Exception as e:
                print(f"  Warm-up failed: {e}")
                continue

        # Actual benchmark
        start_time = time.perf_counter()
        try:
            paths = func(config)
            end_time = time.perf_counter()

            runtime = end_time - start_time
            results[name] = {
                'time': runtime,
                'paths': paths,
                'mean_final': np.mean(paths[:, -1], axis=0)
            }
            print(f"  Time: {runtime:.3f}s")
            print(f"  Final prices (mean): {results[name]['mean_final']}")
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {'time': float('inf'), 'error': str(e)}

    # Performance comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    if results:
        valid_results = {k: v for k, v in results.items()
                         if 'time' in v and v['time'] != float('inf')}

        if valid_results:
            baseline_time = results['Original']['time'] if 'Original' in results else min(
                r['time'] for r in valid_results.values())

            for name, result in results.items():
                if 'time' in result and result['time'] != float('inf'):
                    speedup = baseline_time / result['time']
                    print(f"{name:20s}: {result['time']:8.3f}s  (speedup: {speedup:6.2f}x)")
                else:
                    print(f"{name:20s}: FAILED - {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    benchmark_cpu_implementations()
    benchmark_numba_single_asset()
