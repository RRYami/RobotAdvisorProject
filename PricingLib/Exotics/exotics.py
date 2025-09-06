from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from scipy.linalg import eigh, eigvalsh
from scipy.optimize import minimize
from scipy.stats import norm, qmc

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
)


# hypersphere decomposition
class HypersphereDecomposition:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    @staticmethod
    def is_positive_definite(matrix: np.ndarray, tol: float = 1e-12) -> bool:
        """
        Check if a matrix is (numerically) positive semi-definite.

        Uses the symmetric eigenvalues (via `eigvalsh`) and allows a small negative
        tolerance to account for numerical noise.
        """
        eigs = eigvalsh(matrix)
        # np.all returns a numpy.bool_, cast to Python bool for consistent return type
        return bool(np.all(eigs > -tol))

    @staticmethod
    def objective(opti_matrix: np.ndarray, target_matrix: np.ndarray):
        """
        Objective function to minimize: Frobenius norm of the difference between
        the correlation matrix and the target matrix.
        """
        # Reshape the matrix for the optimazation
        matrix = opti_matrix.reshape(target_matrix.shape)
        # Symmetrize the matrix
        sym_matrix = (matrix + matrix.T) / 2

        return np.sum((sym_matrix - target_matrix) ** 2)

    @staticmethod
    def decomposition(matrix) -> np.ndarray:
        """
        Decomposes the correlation matrix using hypersphere decomposition.
        Parameters:
        - correlation_matrix (np.ndarray): The correlation matrix to decompose.

        Returns:
        - np.ndarray: The correlation matrix reconstructed from hypersphere decomposition.
        """

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Only keep positive eigenvalues (zero for negative ones)
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        # Reconstruct the matrix using the adjusted eigenvalues and eigenvectors
        psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Normalize to ensure unit diagonal
        norms = np.sqrt(np.diag(psd_matrix))
        psd_matrix = psd_matrix / (norms[:, None] * norms[None, :])

        return psd_matrix

    def optimization(self, initial_guess=None):
        """
        Optimize the correlation matrix to resemble the target matrix, while ensuring the matrix is PSD.

        Parameters:
        - target_matrix (np.ndarray): The target (non-PSD) matrix you want to approximate.
        - initial_guess (np.ndarray): The starting guess for the optimization.

        Returns:
        - np.ndarray: The optimized correlation matrix.
        """
        # If no initial guess, start with the target matrix itself
        if initial_guess is None:
            initial_guess = self.matrix

        # Flatten the initial guess matrix to use in optimization
        initial_guess_flat = initial_guess.flatten()

        # Minimize the objective function
        result = minimize(self.objective, initial_guess_flat, args=(self.matrix,),
                          method='trust-constr', options={'disp': False})

        # Reshape the result back into a matrix
        optimized_matrix = result.x.reshape(self.matrix.shape)

        # Project the result onto the space of valid correlation matrices (PSD + unit diagonal)
        optimized_matrix = self.decomposition(optimized_matrix)

        return optimized_matrix


# Abstract Class for Simulation methods
class PriceSimulation(ABC):
    @abstractmethod
    def simulate(self, **params) -> np.ndarray:
        pass


# Strategy Base Class
class PayoffStrategy(ABC):
    # Simulation Configuration Class
    @abstractmethod
    def calculate_payoff(self, rbc, paths):
        raise NotImplementedError("Payoff calculation not implemented")


@dataclass(slots=True)
class SimulationConfig:
    initial_index_value: float = 100.0
    mu: float = 0.0
    volatility: float = 0.2
    maturity: float = 1.0
    nb_simulations: int = 10000


class MultiSimulationConfig:
    def __init__(self, initial_index_value: np.ndarray, mu: np.ndarray,
                 volatility: np.ndarray, correlation_matrix: np.ndarray, maturity: float, nb_simulations=10000):
        self.initial_index_value = np.asarray(initial_index_value)
        self.mu = np.asarray(mu)
        self.volatility = np.asarray(volatility)
        self.correlation_matrix = np.asarray(correlation_matrix)
        self.maturity = maturity
        self.nb_simulations = nb_simulations


# Geometric Brownian Motion Simulation Class
class GeometricBrownianMotion(PriceSimulation):
    def simulate(self, config: SimulationConfig, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        dt = 1 / 365.25
        time_steps = int(config.maturity * 365.25)
        paths = np.zeros((config.nb_simulations, time_steps + 1))
        paths[:, 0] = config.initial_index_value
        z = rng.standard_normal((time_steps, config.nb_simulations))
        drift = (config.mu - 0.5 * config.volatility ** 2) * dt
        diffusion = config.volatility * np.sqrt(dt) * z
        log_returns = drift + diffusion
        cum_log_returns = np.cumsum(log_returns, axis=0)
        paths[:, 1:] = config.initial_index_value * np.exp(cum_log_returns.T)
        return paths


# Multi Geometric Brownian Motion Simulation Class
class MultiGeometricBrownianMotion(PriceSimulation):
    def simulate(self, config: MultiSimulationConfig, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        dt = 1 / 365.25
        time_steps = int(365.25 * config.maturity)
        num_underlyings = len(config.initial_index_value)
        # Check for semi-positive definiteness (correlation matrix)
        if not HypersphereDecomposition.is_positive_definite(config.correlation_matrix):
            try:
                config.correlation_matrix = HypersphereDecomposition(config.correlation_matrix).optimization()
            except Exception as e:
                raise ValueError("Error in the Decomposition") from e
        # Cholesky decomposition of the correlation matrix
        L = np.linalg.cholesky(config.correlation_matrix)
        # Initialize paths
        paths = np.zeros((config.nb_simulations, time_steps + 1, num_underlyings))
        paths[:, 0, :] = config.initial_index_value

        for t in range(1, time_steps + 1):
            # Generate correlated random shocks
            z = rng.standard_normal((config.nb_simulations, num_underlyings))
            correlated_z = z @ L.T

            for u in range(num_underlyings):
                drift = (config.mu[u] - 0.5 * config.volatility[u] ** 2) * dt
                diffusion = config.volatility[u] * np.sqrt(dt) * correlated_z[:, u]
                paths[:, t, u] = paths[:, t - 1, u] * np.exp(drift + diffusion)

        return paths


# Multi Geometric Brownian Motion Simulation Class TEST
class MultiGeometricBrownianMotionRobust(PriceSimulation):
    """
    Monte-Carlo simulator for multi-asset geometric Brownian motion with optional
    RQMC (scrambled Sobol), PCA mapping, full time×asset PCA, and Brownian-bridge.

    Options
    -------
    - use_qmc: toggle scrambled Sobol (RQMC) vs pseudo-random.
    - use_pca: use per-time-step PCA on the asset correlation matrix (fast).
    - use_full_pca: use PCA on the full time x asset Brownian covariance (more powerful;
      eigen-decomposition size = (T*M)^2). If True, `use_pca` is ignored.
    - use_brownian_bridge: apply Brownian-bridge ordering on the *time* dimension.
      This is supported together with per-time-step PCA (use_pca=True) but is not
      combined with use_full_pca (raise error if both requested).
    - rqmc_reps: number of independent scramblings (if >1 returns stacked runs).

    Returns
    -------
    - If use_qmc is False: (N, T+1, M)
    - If use_qmc is True and rqmc_reps == 1: (N, T+1, M)
    - If use_qmc is True and rqmc_reps > 1: (rqmc_reps, N, T+1, M)
    """

    def simulate(self,
                 config: 'MultiSimulationConfig',
                 rng: Optional[np.random.Generator] = None,
                 use_qmc: bool = True,
                 use_pca: bool = True,
                 use_full_pca: bool = False,
                 use_brownian_bridge: bool = False,
                 rqmc_reps: int = 1,) -> np.ndarray:

        rng = rng or np.random.default_rng()
        dt = 1.0 / 365.25
        time_steps = int(365.25 * config.maturity)
        n_assets = int(len(config.initial_index_value))
        N = int(config.nb_simulations)

        # Convert market arrays
        S0 = np.asarray(config.initial_index_value, dtype=float).reshape((n_assets,))
        vol = np.asarray(config.volatility, dtype=float).reshape((n_assets,))
        mu = np.asarray(config.mu, dtype=float).reshape((n_assets,))

        # Validate correlation
        if not HypersphereDecomposition.is_positive_definite(config.correlation_matrix):
            config.correlation_matrix = HypersphereDecomposition(config.correlation_matrix).optimization()
        corr = np.asarray(config.correlation_matrix, dtype=float)

        # Precompute drift
        drift = (mu - 0.5 * (vol ** 2)) * dt

        total_dims = time_steps * n_assets

        # Helper to build paths given Brownian values at monitoring times W(t)
        # W shape: (N, time_steps, n_assets) containing Brownian value at each monitoring time
        def build_paths_from_W(W: np.ndarray) -> np.ndarray:
            # compute increments Delta W
            deltaW = np.empty_like(W)
            deltaW[:, 0, :] = W[:, 0, :]
            deltaW[:, 1:, :] = W[:, 1:, :] - W[:, :-1, :]
            # log increments: drift + vol * deltaW
            log_increments = drift.reshape((1, 1, n_assets)) + (vol.reshape((1, 1, n_assets)) * deltaW)
            cum = np.cumsum(log_increments, axis=1)
            paths = np.empty((N, time_steps + 1, n_assets), dtype=float)
            paths[:, 0, :] = S0.reshape((1, n_assets))
            paths[:, 1:, :] = S0.reshape((1, 1, n_assets)) * np.exp(cum)
            return paths

        # --------------------------
        # Pseudo-random branch
        # --------------------------
        if not use_qmc:
            # generate independent standard normal increments per asset per time
            Z = rng.standard_normal((N, time_steps, n_assets))
            # Apply asset correlation per time-step (L from corr)
            L = np.linalg.cholesky(corr)
            asset_normals = Z @ L.T
            # Convert normals (which represent N(0,1)) to Brownian increments
            # deltaW = sqrt(dt) * normals, but we'll directly treat these as deltaW
            # Note: In pseudo branch we follow the previous convention: normals*sqrt(dt)
            deltaW = asset_normals * np.sqrt(dt)
            # Build W(t) by cumulative sum
            W = np.cumsum(deltaW, axis=1)
            return build_paths_from_W(W)

        # --------------------------
        # QMC branch: handle Full PCA or per-time PCA (+ optional Brownian bridge)
        # --------------------------
        if use_full_pca and use_brownian_bridge:
            raise ValueError("use_brownian_bridge is not supported together with use_full_pca."
                             " Use per-time PCA + bridge or full PCA without bridge.")

        # Precompute times and time covariance matrix for Brownian motion
        times = np.arange(1, time_steps + 1, dtype=float) * dt
        C_time = np.minimum.outer(times, times)  # Cov(W(t_i), W(t_j)) = min(t_i,t_j)

        runs = []
        m = int(np.ceil(np.log2(max(1, N))))

        for rep in range(max(1, rqmc_reps)):
            sob = qmc.Sobol(d=total_dims, scramble=True)
            U = sob.random_base2(m)[:N, :]
            U = np.clip(U, 1e-12, 1.0 - 1e-12)
            Z_flat = norm.ppf(U)  # shape (N, total_dims)

            if use_full_pca:
                use_pca = False  # ignore use_pca if full_pca requested
                # Full PCA on time x asset covariance
                Cov_full = np.kron(C_time, corr)  # shape (T*M, T*M)
                vals, vecs = eigh(Cov_full)
                order = np.argsort(vals)[::-1]
                vals_ord = vals[order]
                vecs_ord = vecs[:, order]
                A_full = vecs_ord @ np.diag(np.sqrt(vals_ord))   # maps std normals -> W_flat
                # Map Z_flat -> W_flat (Brownian values at monitoring times concatenated)
                W_flat = Z_flat @ A_full.T   # shape (N, total_dims)
                W = W_flat.reshape((N, time_steps, n_assets))
                paths = build_paths_from_W(W)
                runs.append(paths)
                continue

            # else: per-time-step PCA (asset-wise PCA) possibly combined with Brownian bridge
            if use_pca:
                vals_a, vecs_a = eigh(corr)
                order_a = np.argsort(vals_a)[::-1]
                vals_a_ord = vals_a[order_a]
                vecs_a_ord = vecs_a[:, order_a]
                A_asset = vecs_a_ord @ np.diag(np.sqrt(vals_a_ord))  # (n_assets, n_assets)
                # interpret Z_flat as PC normals per (time, pc)
                Z_pc = Z_flat.reshape((N, time_steps, n_assets))  # (N, T, M) in time-major order

                if use_brownian_bridge:
                    # Build time-ordering for bridge (midpoint refinement)
                    time_order = self._brownian_bridge_ordering(time_steps)

                    # For each principal component p, apply Brownian bridge across time
                    W_pc = np.empty_like(Z_pc)
                    for p in range(n_assets):
                        Z_time = Z_pc[:, :, p]  # shape (N, T) in natural time order
                        # reorder columns to consumption (bridge) order
                        Z_ordered = Z_time[:, time_order]
                        # produce Brownian values at monitoring times from ordered normals
                        W_p = self._brownian_bridge_from_ordered_normals(Z_ordered, times, time_order)
                        W_pc[:, :, p] = W_p

                    # Map PC Brownian values to asset Brownian values via A_asset^T
                    # asset_W[n,t,j] = sum_p W_pc[n,t,p] * A_asset[j,p]
                    # Compute PC Brownian values to asset Brownian values via A_asset^T
                    W = W_pc @ A_asset.T
                    paths = build_paths_from_W(W)
                    runs.append(paths)

                else:
                    # No bridge: treat Z_pc as independent standard normals per time & PC, map to asset normals
                    # Correct mapping: asset_normals = Z_pc @ A_asset.T
                    asset_normals = Z_pc @ A_asset.T
                    # asset_normals are standard normals; convert to Brownian increments deltaW = sqrt(dt)*normals
                    deltaW = asset_normals * np.sqrt(dt)
                    W = np.cumsum(deltaW, axis=1)
                    paths = build_paths_from_W(W)
                    runs.append(paths)

            else:
                # No PCA: simple per-time Cholesky on asset corr, then Brownian bridge on each asset
                # Interpret Z_flat as normals in natural time order
                Z = Z_flat.reshape((N, time_steps, n_assets))
                if use_brownian_bridge:
                    time_order = self._brownian_bridge_ordering(time_steps)
                    # Apply bridge per asset (independent), then apply asset correlation via L per time
                    W_indep = np.empty_like(Z)
                    for j in range(n_assets):
                        Z_time = Z[:, :, j]
                        Z_ordered = Z_time[:, time_order]
                        W_j = self._brownian_bridge_from_ordered_normals(Z_ordered, times, time_order)
                        W_indep[:, :, j] = W_j
                    # Now correlate assets at each time: for each time t, mix vector W_indep[:,t,:] by L
                    L = np.linalg.cholesky(corr)
                    W = np.einsum('ntj,jk->ntk', W_indep, L.T)
                    paths = build_paths_from_W(W)
                    runs.append(paths)
                else:
                    L = np.linalg.cholesky(corr)
                    asset_normals = Z @ L.T
                    deltaW = asset_normals * np.sqrt(dt)
                    W = np.cumsum(deltaW, axis=1)
                    paths = build_paths_from_W(W)
                    runs.append(paths)

        # Return single run or stacked runs
        if len(runs) == 1:
            return runs[0]
        return np.stack(runs, axis=0)

    # ----------------------
    # Utility helper methods
    # ----------------------
    def _brownian_bridge_ordering(self, n: int) -> np.ndarray:
        """Return an ordering of indices [0..n-1] using midpoint refinement.
        This order is used to consume QMC dimensions in the Brownian-bridge construction.
        """
        order = []

        def visit(a, b):
            if a > b:
                return
            m = (a + b) // 2
            order.append(m)
            visit(a, m - 1)
            visit(m + 1, b)
        visit(0, n - 1)
        return np.array(order, dtype=int)

    def _brownian_bridge_from_ordered_normals(self, Z_ordered: np.ndarray, times: np.ndarray,
                                              time_order: np.ndarray) -> np.ndarray:
        """
        Given normals arranged in "bridge consumption order" (Z_ordered shape (N,T)),
        construct Brownian motion values at monitoring times using the Brownian-bridge
        conditional sampling formula. Returns W in "natural time order" shape (N,T).

        Z_ordered[:, k] is the normal variable consumed at the k-th step of the bridge.
        time_order[k] is the corresponding time index in natural order.
        """
        N, T = Z_ordered.shape
        # Prepare output (in natural order)
        W = np.full((N, T), np.nan)
        assigned = [False] * T
        assigned_idx = []

        # For quick access
        t = times

        for k in range(T):
            idx = int(time_order[k])
            z_k = Z_ordered[:, k]
            if k == 0:
                # First assigned point: W(t_idx) = sqrt(t_idx) * z
                W[:, idx] = np.sqrt(t[idx]) * z_k
                assigned[idx] = True
                assigned_idx.append(idx)
                assigned_idx.sort()
                continue

            # Find nearest assigned times to the left and right of idx
            # assigned_idx is sorted
            pos = np.searchsorted(assigned_idx, idx)
            # left exists?
            if pos == 0:
                left = None
                right = assigned_idx[0]
            elif pos == len(assigned_idx):
                left = assigned_idx[-1]
                right = None
            else:
                left = assigned_idx[pos - 1]
                right = assigned_idx[pos]

            if left is None:
                # only right known -> conditional on W(right)
                # For Brownian motion starting at 0, conditioning on future alone gives:
                # W(t) | W(b)=wb ~ Normal( (t/b) wb, (t*(b-t)/b) )
                b = t[right]
                mean = (t[idx] / b) * W[:, right]
                var = (t[idx] * (b - t[idx])) / b
                W[:, idx] = mean + np.sqrt(var) * z_k
            elif right is None:
                # only left known -> conditional on W(left)
                a = t[left]
                mean = W[:, left]  # since future unknown, but for bridge we rarely hit this
                # variance = (t-a) * ??? -> better to condition on left and treat as increment
                # However with midpoint refinement there will usually be both neighbors
                var = t[idx] - a
                W[:, idx] = mean + np.sqrt(var) * z_k
            else:
                a = t[left]
                b = t[right]
                # conditional mean: linear interpolation between W[a] and W[b]
                factor = (t[idx] - a) / (b - a)
                mean = W[:, left] + factor * (W[:, right] - W[:, left])
                var = (t[idx] - a) * (b - t[idx]) / (b - a)
                W[:, idx] = mean + np.sqrt(var) * z_k

            assigned[idx] = True
            assigned_idx.append(idx)
            assigned_idx.sort()

        return W


# Base Outperformance Exotic.
class BaseOutperformanceCertificate:
    def __init__(self, nominal, participation_rate, cap_rate, risk_free_rate,
                 max_amount_rate, valuation_date, maturity_date, payoff_strategy):
        self.nominal = nominal
        self.participation_rate = participation_rate
        self.cap_rate = cap_rate
        self.risk_free_rate = risk_free_rate
        self.max_amount_factor = max_amount_rate
        self.valuation_date = valuation_date
        self.maturity = (maturity_date - valuation_date).days / 365.25
        self.payoff_strategy = payoff_strategy

    def set_simulation_method(self, simulation_method: PriceSimulation):
        self.simulation_method = simulation_method

    def calculate_payoff(self, paths: np.ndarray, **kwargs) -> np.ndarray:
        return self.payoff_strategy.calculate_payoff(self, paths, **kwargs)


# Outperformance Exotic Class
class OutperformanceCertificate(BaseOutperformanceCertificate):
    def __init__(self, nominal, participation_rate, cap_rate, risk_free_rate, volatility, mu,
                 initial_index_value, valuation_date, maturity_date, max_amount_rate, payoff_strategy):
        super().__init__(nominal, participation_rate, cap_rate, risk_free_rate,
                         max_amount_rate, valuation_date, maturity_date, payoff_strategy)
        self.volatility = volatility
        self.mu = mu
        self.initial_index_value = initial_index_value
        self.cap_level = initial_index_value * cap_rate
        self.max_amount_value = nominal * max_amount_rate

    def simulate_paths(self, rng: np.random.Generator | None = None) -> np.ndarray:
        config = SimulationConfig(
            initial_index_value=self.initial_index_value,
            mu=self.mu,
            volatility=self.volatility,
            maturity=self.maturity,
            nb_simulations=10000)
        return self.simulation_method.simulate(config=config, rng=rng)

    def plot_paths(self, paths: np.ndarray):
        plt.figure(figsize=(10, 6))

        for path_idx, path in enumerate(paths[:1]):  # Plot first 10 paths for clarity
            plt.plot(path, alpha=0.6, label=f'Path {path_idx + 1}' if path_idx == 0 else "")

            # Detect first breach only
            cap_hit = np.where((path[:-1] >= self.cap_level) & (path[1:] < self.cap_level))[0] + 1
            if len(cap_hit) > 0:
                plt.scatter(
                    cap_hit, path[cap_hit], color='red',
                    label='Cap Breach' if path_idx == 0 else "", s=5
                )

        # Add cap levels
        plt.axhline(y=self.cap_level, color='r', linestyle='--', label='Cap Level')

        # Finalize plot
        plt.xlabel('Time (days)')
        plt.ylabel('Index Level')
        plt.title('Simulated Index Paths with Cap Breaches')
        plt.legend()
        plt.grid(True)
        plt.show()


# Base Reverse Barrier Convertible Class
class BaseReverseBarrierConvertible:
    def __init__(self, nominal, strike, coupon_rate, risk_free_rate, valuation_date: pd.Timestamp, observation_dates: list[str], payoff_strategy):
        self.nominal = nominal
        self.strike = strike
        self.coupon_rate = coupon_rate
        self.risk_free_rate = risk_free_rate
        self.valuation_date = valuation_date
        self.observation_dates = observation_dates
        self.observation_dates_days = [
            (pd.Timestamp(date) - self.valuation_date).days for date in self.observation_dates]
        self.payoff_strategy = payoff_strategy
        self.maturity = (pd.Timestamp(self.observation_dates[-1]) - self.valuation_date).days / 365.25

    def set_simulation_method(self, simulation_method: PriceSimulation):
        self.simulation_method = simulation_method

    def calculate_payoff(self, paths: np.ndarray, **kwargs) -> np.ndarray:
        return self.payoff_strategy.calculate_payoff(self, paths, **kwargs)


# Reverse Barrier Convertible Class
class ReverseBarrierConvertible(BaseReverseBarrierConvertible):
    def __init__(self, nominal, barrier_level, strike, coupon_rate, risk_free_rate, volatility, mu,
                 initial_index_value, observation_dates, valuation_date, payoff_strategy):
        super().__init__(nominal, strike, coupon_rate, risk_free_rate, valuation_date, observation_dates, payoff_strategy)
        self.barrier_level = barrier_level
        self.volatility = volatility
        self.mu = mu
        self.risk_free_rate = risk_free_rate
        self.initial_index_value = initial_index_value
        self.barrier_value = barrier_level * initial_index_value
        self.strike_value = strike * initial_index_value

    def simulate_paths(self, rng: np.random.Generator | None = None) -> np.ndarray:
        config = SimulationConfig(
            initial_index_value=self.initial_index_value,
            mu=self.mu,
            volatility=self.volatility,
            maturity=self.maturity,
            nb_simulations=10000)
        return self.simulation_method.simulate(config=config, rng=rng)

    def plot_paths(self, paths: np.ndarray):
        plt.figure(figsize=(10, 6))

        for path_idx, path in enumerate(paths[:1]):  # Plot first path for clarity
            plt.plot(path, alpha=0.6, label=f'Path {path_idx + 1}' if path_idx == 0 else "")

            # Detect first breach only
            breach_indices = np.where((path[:-1] >= self.barrier_value) & (path[1:] < self.barrier_value))[0] + 1
            if len(breach_indices) > 0:
                plt.scatter(
                    breach_indices, path[breach_indices], color='red',
                    label='Barrier Breach' if path_idx == 0 else "", s=5
                )

        # Add barrier and strike levels
        plt.axhline(y=self.barrier_value, color='r', linestyle='--', label='Barrier Level')
        plt.axhline(y=self.strike_value, color='g', linestyle='-.', label='Strike Level')

        # Mark observation dates
        for obs_date in self.observation_dates_days:
            plt.axvline(x=obs_date, color='C2', alpha=0.5,
                        label='Observation Date' if obs_date == self.observation_dates_days[0] else "")

        # Finalize plot
        plt.xlabel('Time (days)')
        plt.ylabel('Index Level')
        plt.title('Simulated Index Paths with First Barrier Breaches')
        plt.legend()
        plt.grid(True)
        plt.show()


# Multi Barrier Reverse Convertible Class
class MultiBarrierReverseConvertible(BaseReverseBarrierConvertible):
    def __init__(self, nominal, barriers, strike, coupon_rate, risk_free_rate, volatilities, mu,
                 initial_index_values, correlation_matrix, valuation_date, observation_dates, payoff_strategy):
        super().__init__(nominal, strike, coupon_rate, risk_free_rate,
                         valuation_date, observation_dates, payoff_strategy)
        self.barriers = barriers
        self.volatilities = volatilities
        self.mu = mu
        self.initial_index_values = initial_index_values
        self.correlation_matrix = correlation_matrix
        self.barrier_values = [b * iv for b, iv in zip(barriers, initial_index_values)]
        # strike * min(initial_index_values)
        self.strike_value = [b * iv for b, iv in zip(strike, initial_index_values)]

    def simulate_paths(self, rng: np.random.Generator | None = None, **kwargs) -> np.ndarray:
        config = MultiSimulationConfig(
            initial_index_value=self.initial_index_values,
            mu=self.mu,
            volatility=self.volatilities,
            correlation_matrix=self.correlation_matrix,
            maturity=self.maturity,
            nb_simulations=10000)
        return self.simulation_method.simulate(config=config, rng=rng, **kwargs)

    def plot_paths(self, paths: np.ndarray):
        plt.figure(figsize=(12, 8))
        num_underlyings = paths.shape[2]

        for underlying in range(num_underlyings):
            plt.subplot(num_underlyings, 1, underlying + 1)
            plt.title(f"Underlying {underlying + 1}")

            for path_idx, path in enumerate(paths[:10, :, underlying]):
                plt.plot(path, alpha=0.6)

                # Detect first breach
                breach_indices = np.where((path[:-1] >= self.barrier_values[underlying]) &
                                          (path[1:] < self.barrier_values[underlying]))[0] + 1
                if len(breach_indices) > 0:
                    plt.scatter(breach_indices, path[breach_indices], color='red', s=10)

            for observation_date in self.observation_dates_days:
                plt.axvline(x=observation_date, color='C1', linestyle='--', alpha=0.5,
                            label='Observation Date' if observation_date == self.observation_dates_days[0] else "")

            plt.axhline(y=self.barrier_values[underlying], color='red', linestyle='--', label='Barrier Level')
            plt.axhline(y=self.strike_value[underlying], color='green', linestyle='--', label='Strike Level')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()


class OutperformancePayoffStrategy(PayoffStrategy):
    def calculate_payoff(self, opc: OutperformanceCertificate, paths: np.ndarray, **kwargs) -> np.ndarray:
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
        nominal = float(opc.nominal)
        participation = float(opc.participation_rate)
        cap_rate = float(opc.cap_rate)
        cap_level = initial * cap_rate
        maturity_time = float(opc.maturity)
        r = float(opc.risk_free_rate)
        max_amount_rate = float(opc.max_amount_factor)

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


# Continuous Monitoring Payoff Strategy
class RBCContinuousPayoffStrategy(PayoffStrategy):
    def calculate_payoff(self, rbc: ReverseBarrierConvertible, paths: np.ndarray) -> np.ndarray:
        nb_simulations = paths.shape[0]
        maturity_time = rbc.maturity
        payoff = np.zeros(nb_simulations)

        # Vectorized barrier breach check
        barrier_breached = np.any(paths < rbc.barrier_value, axis=1)
        final_prices = paths[:, -1]

        # Vectorized payoff calculation
        coupon_total = len(rbc.observation_dates) * rbc.coupon_rate * rbc.nominal
        full_payoff = rbc.nominal + coupon_total

        # No breach: full payoff
        payoff[~barrier_breached] = full_payoff

        # Breach and final < strike: proportional + coupon
        breach_and_low = barrier_breached & (final_prices < rbc.strike_value)
        payoff[breach_and_low] = (final_prices[breach_and_low] / rbc.strike_value) * rbc.nominal + coupon_total

        # Breach and final >= strike: full payoff
        breach_and_high = barrier_breached & (final_prices >= rbc.strike_value)
        payoff[breach_and_high] = full_payoff

        discounted_payoff = payoff * np.exp(-rbc.risk_free_rate * maturity_time)
        return discounted_payoff


# Discrete Monitoring Payoff Strategy
class RBCDiscretePayoffStrategy(PayoffStrategy):
    def calculate_payoff(self, rbc: ReverseBarrierConvertible, paths: np.ndarray) -> np.ndarray:
        nb_simulations, n_steps = paths.shape
        payoff = np.zeros(nb_simulations)

        # Vectorized barrier breach check at observation dates
        observed_paths = paths[:, rbc.observation_dates_days]
        barrier_breached = np.any(observed_paths < rbc.barrier_value, axis=1)
        final_prices = paths[:, -1]

        # Vectorized payoff calculation
        coupon_total = len(rbc.observation_dates) * rbc.coupon_rate * rbc.nominal
        full_payoff = rbc.nominal + coupon_total

        # No breach: full payoff
        payoff[~barrier_breached] = full_payoff

        # Breach and final < strike: proportional + coupon
        breach_and_low = barrier_breached & (final_prices < rbc.strike_value)
        payoff[breach_and_low] = (final_prices[breach_and_low] / rbc.strike_value) * rbc.nominal + coupon_total

        # Breach and final >= strike: full payoff
        breach_and_high = barrier_breached & (final_prices >= rbc.strike_value)
        payoff[breach_and_high] = full_payoff

        maturity_time = rbc.observation_dates_days[-1] / 252
        discounted_payoff = payoff * np.exp(-rbc.risk_free_rate * maturity_time)
        return discounted_payoff


# Continuous Monitoring Multi Underlying Payoff Strategy
class RBCContinuousMultiUnderlyingPayoffStrategy(PayoffStrategy):
    """Payoff strategy for continuous monitoring of multi-underlying assets.

    Args:
        PayoffStrategy (ABC): Abstract base class for payoff strategies.
    Returns:
        np.ndarray: Discounted payoff per simulation."""

    def calculate_payoff(self, rbc: MultiBarrierReverseConvertible, paths: np.ndarray) -> np.ndarray:
        """
        Calculate discounted payoff for a Multi-Barrier Reverse Convertible (worst-of) with continuous monitoring.
        Parameters
        ----------
        rbc : MultiBarrierReverseConvertible
            The structured product definition (barriers, strikes, coupon, etc.)
        paths : np.ndarray
            Simulated price paths of shape (n_sims, n_steps, n_underlyings).
        Returns
        -------
        np.ndarray
            Discounted payoff per simulation.
        """
        nb_simulations = paths.shape[0]
        payoff = np.zeros(nb_simulations)
        maturity_time = rbc.maturity

        # Vectorized barrier breach check for each underlying
        barrier_breached_per_sim = np.zeros((nb_simulations, len(rbc.barriers)), dtype=bool)
        barrier_breached_per_sim = np.any(paths < np.array(rbc.barrier_values), axis=1)
        barrier_breached = np.any(barrier_breached_per_sim, axis=1)
        # for u in range(len(rbc.barriers)):
        #     barrier_breached_per_sim[:, u] = np.any(paths[:, :, u] < rbc.barrier_values[u], axis=1)
        # barrier_breached = np.any(barrier_breached_per_sim, axis=1)

        # Final prices for all underlyings
        final_prices = paths[:, -1, :]  # Shape: (nb_simulations, num_underlyings)

        # Worst-of logic: min(final_prices / strike_values) per simulation
        worst_final_price = np.min(final_prices / np.array(rbc.strike_value), axis=1)

        # Vectorized payoff calculation
        coupon_total = len(rbc.observation_dates) * rbc.coupon_rate * rbc.nominal
        full_payoff = rbc.nominal + coupon_total

        # No breach: full payoff
        payoff[~barrier_breached] = full_payoff

        # Breach and worst_final_price < 1: proportional + coupon
        breach_and_low = barrier_breached & (worst_final_price < 1)
        payoff[breach_and_low] = worst_final_price[breach_and_low] * rbc.nominal + coupon_total

        # Breach and worst_final_price >= 1: full payoff
        breach_and_high = barrier_breached & (worst_final_price >= 1)
        payoff[breach_and_high] = full_payoff

        discounted_payoff = payoff * np.exp(-rbc.risk_free_rate * maturity_time)
        return discounted_payoff


class RBCDiscreteMultiUnderlyingPayoffStrategy(PayoffStrategy):
    """Payoff strategy for discrete monitoring of multi-underlying assets.

    Args:
        PayoffStrategy (ABC): Abstract base class for payoff strategies.
    Returns:
        np.ndarray: Discounted payoff per simulation.
    """

    def calculate_payoff(self, rbc: MultiBarrierReverseConvertible,
                         paths: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """
        Calculate discounted payoff for a Multi-Barrier Reverse Convertible (worst-of) with discrete monitoring.

        Parameters
        ----------
        rbc : MultiBarrierReverseConvertible
            The structured product definition (barriers, strikes, coupon, etc.)
        paths : np.ndarray
            Simulated price paths of shape (n_sims, n_steps, n_underlyings).
        time_grid : np.ndarray
            Time discretization array, same length as paths' second dimension.

        Returns
        -------
        np.ndarray
            Discounted payoff per simulation.
        """
        n_sims, n_steps, _ = paths.shape
        maturity_time = rbc.maturity

        # --- Select observation indices ---
        if hasattr(rbc, "observation_dates") and rbc.observation_dates is not None:
            obs_indices = np.searchsorted(time_grid, rbc.observation_dates, side='right') - 1
        else:
            obs_indices = np.arange(n_steps)  # fallback: monitor all time steps

        # --- Barrier check (vectorized) ---
        # Shape: (n_sims, len(obs_indices), n_underlyings)
        obs_prices = paths[:, obs_indices, :]
        barrier_breached_per_sim = np.any(obs_prices < np.array(rbc.barrier_values), axis=1)
        barrier_breached = np.any(barrier_breached_per_sim, axis=1)

        # --- Final prices ---
        final_prices = paths[:, -1, :]  # (n_sims, n_underlyings)
        worst_final_price = np.min(final_prices / np.array(rbc.strike_value), axis=1)

        # --- Payoff logic ---
        coupon_total = len(rbc.observation_dates) * rbc.coupon_rate * rbc.nominal
        full_payoff = rbc.nominal + coupon_total
        payoff = np.full(n_sims, full_payoff)

        # Breach & worst < 1 → proportional + coupon
        breach_and_low = barrier_breached & (worst_final_price < 1)
        payoff[breach_and_low] = worst_final_price[breach_and_low] * rbc.nominal + coupon_total

        # Breach & worst ≥ 1 → full payoff (already default)
        # Breach and worst_final_price >= 1: full payoff
        breach_and_high = barrier_breached & (worst_final_price >= 1)
        payoff[breach_and_high] = full_payoff
        # --- Discounted payoff ---
        discounted_payoff = payoff * np.exp(-rbc.risk_free_rate * maturity_time)
        return discounted_payoff


# Usage
if __name__ == "__main__":
    ###########################################################
    # TEST 0 Single Barrier Reverse Convertible
    ###########################################################

    rbc = ReverseBarrierConvertible(
        nominal=100,
        barrier_level=0.8,
        strike=1,
        coupon_rate=0.02080,
        risk_free_rate=0.05,
        volatility=0.279,
        mu=0.03,
        initial_index_value=32.39,
        valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
        observation_dates=["2021-04-04", "2021-07-04", "2021-10-04",
                           "2022-01-04", "2022-04-04", "2022-07-04", "2022-10-04",
                           "2023-01-04", "2023-04-04", "2023-07-04", "2023-10-04"],
        payoff_strategy=RBCContinuousPayoffStrategy()
    )

    rbc.set_simulation_method(GeometricBrownianMotion())
    simulated_paths = rbc.simulate_paths()
    # rbc.plot_paths(simulated_paths)
    payoffs = rbc.calculate_payoff(simulated_paths)
    rbc.plot_paths(simulated_paths)
    print("\n")
    print("-----------------------TEST Single Underlying BRC-----------------------")
    print("------------------------------------------------------------------------")
    print("Payout Paths:", payoffs)
    print("Average RBC Price:", np.mean(payoffs))

    ##########################################################
    # TEST 1 Multi Barrier Reverse Convertible
    ##########################################################
    nominal = 100
    barriers = [0.5, 0.5, 0.5, 0.5]  # 50% of initial index values
    strike = [1.0, 1.0, 1.0, 1.0]  # 100% of the lowest initial index value
    coupon_rate = 0.00428
    risk_free_rate = 0.04
    volatilities = [0.10, 0.14, 0.15, 0.18]  # Different volatilities for each underlying
    underlyings_mu = [0.10, 0.15, 0.16, 0.17]  # Different dividend yields
    initial_index_values = [5667.20, 11080.90, 12217.70, 4939.44]  # Different initial values for each underlying
    correlation_matrix = np.array([
        [1.0, 0.51, 0.49, 0.57],
        [0.51, 1.0, 0.71, 0.87],
        [0.49, 0.71, 1.0, 0.81],
        [0.57, 0.87, 0.81, 1.0]])  # Correlated underlyings
    maturity = 1.95  # in years
    observation_dates = ["2021-04-04", "2021-07-04", "2021-10-04",
                         "2022-01-04", "2022-04-04", "2022-07-04", "2022-10-04",
                         "2023-01-04", "2023-04-04", "2023-07-04", "2023-10-04"]

    mbrc = MultiBarrierReverseConvertible(
        nominal=nominal,
        barriers=barriers,
        strike=strike,
        coupon_rate=coupon_rate,
        risk_free_rate=risk_free_rate,
        volatilities=volatilities,
        mu=underlyings_mu,
        initial_index_values=initial_index_values,
        correlation_matrix=correlation_matrix,
        valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
        observation_dates=observation_dates,
        payoff_strategy=RBCDiscreteMultiUnderlyingPayoffStrategy()
    )
    mbrc2 = MultiBarrierReverseConvertible(
        nominal=nominal,
        barriers=barriers,
        strike=strike,
        coupon_rate=coupon_rate,
        risk_free_rate=risk_free_rate,
        volatilities=volatilities,
        mu=underlyings_mu,
        initial_index_values=initial_index_values,
        correlation_matrix=correlation_matrix,
        valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
        observation_dates=observation_dates,
        payoff_strategy=RBCDiscreteMultiUnderlyingPayoffStrategy()
    )
    mbrc.set_simulation_method(MultiGeometricBrownianMotion())
    mbrc2.set_simulation_method(MultiGeometricBrownianMotionRobust())
    mbrc_paths = mbrc.simulate_paths(rng=None)
    mbrc2_paths = mbrc2.simulate_paths(rng=None, use_pca=True, use_brownian_bridge=True)
    mbrc.plot_paths(mbrc_paths)
    mbrc_payoffs = mbrc.calculate_payoff(mbrc_paths, time_grid=np.linspace(0, maturity, mbrc_paths.shape[1]))
    mbrc2_payoffs = mbrc2.calculate_payoff(mbrc2_paths, time_grid=np.linspace(0, maturity, mbrc2_paths.shape[1]))

    # mbrc_payoffs.tofile('mbrc_payoffs2.csv', sep=',')
    # print("Payout Path 1:", mbrc_payoffs[0])
    print("\n")
    print("---------------------------TEST RBC---------------------------")
    print("--------------------------------------------------------------")
    print("Payout Paths Multi Geom Test:", mbrc_payoffs)
    print("Average MBRC Price Multi Geom Test:", np.mean(mbrc_payoffs))
    print("Payout Paths Multi Geom:", mbrc2_payoffs)
    print("Average MBRC Price Multi Geom:", np.mean(mbrc2_payoffs))

    # ###########################################################
    # # TEST 2 outperformance certificate
    # ###########################################################
    opc = OutperformanceCertificate(
        nominal=100,
        participation_rate=1.75,
        cap_rate=1.25,
        risk_free_rate=0.04,
        volatility=0.34,
        mu=0.05,
        initial_index_value=88.64,
        valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
        maturity_date=pd.Timestamp(dt.date(2024, 1, 4)),
        max_amount_rate=1.4375,
        payoff_strategy=OutperformancePayoffStrategy()
    )
    opc.set_simulation_method(GeometricBrownianMotion())
    opc_paths = opc.simulate_paths(rng=None)
    opc_discounted_payoffs = opc.calculate_payoff(opc_paths)
    print("\n")
    print("-----------------TEST Outperformance Certificate-----------------")
    print("-----------------------------------------------------------------")
    print("Payout Paths Discounted:", opc_discounted_payoffs)
    print("Average OPC Price Discounted:", np.mean(opc_discounted_payoffs))
    opc.plot_paths(opc_paths)
