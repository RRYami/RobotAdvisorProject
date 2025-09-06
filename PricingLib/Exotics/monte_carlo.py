from scipy.linalg import eigh, eigvalsh
from scipy.optimize import minimize
from scipy.stats import norm, qmc
import numpy as np
from dataclasses import dataclass
from typing import Optional


TIME_STEPS_PER_YEAR = 365.25
NB_SIMULATIONS = 10000


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


@dataclass(frozen=True)
class SimulationConfig:
    initial_index_value: float = 100.0
    mu: float = 0.0
    volatility: float = 0.2
    maturity: float = 1.0
    nb_simulations: int = NB_SIMULATIONS
    time_step_per_year: float = TIME_STEPS_PER_YEAR


@dataclass
class MultiSimulationConfig:
    initial_index_value: np.ndarray
    mu: np.ndarray
    volatility: np.ndarray
    correlation_matrix: np.ndarray
    maturity: float
    nb_simulations: int = NB_SIMULATIONS
    time_step_per_year: float = TIME_STEPS_PER_YEAR

    def __post_init__(self):
        self.initial_index_value = np.asarray(self.initial_index_value)
        self.mu = np.asarray(self.mu)
        self.volatility = np.asarray(self.volatility)
        self.correlation_matrix = np.asarray(self.correlation_matrix)


# Single-asset Geometric Brownian Motion simulation
def simulate_geometric_brownian_motion(config: SimulationConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    dt = 1 / config.time_step_per_year
    time_steps = int(config.maturity * config.time_step_per_year)
    paths = np.zeros((config.nb_simulations, time_steps + 1))
    paths[:, 0] = config.initial_index_value
    z = rng.standard_normal((time_steps, config.nb_simulations))
    drift = (config.mu - 0.5 * config.volatility ** 2) * dt
    diffusion = config.volatility * np.sqrt(dt) * z
    log_returns = drift + diffusion
    cum_log_returns = np.cumsum(log_returns, axis=0)
    paths[:, 1:] = config.initial_index_value * np.exp(cum_log_returns.T)
    return paths


# Multi-asset Geometric Brownian Motion simulation
def simulate_multi_geometric_brownian_motion(self, config: MultiSimulationConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    dt = 1 / config.time_step_per_year
    time_steps = int(config.maturity * config.time_step_per_year)
    paths = np.zeros((config.nb_simulations, time_steps + 1))
    paths[:, 0] = config.initial_index_value
    z = rng.standard_normal((time_steps, config.nb_simulations))
    drift = (config.mu - 0.5 * config.volatility ** 2) * dt
    diffusion = config.volatility * np.sqrt(dt) * z
    log_returns = drift + diffusion
    cum_log_returns = np.cumsum(log_returns, axis=0)
    paths[:, 1:] = config.initial_index_value * np.exp(cum_log_returns.T)
    return paths


def simulate_multi_geometric_brownian_motion_robust(
        config: MultiSimulationConfig,
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
                time_order = brownian_bridge_ordering(time_steps)

                # For each principal component p, apply Brownian bridge across time
                W_pc = np.empty_like(Z_pc)
                for p in range(n_assets):
                    Z_time = Z_pc[:, :, p]  # shape (N, T) in natural time order
                    # reorder columns to consumption (bridge) order
                    Z_ordered = Z_time[:, time_order]
                    # produce Brownian values at monitoring times from ordered normals
                    W_p = brownian_bridge_from_ordered_normals(Z_ordered, times, time_order)
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
                time_order = brownian_bridge_ordering(time_steps)
                # Apply bridge per asset (independent), then apply asset correlation via L per time
                W_indep = np.empty_like(Z)
                for j in range(n_assets):
                    Z_time = Z[:, :, j]
                    Z_ordered = Z_time[:, time_order]
                    W_j = brownian_bridge_from_ordered_normals(Z_ordered, times, time_order)
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


def brownian_bridge_ordering(n: int) -> np.ndarray:
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


def brownian_bridge_from_ordered_normals(Z_ordered: np.ndarray, times: np.ndarray,
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
