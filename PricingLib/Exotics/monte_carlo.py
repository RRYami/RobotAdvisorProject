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
class SingleSimulationConfig:
    initial_index_value: float
    mu: float
    volatility: float
    maturity: float
    nb_simulations: int = NB_SIMULATIONS
    time_step_per_year: float = TIME_STEPS_PER_YEAR


@dataclass
class MultiSimulationConfig:
    initial_index_values: np.ndarray
    mu: np.ndarray
    volatility: np.ndarray
    correlation_matrix: np.ndarray
    maturity: float
    nb_simulations: int = NB_SIMULATIONS
    time_step_per_year: float = TIME_STEPS_PER_YEAR

    def __post_init__(self):
        self.initial_index_values = np.asarray(self.initial_index_values)
        self.mu = np.asarray(self.mu)
        self.volatility = np.asarray(self.volatility)
        self.correlation_matrix = np.asarray(self.correlation_matrix)


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


def _prepare_simulation_config(config: MultiSimulationConfig, rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int, int]:
    """Extract and validate simulation parameters."""
    rng = rng or np.random.default_rng()
    dt = 1.0 / config.time_step_per_year
    time_steps = int(config.time_step_per_year * config.maturity)
    n_assets = len(config.initial_index_values)
    N = config.nb_simulations

    # Convert and validate market arrays
    S0 = np.asarray(config.initial_index_values, dtype=float).reshape((n_assets,))
    vol = np.asarray(config.volatility, dtype=float).reshape((n_assets,))
    mu = np.asarray(config.mu, dtype=float).reshape((n_assets,))

    # Validate and fix correlation matrix
    if not HypersphereDecomposition.is_positive_definite(config.correlation_matrix):
        config.correlation_matrix = HypersphereDecomposition(config.correlation_matrix).optimization()
    corr = np.asarray(config.correlation_matrix, dtype=float)

    # Precompute drift
    drift = (mu - 0.5 * (vol ** 2)) * dt

    return S0, vol, mu, corr, drift, dt, time_steps, n_assets, N


def _generate_qmc_normals(N: int, total_dims: int, rqmc_reps: int = 1) -> list[np.ndarray]:
    """Generate QMC normal variates using Sobol sequences."""
    runs = []
    m = int(np.ceil(np.log2(max(1, N))))

    for rep in range(max(1, rqmc_reps)):
        sob = qmc.Sobol(d=total_dims, scramble=True)
        U = sob.random_base2(m)[:N, :]
        U = np.clip(U, 1e-12, 1.0 - 1e-12)
        Z_flat = norm.ppf(U)  # shape (N, total_dims)
        runs.append(Z_flat)

    return runs


def _apply_full_pca(Z_flat: np.ndarray, corr: np.ndarray, time_steps: int,
                    n_assets: int, dt: float) -> np.ndarray:
    """Apply full PCA on time x asset covariance."""
    N = Z_flat.shape[0]

    # Precompute time covariance
    times = np.arange(1, time_steps + 1, dtype=float) * dt
    C_time = np.minimum.outer(times, times)  # Cov(W(t_i), W(t_j)) = min(t_i,t_j)

    # Full PCA on time x asset covariance
    Cov_full = np.kron(C_time, corr)  # shape (T*M, T*M)
    vals, vecs = eigh(Cov_full)
    order = np.argsort(vals)[::-1]
    vals_ord = vals[order]
    vecs_ord = vecs[:, order]
    A_full = vecs_ord @ np.diag(np.sqrt(vals_ord))

    # Map Z_flat -> W_flat (Brownian values at monitoring times)
    W_flat = Z_flat @ A_full.T
    W = W_flat.reshape((N, time_steps, n_assets))

    return W


def _apply_asset_pca(Z_flat: np.ndarray, corr: np.ndarray, time_steps: int,
                     n_assets: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply PCA on asset correlation matrix."""
    N = Z_flat.shape[0]

    # Asset PCA
    vals_a, vecs_a = eigh(corr)
    order_a = np.argsort(vals_a)[::-1]
    vals_a_ord = vals_a[order_a]
    vecs_a_ord = vecs_a[:, order_a]
    A_asset = vecs_a_ord @ np.diag(np.sqrt(vals_a_ord))

    # Reshape to principal component space
    Z_pc = Z_flat.reshape((N, time_steps, n_assets))

    return Z_pc, A_asset


def _apply_brownian_bridge_to_pcs(Z_pc: np.ndarray, time_steps: int,
                                  dt: float, n_assets: int) -> np.ndarray:
    """Apply Brownian bridge to principal components."""
    N = Z_pc.shape[0]
    times = np.arange(1, time_steps + 1, dtype=float) * dt
    time_order = brownian_bridge_ordering(time_steps)

    W_pc = np.empty_like(Z_pc)
    for p in range(n_assets):
        Z_time = Z_pc[:, :, p]
        Z_ordered = Z_time[:, time_order]
        W_p = brownian_bridge_from_ordered_normals(Z_ordered, times, time_order)
        W_pc[:, :, p] = W_p

    return W_pc


def _apply_simple_correlation(Z: np.ndarray, corr: np.ndarray, dt: float) -> np.ndarray:
    """Apply correlation using Cholesky decomposition."""
    L = np.linalg.cholesky(corr)
    asset_normals = Z @ L.T
    deltaW = asset_normals * np.sqrt(dt)
    W = np.cumsum(deltaW, axis=1)
    return W


def _apply_independent_brownian_bridge(Z: np.ndarray, corr: np.ndarray,
                                       time_steps: int, dt: float, n_assets: int) -> np.ndarray:
    """Apply Brownian bridge per asset then correlate."""
    N = Z.shape[0]
    times = np.arange(1, time_steps + 1, dtype=float) * dt
    time_order = brownian_bridge_ordering(time_steps)

    # Apply bridge per asset independently
    W_indep = np.empty_like(Z)
    for j in range(n_assets):
        Z_time = Z[:, :, j]
        Z_ordered = Z_time[:, time_order]
        W_j = brownian_bridge_from_ordered_normals(Z_ordered, times, time_order)
        W_indep[:, :, j] = W_j

    # Apply correlation
    L = np.linalg.cholesky(corr)
    W = np.einsum('ntj,jk->ntk', W_indep, L.T)
    return W


def _build_paths_from_brownian(W: np.ndarray, S0: np.ndarray, vol: np.ndarray,
                               drift: np.ndarray, time_steps: int) -> np.ndarray:
    """Convert Brownian motion to asset paths."""
    N, _, n_assets = W.shape

    # Compute increments
    deltaW = np.empty_like(W)
    deltaW[:, 0, :] = W[:, 0, :]
    deltaW[:, 1:, :] = W[:, 1:, :] - W[:, :-1, :]

    # Log increments
    log_increments = drift.reshape((1, 1, n_assets)) + (vol.reshape((1, 1, n_assets)) * deltaW)
    cum = np.cumsum(log_increments, axis=1)

    # Build paths
    paths = np.empty((N, time_steps + 1, n_assets), dtype=float)
    paths[:, 0, :] = S0.reshape((1, n_assets))
    paths[:, 1:, :] = S0.reshape((1, 1, n_assets)) * np.exp(cum)

    return paths


# Single-asset Geometric Brownian Motion simulation
def simulate_geometric_brownian_motion(config: SingleSimulationConfig, rng: np.random.Generator | None = None) -> np.ndarray:
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
def simulate_multi_geometric_brownian_motion(config: MultiSimulationConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    dt = 1 / config.time_step_per_year
    time_steps = int(config.time_step_per_year * config.maturity)
    num_underlyings = len(config.initial_index_values)
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
    paths[:, 0, :] = config.initial_index_values

    for t in range(1, time_steps + 1):
        # Generate correlated random shocks
        z = rng.standard_normal((config.nb_simulations, num_underlyings))
        correlated_z = z @ L.T

        for u in range(num_underlyings):
            drift = (config.mu[u] - 0.5 * config.volatility[u] ** 2) * dt
            diffusion = config.volatility[u] * np.sqrt(dt) * correlated_z[:, u]
            paths[:, t, u] = paths[:, t - 1, u] * np.exp(drift + diffusion)

    return paths


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


def simulate_multi_geometric_brownian_motion_robust(
        config: MultiSimulationConfig,
        rng: Optional[np.random.Generator] = None,
        use_qmc: bool = True,
        use_pca: bool = True,
        use_full_pca: bool = False,
        use_brownian_bridge: bool = False,
        rqmc_reps: int = 1) -> np.ndarray:

    # Validate incompatible options
    if use_full_pca and use_brownian_bridge:
        raise ValueError("use_brownian_bridge is not supported with use_full_pca")

    # Setup
    S0, vol, mu, corr, drift, dt, time_steps, n_assets, N = _prepare_simulation_config(config)
    total_dims = time_steps * n_assets
    rng = rng or np.random.default_rng()

    # Pseudo-random fallback
    if not use_qmc:
        Z = rng.standard_normal((N, time_steps, n_assets))
        W = _apply_simple_correlation(Z, corr, dt)
        return _build_paths_from_brownian(W, S0, vol, drift, time_steps)

    # QMC path
    Z_runs = _generate_qmc_normals(N, total_dims, rqmc_reps)
    path_runs = []

    for Z_flat in Z_runs:
        if use_full_pca:
            W = _apply_full_pca(Z_flat, corr, time_steps, n_assets, dt)

        elif use_pca and use_brownian_bridge:
            Z_pc, A_asset = _apply_asset_pca(Z_flat, corr, time_steps, n_assets)
            W_pc = _apply_brownian_bridge_to_pcs(Z_pc, time_steps, dt, n_assets)
            W = W_pc @ A_asset.T

        elif use_pca:
            Z_pc, A_asset = _apply_asset_pca(Z_flat, corr, time_steps, n_assets)
            asset_normals = Z_pc @ A_asset.T
            deltaW = asset_normals * np.sqrt(dt)
            W = np.cumsum(deltaW, axis=1)

        elif use_brownian_bridge:
            Z = Z_flat.reshape((N, time_steps, n_assets))
            W = _apply_independent_brownian_bridge(Z, corr, time_steps, dt, n_assets)

        else:
            Z = Z_flat.reshape((N, time_steps, n_assets))
            W = _apply_simple_correlation(Z, corr, dt)

        paths = _build_paths_from_brownian(W, S0, vol, drift, time_steps)
        path_runs.append(paths)

    return path_runs[0] if len(path_runs) == 1 else np.stack(path_runs, axis=0)
