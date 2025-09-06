import numpy as np

from exotics import (GeometricBrownianMotion,
                     HypersphereDecomposition,
                     MultiGeometricBrownianMotion,
                     MultiSimulationConfig, SimulationConfig)


def test_decomposition_returns_psd_and_unit_diag():
    mat = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.3], [0.7, 0.3, 1.0]])
    hd = HypersphereDecomposition(mat)
    psd = hd.decomposition(mat)

    # PSD check
    eigs = np.linalg.eigvalsh(psd)
    assert np.all(eigs >= -1e-12)

    # Unit diagonal
    assert np.allclose(np.diag(psd), 1.0, atol=1e-12)


def test_hypersphere_decomposition_reproducibility():
    mat = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.3], [0.7, 0.3, 1.0]])
    hd = HypersphereDecomposition(mat)
    psd1 = hd.decomposition(mat)
    psd2 = hd.decomposition(mat)

    assert np.allclose(psd1, psd2)


def test_hypersphere_optimization():
    target_matrix = np.array([[1.0, 0.9, 0.7],
                              [0.9, 1.0, 0.3],
                              [0.7, 0.3, 1.0]])
    # Optimize the correlation matrix
    optimized_matrix = HypersphereDecomposition(target_matrix)
    optimized_matrix.optimization()
    # Verify properties
    assert np.allclose(np.diag(optimized_matrix.optimization()), 1.0)
    assert HypersphereDecomposition.is_positive_definite(optimized_matrix.optimization())
    assert np.all(np.linalg.eigvals(optimized_matrix.optimization()) > 0)


def test_gbm_reproducibility():
    nb_sim = 10
    maturity = 0.01
    initial_value = 100
    vol = 0.1
    mu = 0.0

    config = SimulationConfig(
        initial_index_value=initial_value,
        mu=mu,
        volatility=vol,
        maturity=maturity,
        nb_simulations=nb_sim,
    )

    rng1 = np.random.default_rng(seed=456)
    rng2 = np.random.default_rng(seed=456)
    gbm = GeometricBrownianMotion()
    paths1 = gbm.simulate(config=config, rng=rng1)
    paths2 = gbm.simulate(config=config, rng=rng2)

    # Paths should be identical with same seed
    assert np.allclose(paths1, paths2)


def test_multi_gbm_shape_and_values():
    nb_sim = 20
    maturity = 0.02
    initial_values = [100.0, 120.0]
    vols = [0.1, 0.15]
    mu = [0.0, 0.0]
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])

    config = MultiSimulationConfig(
        initial_index_value=np.array(initial_values),
        mu=np.array(mu),
        volatility=np.array(vols),
        correlation_matrix=corr,
        maturity=maturity,
        nb_simulations=nb_sim,
    )

    rng = np.random.default_rng(seed=42)
    mgbm = MultiGeometricBrownianMotion()
    paths = mgbm.simulate(config=config, rng=rng)

    # shape (nb_simulations, time_steps+1, num_underlyings)
    assert paths.shape[0] == nb_sim
    assert paths.shape[2] == 2
    # initial values match
    assert np.allclose(paths[:, 0, 0], initial_values[0])
    assert np.allclose(paths[:, 0, 1], initial_values[1])


def test_multi_gbm_reproducibility():
    nb_sim = 10
    maturity = 0.01
    initial_values = [100.0, 120.0]
    vols = [0.1, 0.15]
    mu = [0.0, 0.0]
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])

    config = MultiSimulationConfig(
        initial_index_value=np.array(initial_values),
        mu=np.array(mu),
        volatility=np.array(vols),
        correlation_matrix=corr,
        maturity=maturity,
        nb_simulations=nb_sim,
    )

    rng1 = np.random.default_rng(seed=123)
    rng2 = np.random.default_rng(seed=123)
    mgbm = MultiGeometricBrownianMotion()
    paths1 = mgbm.simulate(config=config, rng=rng1)
    paths2 = mgbm.simulate(config=config, rng=rng2)

    # Paths should be identical with same seed
    assert np.allclose(paths1, paths2)
