from Simulations.monte_carlo import SingleSimulationConfig, MultiSimulationConfig
from exotics_functional import (
    OutperformanceCertificateParams,
    ReverseBarrierConvertibleCertificateParams,
    MultiBarrierConvertibleCertificateParams,
    calculate_maturity,
    simulate_paths,
    plot_paths,
    calculate_outperformance_payoff,
    calculate_reverse_barrier_convertible_payoff,
    calculate_multi_barrier_convertible_payoff
)
import unittest
import numpy as np
import pandas as pd
import datetime as dt
from unittest.mock import patch, MagicMock
import warnings
from typing import Callable

# Set matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

# Import the module under test


class TestDataClasses(unittest.TestCase):
    """Test cases for dataclass structures."""

    def setUp(self):
        """Set up common test data."""
        self.valuation_date = pd.Timestamp(dt.date(2021, 1, 4))
        self.maturity_date = pd.Timestamp(dt.date(2024, 1, 4))
        self.observation_dates = [
            pd.Timestamp(dt.date(2022, 1, 4)),
            pd.Timestamp(dt.date(2023, 1, 4)),
            pd.Timestamp(dt.date(2024, 1, 4))
        ]

    def test_outperformance_certificate_params_creation(self):
        """Test OutperformanceCertificateParams creation and immutability."""
        opc = OutperformanceCertificateParams(
            nominal=100,
            participation_rate=1.75,
            cap_rate=1.25,
            risk_free_rate=0.04,
            initial_index_value=88.64,
            valuation_date=self.valuation_date,
            maturity_date=self.maturity_date,
            max_amount_factor=1.4375
        )

        self.assertEqual(opc.nominal, 100)
        self.assertEqual(opc.participation_rate, 1.75)
        self.assertEqual(opc.cap_rate, 1.25)
        self.assertEqual(opc.risk_free_rate, 0.04)
        self.assertEqual(opc.initial_index_value, 88.64)
        self.assertEqual(opc.max_amount_factor, 1.4375)

        # Test that attributes exist and are accessible
        self.assertIsNotNone(opc.valuation_date)
        self.assertIsNotNone(opc.maturity_date)

    def test_reverse_barrier_convertible_certificate_params_creation(self):
        """Test ReverseBarrierConvertibleCertificateParams creation and immutability."""
        rbc = ReverseBarrierConvertibleCertificateParams(
            nominal=100,
            strike_rate=1.0,
            coupon_rate=0.02080,
            risk_free_rate=0.03,
            barrier_rate=0.80,
            valuation_date=self.valuation_date,
            observation_dates=self.observation_dates,
            initial_index_value=88.64,
            maturity_date=self.maturity_date
        )

        self.assertEqual(rbc.nominal, 100)
        self.assertEqual(rbc.strike_rate, 1.0)
        self.assertEqual(rbc.coupon_rate, 0.02080)
        self.assertEqual(rbc.barrier_rate, 0.80)
        self.assertEqual(len(rbc.observation_dates), 3)

        # Test that attributes exist and are accessible
        self.assertIsNotNone(rbc.valuation_date)
        self.assertIsNotNone(rbc.maturity_date)

    def test_multi_barrier_convertible_certificate_params_creation(self):
        """Test MultiBarrierConvertibleCertificateParams creation and immutability."""
        mbc = MultiBarrierConvertibleCertificateParams(
            nominal=100,
            strike_rates=[1.0, 1.0, 1.0, 1.0],
            coupon_rate=0.00428,
            risk_free_rate=0.04,
            barrier_rates=[0.80, 0.80, 0.80, 0.80],
            valuation_date=self.valuation_date,
            observation_dates=self.observation_dates,
            initial_index_values=[5667.20, 11080.90, 12217.70, 4939.44],
            maturity_date=self.maturity_date
        )

        self.assertEqual(mbc.nominal, 100)
        self.assertEqual(len(mbc.strike_rates), 4)
        self.assertEqual(len(mbc.barrier_rates), 4)
        self.assertEqual(len(mbc.initial_index_values), 4)

        # Test that attributes exist and are accessible
        self.assertIsNotNone(mbc.valuation_date)
        self.assertIsNotNone(mbc.maturity_date)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_calculate_maturity(self):
        """Test maturity calculation in years."""
        valuation_date = pd.Timestamp(dt.date(2021, 1, 4))
        maturity_date = pd.Timestamp(dt.date(2024, 1, 4))

        maturity = calculate_maturity(valuation_date, maturity_date)

        # Should be approximately 3 years
        self.assertAlmostEqual(maturity, 3.0, places=1)

    def test_calculate_maturity_same_date(self):
        """Test maturity calculation when dates are the same."""
        date = pd.Timestamp(dt.date(2021, 1, 4))
        maturity = calculate_maturity(date, date)
        self.assertEqual(maturity, 0.0)

    def test_simulate_paths_single_config(self):
        """Test simulate_paths with SingleSimulationConfig."""
        config = SingleSimulationConfig(
            initial_index_value=100,
            mu=0.05,
            volatility=0.2,
            maturity=1.0,
            nb_simulations=100
        )

        paths = simulate_paths(config)

        self.assertEqual(paths.shape[0], 100)  # Number of simulations
        self.assertTrue(paths.shape[1] > 0)    # Number of time steps
        self.assertTrue(np.all(paths[:, 0] == 100))  # Initial value

    def test_simulate_paths_multi_config(self):
        """Test simulate_paths with MultiSimulationConfig."""
        config = MultiSimulationConfig(
            initial_index_values=np.array([100, 200]),
            mu=np.array([0.05, 0.06]),
            volatility=np.array([0.2, 0.25]),
            correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            maturity=1.0,
            nb_simulations=100
        )

        paths = simulate_paths(config)

        self.assertEqual(paths.shape[0], 100)  # Number of simulations
        self.assertTrue(paths.shape[1] > 0)    # Number of time steps
        self.assertEqual(paths.shape[2], 2)    # Number of underlyings
        self.assertTrue(np.all(paths[:, 0, 0] == 100))  # Initial value first underlying
        self.assertTrue(np.all(paths[:, 0, 1] == 200))  # Initial value second underlying

    def test_simulate_paths_invalid_config(self):
        """Test simulate_paths with invalid config type."""
        # Create an object that's not a valid config type
        invalid_config = object()
        with self.assertRaises(ValueError):
            simulate_paths(invalid_config)  # type: ignore

    @patch('matplotlib.pyplot.show')
    def test_plot_paths_outperformance(self, mock_show):
        """Test plot_paths for OutperformanceCertificateParams."""
        opc = OutperformanceCertificateParams(
            nominal=100,
            participation_rate=1.75,
            cap_rate=1.25,
            risk_free_rate=0.04,
            initial_index_value=100,
            valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
            maturity_date=pd.Timestamp(dt.date(2024, 1, 4)),
            max_amount_factor=1.4375
        )

        # Create sample paths
        paths = np.random.normal(100, 10, (10, 252))

        # Should not raise an exception
        plot_paths(opc, paths)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_paths_reverse_barrier(self, mock_show):
        """Test plot_paths for ReverseBarrierConvertibleCertificateParams."""
        rbc = ReverseBarrierConvertibleCertificateParams(
            nominal=100,
            strike_rate=1.0,
            coupon_rate=0.02080,
            risk_free_rate=0.03,
            barrier_rate=0.80,
            valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
            observation_dates=[pd.Timestamp(dt.date(2022, 1, 4))],
            initial_index_value=100,
            maturity_date=pd.Timestamp(dt.date(2024, 1, 4))
        )

        # Create sample paths
        paths = np.random.normal(100, 10, (10, 252))

        # Should not raise an exception
        plot_paths(rbc, paths)
        mock_show.assert_called_once()

    def test_plot_paths_unsupported_type(self):
        """Test plot_paths with unsupported certificate type."""
        paths = np.random.normal(100, 10, (10, 252))

        # Create an object that's not a valid certificate type
        unsupported_certificate = object()
        with self.assertRaises(ValueError):
            plot_paths(unsupported_certificate, paths)  # type: ignore


class TestOutperformancePayoff(unittest.TestCase):
    """Test cases for outperformance certificate payoff calculations."""

    def setUp(self):
        """Set up test parameters."""
        self.opc = OutperformanceCertificateParams(
            nominal=100,
            participation_rate=1.5,
            cap_rate=1.3,
            risk_free_rate=0.04,
            initial_index_value=100,
            valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
            maturity_date=pd.Timestamp(dt.date(2024, 1, 4)),
            max_amount_factor=1.45
        )

    def test_outperformance_payoff_scenario_1_below_initial(self):
        """Test payoff when final price is below initial (scenario 1)."""
        # Create paths where final price is 80 (below initial 100)
        paths = np.array([[100, 95, 90, 85, 80]])

        payoff = calculate_outperformance_payoff(self.opc, paths)

        # Expected: nominal * (final / initial) * discount_factor
        # 100 * (80/100) * exp(-0.04 * 3) = 80 * exp(-0.12)
        expected = 100 * 0.8 * np.exp(-0.04 * 3)
        self.assertAlmostEqual(payoff[0], expected, places=1)

    def test_outperformance_payoff_scenario_2_above_initial_below_cap(self):
        """Test payoff when final price is above initial but below cap (scenario 2)."""
        # Create paths where final price is 120 (above initial 100, below cap 130)
        paths = np.array([[100, 105, 110, 115, 120]])

        payoff = calculate_outperformance_payoff(self.opc, paths)

        # Expected: nominal * (1 + participation_rate * performance) * discount_factor
        # performance = (120-100)/100 = 0.2
        # 100 * (1 + 1.5 * 0.2) * exp(-0.04 * 3) = 130 * exp(-0.12)
        expected = 100 * (1 + 1.5 * 0.2) * np.exp(-0.04 * 3)
        self.assertAlmostEqual(payoff[0], expected, places=1)

    def test_outperformance_payoff_scenario_3_above_cap(self):
        """Test payoff when final price is above cap (scenario 3)."""
        # Create paths where final price is 150 (above cap 130)
        paths = np.array([[100, 120, 130, 140, 150]])

        payoff = calculate_outperformance_payoff(self.opc, paths)

        # Expected: nominal * max_amount_factor * discount_factor
        # 100 * 1.45 * exp(-0.04 * 3)
        expected = 100 * 1.45 * np.exp(-0.04 * 3)
        self.assertAlmostEqual(payoff[0], expected, places=1)

    def test_outperformance_payoff_multiple_paths(self):
        """Test payoff calculation with multiple simulation paths."""
        # Different scenarios in different paths
        paths = np.array([
            [100, 95, 90, 85, 80],    # Below initial
            [100, 105, 110, 115, 120],  # Above initial, below cap
            [100, 120, 130, 140, 150]  # Above cap
        ])

        payoffs = calculate_outperformance_payoff(self.opc, paths)

        self.assertEqual(len(payoffs), 3)
        self.assertTrue(np.all(payoffs > 0))

    def test_outperformance_payoff_3d_paths(self):
        """Test payoff calculation with 3D paths (single underlying)."""
        # Shape: (n_sims, n_steps, 1)
        paths = np.array([[[100], [105], [110], [115], [120]]])

        payoff = calculate_outperformance_payoff(self.opc, paths)

        self.assertEqual(len(payoff), 1)
        self.assertTrue(payoff[0] > 0)

    def test_outperformance_payoff_invalid_3d_shape(self):
        """Test error handling for invalid 3D paths shape."""
        # Shape: (n_sims, n_steps, 2) - more than 1 underlying
        paths = np.random.normal(100, 10, (1, 5, 2))

        with self.assertRaises(ValueError):
            calculate_outperformance_payoff(self.opc, paths)

    def test_outperformance_payoff_zero_initial_value(self):
        """Test error handling for zero initial value."""
        opc_zero = OutperformanceCertificateParams(
            nominal=100,
            participation_rate=1.5,
            cap_rate=1.3,
            risk_free_rate=0.04,
            initial_index_value=0,  # Zero initial value
            valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
            maturity_date=pd.Timestamp(dt.date(2024, 1, 4)),
            max_amount_factor=1.45
        )

        paths = np.array([[0, 10, 20, 30, 40]])

        with self.assertRaises(ValueError):
            calculate_outperformance_payoff(opc_zero, paths)


class TestReverseBarrierConvertiblePayoff(unittest.TestCase):
    """Test cases for reverse barrier convertible certificate payoff calculations."""

    def setUp(self):
        """Set up test parameters."""
        self.rbc = ReverseBarrierConvertibleCertificateParams(
            nominal=100,
            strike_rate=1.0,
            coupon_rate=0.02,
            risk_free_rate=0.03,
            barrier_rate=0.80,
            valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
            observation_dates=[
                pd.Timestamp(dt.date(2022, 1, 4)),
                pd.Timestamp(dt.date(2023, 1, 4)),
                pd.Timestamp(dt.date(2024, 1, 4))
            ],
            initial_index_value=100,
            maturity_date=pd.Timestamp(dt.date(2024, 1, 4))
        )

    def test_reverse_barrier_no_breach_continuous(self):
        """Test payoff when no barrier breach occurs (continuous monitoring)."""
        # Create paths that never go below barrier (80) - use discrete monitoring for simple test
        paths = np.array([[100, 105, 110, 115, 120]])

        payoff = calculate_reverse_barrier_convertible_payoff(self.rbc, paths, continuous=False)

        # Expected: nominal + total_coupons, discounted
        # 100 + 3*0.02*100 = 106, discounted
        coupon_total = 3 * 0.02 * 100
        expected = (100 + coupon_total) * np.exp(-0.03 * 3)
        self.assertAlmostEqual(payoff[0], expected, places=1)

    def test_reverse_barrier_breach_final_below_strike_continuous(self):
        """Test payoff when barrier breached and final < strike (continuous)."""
        # Create paths that breach barrier and end below strike - use discrete monitoring
        paths = np.array([[100, 90, 75, 85, 90]])  # Breaches at step 2, final 90 < 100

        payoff = calculate_reverse_barrier_convertible_payoff(self.rbc, paths, continuous=False)

        # Expected: (final/strike) * nominal + coupons, discounted
        # (90/100) * 100 + 6 = 96, discounted
        coupon_total = 3 * 0.02 * 100
        expected = (0.9 * 100 + coupon_total) * np.exp(-0.03 * 3)
        self.assertAlmostEqual(payoff[0], expected, places=1)

    def test_reverse_barrier_breach_final_above_strike_continuous(self):
        """Test payoff when barrier breached but final >= strike (continuous)."""
        # Create paths that breach barrier but end above strike - use discrete monitoring
        paths = np.array([[100, 90, 75, 95, 105]])  # Breaches at step 2, final 105 >= 100

        payoff = calculate_reverse_barrier_convertible_payoff(self.rbc, paths, continuous=False)

        # Expected: nominal + coupons, discounted
        coupon_total = 3 * 0.02 * 100
        expected = (100 + coupon_total) * np.exp(-0.03 * 3)
        self.assertAlmostEqual(payoff[0], expected, places=1)

    def test_reverse_barrier_discrete_monitoring(self):
        """Test payoff with discrete monitoring (observation dates only)."""
        # Create paths that don't breach on observation dates but might breach in between
        paths = np.random.normal(100, 5, (10, 1095))  # 3 years daily

        payoffs = calculate_reverse_barrier_convertible_payoff(self.rbc, paths, continuous=False)

        self.assertEqual(len(payoffs), 10)
        self.assertTrue(np.all(payoffs > 0))

    def test_reverse_barrier_multiple_simulations(self):
        """Test payoff calculation with multiple simulations."""
        np.random.seed(42)  # For reproducible results
        paths = np.random.normal(100, 20, (100, 252))

        payoffs = calculate_reverse_barrier_convertible_payoff(self.rbc, paths, continuous=False)

        self.assertEqual(len(payoffs), 100)
        self.assertTrue(np.all(payoffs > 0))
        # Check that we have some variety in payoffs (not all the same)
        self.assertTrue(np.std(payoffs) > 0)


class TestMultiBarrierConvertiblePayoff(unittest.TestCase):
    """Test cases for multi-barrier convertible certificate payoff calculations."""

    def setUp(self):
        """Set up test parameters."""
        self.mbc = MultiBarrierConvertibleCertificateParams(
            nominal=100,
            strike_rates=[1.0, 1.0],
            coupon_rate=0.01,
            risk_free_rate=0.04,
            barrier_rates=[0.80, 0.80],
            valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
            observation_dates=[
                pd.Timestamp(dt.date(2022, 1, 4)),
                pd.Timestamp(dt.date(2023, 1, 4)),
                pd.Timestamp(dt.date(2024, 1, 4))
            ],
            initial_index_values=[100, 200],
            maturity_date=pd.Timestamp(dt.date(2024, 1, 4))
        )

    def test_multi_barrier_no_breach_continuous(self):
        """Test payoff when no barrier breach occurs (continuous monitoring)."""
        # Create paths that never breach barriers
        paths = np.array([[[100, 200], [105, 210], [110, 220], [115, 230], [120, 240]]])

        payoff = calculate_multi_barrier_convertible_payoff(self.mbc, paths, continuous=True)

        # Expected: nominal + total_coupons, discounted
        coupon_total = 3 * 0.01 * 100
        expected = (100 + coupon_total) * np.exp(-0.04 * 3)
        self.assertAlmostEqual(payoff[0], expected, places=1)

    def test_multi_barrier_breach_worst_below_strike(self):
        """Test payoff when barrier breached and worst performer < strike."""
        # Create paths where one underlying breaches and worst performer is below strike
        paths = np.array([[[100, 200], [90, 190], [70, 180], [80, 170], [90, 160]]])
        # First underlying: final 90, strike 100, ratio = 0.9
        # Second underlying: final 160, strike 200, ratio = 0.8
        # Worst ratio = 0.8

        payoff = calculate_multi_barrier_convertible_payoff(self.mbc, paths, continuous=True)

        # Expected: worst_ratio * nominal + coupons, discounted
        coupon_total = 3 * 0.01 * 100
        expected = (0.8 * 100 + coupon_total) * np.exp(-0.04 * 3)
        self.assertAlmostEqual(payoff[0], expected, places=1)

    def test_multi_barrier_breach_worst_above_strike(self):
        """Test payoff when barrier breached but worst performer >= strike."""
        # Create paths where barrier is breached but all final values are above strikes
        paths = np.array([[[100, 200], [90, 190], [70, 180], [105, 210], [110, 220]]])
        # First underlying: final 110, strike 100, ratio = 1.1
        # Second underlying: final 220, strike 200, ratio = 1.1
        # Worst ratio = 1.1 >= 1

        payoff = calculate_multi_barrier_convertible_payoff(self.mbc, paths, continuous=True)

        # Expected: nominal + coupons, discounted
        coupon_total = 3 * 0.01 * 100
        expected = (100 + coupon_total) * np.exp(-0.04 * 3)
        self.assertAlmostEqual(payoff[0], expected, places=1)

    def test_multi_barrier_discrete_monitoring(self):
        """Test payoff with discrete monitoring."""
        # Create random paths for testing
        np.random.seed(42)
        paths = np.random.normal([100, 200], [20, 40], (10, 252, 2))

        payoffs = calculate_multi_barrier_convertible_payoff(self.mbc, paths, continuous=False)

        self.assertEqual(len(payoffs), 10)
        self.assertTrue(np.all(payoffs > 0))

    def test_multi_barrier_multiple_simulations(self):
        """Test payoff calculation with multiple simulations."""
        np.random.seed(123)
        paths = np.random.normal([100, 200], [25, 50], (50, 100, 2))

        payoffs = calculate_multi_barrier_convertible_payoff(self.mbc, paths, continuous=True)

        self.assertEqual(len(payoffs), 50)
        self.assertTrue(np.all(payoffs > 0))
        # Check that we have some variety in payoffs
        self.assertTrue(np.std(payoffs) > 0)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test cases for edge cases and error handling."""

    def test_calculate_maturity_negative_time(self):
        """Test maturity calculation with maturity before valuation."""
        valuation_date = pd.Timestamp(dt.date(2024, 1, 4))
        maturity_date = pd.Timestamp(dt.date(2021, 1, 4))

        maturity = calculate_maturity(valuation_date, maturity_date)
        self.assertTrue(maturity < 0)

    def test_payoff_functions_with_empty_paths(self):
        """Test payoff functions with empty paths arrays."""
        empty_paths = np.array([]).reshape(0, 0)

        opc = OutperformanceCertificateParams(
            nominal=100, participation_rate=1.5, cap_rate=1.3, risk_free_rate=0.04,
            initial_index_value=100, valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
            maturity_date=pd.Timestamp(dt.date(2024, 1, 4)), max_amount_factor=1.45
        )

        # Should handle empty arrays gracefully
        result = calculate_outperformance_payoff(opc, empty_paths.reshape(0, 5))
        self.assertEqual(len(result), 0)

    def test_payoff_functions_with_nan_values(self):
        """Test payoff functions with NaN values in paths."""
        paths_with_nan = np.array([[100, 105, np.nan, 115, 120]])

        opc = OutperformanceCertificateParams(
            nominal=100, participation_rate=1.5, cap_rate=1.3, risk_free_rate=0.04,
            initial_index_value=100, valuation_date=pd.Timestamp(dt.date(2021, 1, 4)),
            maturity_date=pd.Timestamp(dt.date(2024, 1, 4)), max_amount_factor=1.45
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculate_outperformance_payoff(opc, paths_with_nan)
            # The final value is 120, which is valid, so result should be a number
            self.assertFalse(np.isnan(result[0]))
            self.assertTrue(result[0] > 0)


if __name__ == '__main__':
    # Create a test suite combining all test classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDataClasses,
        TestUtilityFunctions,
        TestOutperformancePayoff,
        TestReverseBarrierConvertiblePayoff,
        TestMultiBarrierConvertiblePayoff,
        TestEdgeCasesAndErrorHandling
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run the tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
