# Unit Tests for Exotics_functional.py

## Overview

This document describes the comprehensive unit test suite created for the `exotics_functional.py` module. The test file `test_exotics_functional.py` provides thorough coverage of all the financial instrument calculations and utility functions.

## Test Structure

### 1. TestDataClasses

Tests the creation and immutability of the three main dataclass types:

- `OutperformanceCertificateParams`
- `ReverseBarrierConvertibleCertificateParams`
- `MultiBarrierConvertibleCertificateParams`

**Key Tests:**

- Proper instantiation with all required parameters
- Immutability enforcement (frozen dataclasses)
- Correct attribute access

### 2. TestUtilityFunctions

Tests core utility functions:

- `calculate_maturity()`: Time to maturity calculations
- `simulate_paths()`: Path simulation for both single and multi-asset configurations
- `plot_paths()`: Visualization functions (mocked to avoid display issues)

**Key Tests:**

- Maturity calculation accuracy
- Path simulation shape validation
- Support for both SingleSimulationConfig and MultiSimulationConfig
- Error handling for invalid configurations

### 3. TestOutperformancePayoff

Comprehensive tests for outperformance certificate payoff calculations:

**Scenarios Tested:**

1. **Below Initial Value**: Final price < initial price → proportional payoff
2. **Above Initial, Below Cap**: initial < final ≤ cap → participation in upside
3. **Above Cap**: final > cap → capped payoff at maximum amount

**Key Features:**

- Vectorized calculations verification
- 2D and 3D path array support
- Proper discounting validation
- Edge cases (zero initial value, NaN handling)

### 4. TestReverseBarrierConvertiblePayoff

Tests for reverse barrier convertible certificate calculations:

**Monitoring Types:**

- Continuous monitoring (all time steps)
- Discrete monitoring (observation dates only)

**Scenarios Tested:**

1. **No Barrier Breach**: Full nominal + coupons
2. **Breach + Final < Strike**: Proportional payoff + coupons
3. **Breach + Final ≥ Strike**: Full nominal + coupons

### 5. TestMultiBarrierConvertiblePayoff

Tests for multi-underlying barrier convertible certificates:

**Key Features:**

- Multiple underlying assets
- Worst-of performance logic
- Barrier monitoring across all assets
- Both continuous and discrete monitoring

**Scenarios Tested:**

1. **No Barrier Breach**: Full payoff
2. **Breach + Worst Performer < Strike**: Worst-of proportional payoff
3. **Breach + All Performers ≥ Strike**: Full payoff

### 6. TestEdgeCasesAndErrorHandling

Tests for boundary conditions and error scenarios:

- Negative time to maturity
- Empty path arrays
- NaN values in simulation paths
- Invalid input types

## Running the Tests

### Prerequisites

Since you're using `uv`, the required packages are already defined in your `pyproject.toml`:

- numpy
- pandas
- matplotlib
- numba
- pytest

### Running with uv (Recommended)

```bash
# From project root (will auto-discover tests in Exotics/)
cd c:\Users\renar\PythonProjects\PricingLib
uv run pytest

# Run specific test file with verbose output
uv run pytest Exotics/test_exotics_functional.py -v

# From Exotics directory
cd Exotics
uv run pytest test_exotics_functional.py -v
```

### Running with unittest

```bash
cd Exotics
uv run python -m unittest test_exotics_functional.py -v
```

### Running individual test classes

```bash
uv run pytest test_exotics_functional.py::TestOutperformancePayoff -v
```

## Test Coverage

The test suite provides comprehensive coverage including:

- ✅ All three certificate types
- ✅ All payoff calculation functions
- ✅ Utility functions
- ✅ Error handling and edge cases
- ✅ Both 2D and 3D path arrays
- ✅ Continuous and discrete monitoring
- ✅ Multiple simulation scenarios
- ✅ Proper discounting calculations

## Expected Test Results

When all tests pass, you should see output similar to:

```
test_calculate_maturity ... ok
test_outperformance_payoff_scenario_1_below_initial ... ok
test_reverse_barrier_no_breach_continuous ... ok
...
Ran XX tests in X.XXXs
OK
```

## Key Benefits

1. **Validation**: Ensures mathematical correctness of financial calculations
2. **Regression Testing**: Catches bugs when modifying existing code
3. **Documentation**: Tests serve as executable examples of expected behavior
4. **Confidence**: Provides assurance that complex financial logic works correctly
5. **Maintainability**: Makes refactoring safer and easier

## Notes

- Tests use mocked plotting functions to avoid GUI dependencies
- Random seeds are used where appropriate for reproducible results
- Type checking is handled gracefully with `# type: ignore` comments where needed
- Tests include both positive and negative scenarios for comprehensive coverage
