# Testing Framework Setup

## Overview

A comprehensive unit testing framework has been set up for the thermal terrain simulation project using pytest. The tests serve as regression tests to ensure code changes don't break existing functionality.

## Test Structure

### Test Files Created

1. **tests/test_terrain.py** (16 tests)
   - Grid creation and dimensions
   - Normal vector computation for flat and sloped terrain
   - Unit vector validation
   - Sky view factor calculations
   - Geometric computations and sign conventions

2. **tests/test_materials.py** (15 tests)
   - Material class creation and properties
   - Material database loading from JSON
   - Material field assignment (uniform and heterogeneous)
   - Thermal diffusivity calculations
   - Physical property range validation

3. **tests/test_atmosphere.py** (20 tests)
   - Diurnal temperature and wind variation functions
   - Atmospheric conditions querying
   - Sky temperature models (Idso, Swinbank, Brunt)
   - Forced convection correlations
   - Natural convection correlations
   - Mixed convection behavior

4. **tests/test_solar.py** (19 tests)
   - Solar position calculations (azimuth, elevation)
   - Sun vector geometry
   - Day of year calculations
   - Clear sky irradiance models
   - Shadow cache functionality
   - Solar geometry symmetry

5. **tests/test_solver.py** (17 tests)
   - Subsurface grid creation and properties
   - Energy balance computation
   - Tridiagonal solver correctness
   - Energy conservation validation
   - Thermal solver integration

**Total: 87 unit tests**

## Configuration Files

### pytest.ini

Pytest configuration with:
- Test discovery patterns
- Output formatting options
- Test markers (unit, integration, slow, energy)

### requirements.txt

Updated with testing dependencies:
- pytest >= 7.4.0
- pytest-cov >= 4.1.0

## Installation

To use the testing framework:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Or install all requirements
pip install -r requirements.txt
```

## Running Tests

### Basic test execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific module
pytest tests/test_terrain.py
pytest tests/test_solver.py
```

### Coverage analysis

```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# View coverage in browser
# Open htmlcov/index.html
```

### Using markers

```bash
# Run only unit tests
pytest -m unit

# Run energy conservation tests
pytest -m energy

# Skip slow tests
pytest -m "not slow"
```

## Test Organization

### Debug Scripts

Debug scripts have been organized into [examples/debug/](examples/debug/):
- debug_energy.py
- debug_equilibrium.py
- debug_first_hour.py
- debug_solver.py
- debug_timing.py

### Test Coverage

The unit tests cover:

**Terrain Module:**
- Grid creation and validation
- Normal vector computation
- Sky view factors
- Geometric properties

**Materials Module:**
- Material database operations
- Property assignment to fields
- Thermal property calculations
- Validation of physical ranges

**Atmosphere Module:**
- Diurnal variation functions
- Atmospheric condition queries
- Convection heat transfer correlations
- Sky temperature models

**Solar Module:**
- Solar position astronomy
- Irradiance calculations
- Shadow caching and retrieval
- Geometric validations

**Solver Module:**
- Subsurface grid generation
- Energy balance calculations
- Tridiagonal system solution
- **Energy conservation (< 1e-4 relative error)**
- Full solver integration

## Energy Conservation Testing

The test suite includes rigorous energy conservation validation:

```python
@pytest.mark.energy
def test_energy_conservation_single_step():
    # Verifies:
    # dE_actual = ∫ ρ·cp·dT·dV
    # dE_expected = ∫ Q_net·dt·dA
    # |dE_actual - dE_expected| / |dE_expected| < 1e-4
```

This ensures the von Neumann boundary condition correctly conserves energy.

## Continuous Integration

Before committing changes:

```bash
# Run full test suite
pytest -v

# Verify all tests pass
# Check for any warnings
```

## Next Steps

To expand the testing framework:

1. **Integration tests**: Add tests that verify multiple modules working together
2. **Performance tests**: Add benchmarks for solver performance
3. **Validation tests**: Compare against analytical solutions for simple cases
4. **CI/CD**: Set up automated testing with GitHub Actions or similar
5. **Property-based testing**: Use hypothesis library for fuzzing tests

## Test Maintenance

- Update tests when module interfaces change
- Add tests for any new features
- Keep test data (materials JSON, etc.) synchronized with main data files
- Review and update test tolerances as needed
