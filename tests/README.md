# Unit Tests

This directory contains unit tests for the thermal terrain simulation modules.

## Running Tests

### Install test dependencies

```bash
pip install pytest pytest-cov
```

### Run all tests

```bash
pytest
```

### Run tests with verbose output

```bash
pytest -v
```

### Run tests for a specific module

```bash
pytest tests/test_terrain.py
pytest tests/test_materials.py
pytest tests/test_atmosphere.py
pytest tests/test_solar.py
pytest tests/test_solver.py
```

### Run tests with coverage report

```bash
pytest --cov=src --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run only tests with specific markers

```bash
pytest -m unit          # Run only unit tests
pytest -m energy        # Run only energy conservation tests
pytest -m "not slow"    # Skip slow tests
```

## Test Organization

- **test_terrain.py**: Tests for terrain grid, normals, slopes, sky view factors
- **test_materials.py**: Tests for material database and material field assignment
- **test_atmosphere.py**: Tests for atmospheric conditions and convection correlations
- **test_solar.py**: Tests for solar position, irradiance, and shadow calculations
- **test_solver.py**: Tests for subsurface grid, energy balance, and thermal solver

## Test Markers

- `@pytest.mark.unit`: Basic unit tests for individual functions
- `@pytest.mark.integration`: Integration tests across modules
- `@pytest.mark.slow`: Tests that take significant time
- `@pytest.mark.energy`: Energy conservation validation tests

## Continuous Integration

These tests serve as regression tests to ensure that code changes don't break existing functionality. Run tests before committing changes:

```bash
pytest -v
```

All tests should pass before merging changes.
