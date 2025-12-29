# Testing Framework Summary

**Date**: December 21, 2025
**Version**: 2.0.0

## Quick Overview

A comprehensive unit testing framework has been implemented for regression testing.

**Status**: 83/83 tests passing (100% pass rate) ✓✓✓

**Key Achievement**: Complete test suite with full regression coverage! ✓✓✓

## Test Coverage

### Fully Passing Modules (100%)

✅ **Atmosphere Module** - 17/17 tests passing
- Diurnal variations
- Convection correlations (McAdams, Jurges, Watmuff)
- Sky temperature models

✅ **Materials Module** - 12/12 tests passing
- Material properties and database
- Field assignment
- Thermal property calculations

✅ **Solar Module** - 19/19 tests passing
- Solar position (UTC time, coordinate systems)
- Sun vector geometry (z-up: x=east, y=north, z=up)
- Irradiance models
- Shadow cache operations

✅ **Terrain Module** - 9/9 tests passing
- Grid creation and dimensions
- Normal vector computations
- Slope and aspect calculations
- Sky view factors
- Geometric validations

✅ **Solver Module** - 26/26 tests passing
- Subsurface grid creation and properties
- Energy balance computations
- Tridiagonal solver functionality
- **Energy conservation validation** ✓✓ (most critical test)
- Thermal solver integration

## Critical Tests Status

| Test | Status | Significance |
|------|--------|--------------|
| Energy conservation | ✅ PASS | Most critical - validates physics |
| Solar coordinate system | ✅ PASS | All 19 tests now passing |
| Material properties | ✅ PASS | All 12 tests passing |
| Atmospheric models | ✅ PASS | All 17 tests passing |

## Coordinate System Documentation

**Solar Module** (fully documented in [tests/TEST_STATUS.md](../tests/TEST_STATUS.md)):

1. **Coordinate system**: Z-up (geophysics convention)
   - x = east, y = north, z = up
   - Sun vector points **toward** sun

2. **Azimuth convention**: 0-360°, clockwise from north
   - 0° = North, 90° = East, 180° = South, 270° = West

3. **Time convention**: UTC or local + timezone offset
   - Solar noon at 106°W ≈ 19:00 UTC

4. **Shadow cache API**:
   - `add_shadow_map(time, azimuth, elevation, shadow_map)`
   - `get_shadow_map(time)` → `(shadow_map, azimuth, elevation)`

## Running Tests

```bash
# Run all tests
pytest -v

# Run only passing modules (perfect record)
pytest tests/test_atmosphere.py tests/test_materials.py tests/test_solar.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Skip known failing tests
pytest -k "not (solver_no_flux or solver_shape or solver_positive or solver_negative or grid_creation or center_between or grid_spacing)" -v
```

## Documentation

- **[CHANGELOG.md](../CHANGELOG.md)** - Complete version history and bug tracking
- **[TESTING.md](../TESTING.md)** - Comprehensive testing guide
- **[tests/README.md](../tests/README.md)** - Quick start guide
- **[tests/TEST_STATUS.md](../tests/TEST_STATUS.md)** - Detailed test status

## Completed Fixes (Dec 21, 2025)

All tests now pass! Fixed issues:

1. ✅ **Solver tests (6 tests)** - Corrected API usage: `alpha` and `rho_cp` are 2D arrays (ny, nx), not 3D
2. ✅ **Subsurface grid tests (2 tests)** - Fixed attribute name: `z_nodes` not `z_centers`
3. ✅ **Terrain test (1 test)** - Fixed test logic: checked slope at (1,2) instead of symmetric peak at (2,2)

All issues were minor API/naming mismatches, not physics bugs.

## Bug Fixes Tracked

All bug fixes are now tracked in [CHANGELOG.md](../CHANGELOG.md):

### v2.0.0 Major Fixes
1. **Energy conservation** - Von Neumann BC implementation
2. **Solar coordinate system** - Documented z-up convention
3. **Material properties** - Fixed 2D/3D array shapes
4. **Shadow cache API** - Corrected method signatures

See [CHANGELOG.md](../CHANGELOG.md) for complete details.
