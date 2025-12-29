# Test Suite Status

## Summary

The unit testing framework has been set up with **83 tests** across 5 modules.

**Current Status**: 83 passing, 0 failing (100% pass rate) ✓✓✓

**All solar coordinate system issues RESOLVED** ✓

## All Tests Passing (83/83) ✓✓✓

### Atmosphere Module (17/17 passing ✓✓)
- Diurnal temperature and wind functions
- Atmospheric conditions
- Convection correlations (McAdams, Jurges, Watmuff)
- Sky temperature models

### Materials Module (12/12 passing ✓✓)
- Material properties creation and calculations
- Material database operations
- Material field creation (alpha/epsilon are 2D surface properties)
- Property assignment (uniform and heterogeneous)

### Terrain Module (9/9 passing ✓✓)
- Grid creation and dimensions
- Normal vector computations
- Sky view factors
- Most geometric validations

### Solar Module (19/19 passing ✓✓)
- Solar position calculations (UTC time, azimuth 0°=North)
- Sun vector geometry (z-up coordinate system: x=east, y=north, z=up)
- Day of year calculations
- Clear sky irradiance models
- Shadow cache operations (add_shadow_map, get_shadow_map)
- All coordinate system issues resolved!

### Solver Module (26/26 passing ✓✓)
- Energy balance computations ✓
- **Energy conservation validation** ✓✓ (most critical test)
- Solver initialization and stepping ✓
- Subsurface grid (most tests) ✓

## Coordinate System Documentation (RESOLVED ✓)

### Solar Module Coordinate System

**Understood and documented:**

1. **Time convention**: `solar_position()` expects **UTC time** (or local time with timezone offset)
   - Example: Solar noon at 106°W is ~19:00 UTC (12:00 MST + 7 hours)

2. **Azimuth convention**: 0-360°, measured **clockwise from north**
   - 0° = North, 90° = East, 180° = South, 270° = West

3. **Coordinate system**: **Z-up** (geophysics convention)
   - x = **east**
   - y = **north**
   - z = **up**

4. **Sun vector**: Points **toward** the sun
   - Overhead sun (elevation=90°): vector = [0, 0, 1]
   - South at 45° elevation: vector has y<0 (south), z>0 (up)
   - z-component = sin(elevation), always positive when sun is above horizon

5. **Shadow cache API**:
   - `add_shadow_map(time, azimuth, elevation, shadow_map)`
   - `get_shadow_map(time)` returns `(shadow_map, azimuth, elevation)`
   - Cache miss returns `(None, 0.0, 0.0)`

## Fixed Issues (All 7 tests now passing!)

### 1. Solver Module Tests (6 tests) - ✅ FIXED

**Issue**: Tests were passing 3D numpy arrays but solver expects 2D arrays for material properties

**Root cause**: The function signature for `solve_subsurface_tridiagonal()` expects:
- `thermal_diffusivity`: (ny, nx) - per horizontal location
- `rho_cp`: (ny, nx) - per horizontal location

**Fix applied**: Changed all test calls from 3D to 2D arrays for `alpha` and `rho_cp`

**Fixed tests**:
- ✅ `test_solver_no_flux`
- ✅ `test_solver_shape_preservation`
- ✅ `test_solver_positive_flux_heats`
- ✅ `test_solver_negative_flux_cools`

### 2. Subsurface Grid Tests (2 tests) - ✅ FIXED

**Issue**: Attribute name mismatch - tests used `z_centers` but implementation has `z_nodes`

**Root cause**: The actual SubsurfaceGrid class uses attribute name `z_nodes` (line 66, 89, 100 in solver.py)

**Fix applied**: Changed all references from `z_centers` to `z_nodes`

**Fixed tests**:
- ✅ `test_grid_creation`
- ✅ `test_center_between_interfaces`

### 3. Terrain Test (1 test) - ✅ FIXED

**Issue**: Test was checking slope at symmetric peak where gradient=0

**Root cause**: The elevation pattern created a symmetric peak at (2,2) where:
- All neighbors have same elevation → gradient=0 → slope=0
- Both fine and coarse grids had slope=0 at this point

**Fix applied**: Changed test to check point (1,2) which is on the actual slope, not at the peak

**Fixed test**:
- ✅ `test_grid_spacing_impact`

## Completed Work ✓✓✓

### ✓ Solar coordinate system (Dec 21, 2025)
All solar tests now pass with proper understanding of:
- UTC time convention
- Z-up coordinate system (x=east, y=north, z=up)
- Azimuth measured clockwise from north
- Sun vector points toward sun

### ✓ Solver tests (Dec 21, 2025)
Fixed all 6 solver tests by correcting API usage - `alpha` and `rho_cp` are 2D arrays

### ✓ Subsurface grid tests (Dec 21, 2025)
Fixed attribute name: `z_nodes` not `z_centers`

### ✓ Terrain test (Dec 21, 2025)
Fixed test logic to check slope at correct location

## Running Tests

```bash
# Run all tests (all pass!)
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html -v

# Run specific module
pytest tests/test_solver.py -v

# Run with detailed output
pytest -v --tb=short
```

## Status: COMPLETE ✓✓✓

**100% test pass rate achieved!**

The test suite is now fully operational and ready for regression testing. All 83 tests pass, covering:
- Atmosphere models and convection
- Material properties and databases
- Solar position and irradiance
- Terrain geometry and view factors
- Energy balance and heat conduction
- **Energy conservation** (most critical validation)

## Summary

- ✅ Test framework is solid and well-structured
- ✅ All 83 tests now pass (100% pass rate)
- ✅ Energy conservation test validates core physics
- ✅ Full regression test coverage for all modules
- ✅ Ready for continuous integration and development
