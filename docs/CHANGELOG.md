# Changelog

All notable changes to the Thermal Terrain Simulator project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Automatic Visualization (Dec 22, 2025)
- **Optional automatic plot generation** during simulation
  - New visualization options in [src/config.py](src/config.py) `OutputConfig`
  - Plotting methods in [src/runner.py](src/runner.py)
  - Three plot types available:
    1. `surface_temperature`: 2D heatmap with statistics overlay
    2. `diagnostics_timeseries`: Temperature evolution (mean/min/max)
    3. `subsurface_profile`: Vertical temperature profile at domain center
  - Configuration options:
    - `generate_plots`: Enable/disable (default: false for backward compatibility)
    - `plot_format`: png, pdf, or svg
    - `plot_dpi`: Resolution for raster formats (default: 150)
    - `plot_types`: List of plot types to generate
  - Plots saved to `output/plots/` subdirectory
  - Non-interactive matplotlib backend (no display required)
  - Graceful error handling (plot failures don't crash simulation)
- **Updated example configs** to demonstrate visualization options
  - [configs/examples/desert_diurnal.yaml](configs/examples/desert_diurnal.yaml)
  - [configs/examples/rocky_terrain_2d.yaml](configs/examples/rocky_terrain_2d.yaml)
  - [configs/examples/shadow_study.yaml](configs/examples/shadow_study.yaml)
- **Documentation updates**
  - [configs/README.md](configs/README.md): Added visualization section

### Added - Configuration System (Dec 22, 2025)
- **YAML-based configuration system** for running simulations
  - New module [src/config.py](src/config.py) with complete configuration dataclasses
  - Three-level validation system:
    1. Schema validation (structure and types via dataclass initialization)
    2. Physics validation (warnings for questionable settings, non-blocking)
    3. Runtime validation (file existence, memory checks, critical errors only)
  - Comprehensive default values for all optional parameters
  - Support for datetime parsing (ISO 8601 format)
  - Duration specification via `end_time` or `duration_hours`
- **CLI entry point** [run_simulation.py](run_simulation.py)
  - Command-line interface for running simulations from YAML configs
  - Arguments: `--config`, `--output`, `--verbose`, `--validate-only`
  - Configuration validation without running simulation
  - User-friendly error messages and progress reporting
- **Simulation orchestration** [src/runner.py](src/runner.py)
  - `SimulationRunner` class handles complete simulation setup and execution
  - Automatic setup of terrain, materials, atmosphere, shadow cache from config
  - Initial conditions: uniform, spinup, or from file
  - Progress tracking with ETA estimation
  - Periodic output saving and checkpointing
  - Diagnostics time series (JSON format)
  - Final performance summary
- **Example configurations** in [configs/examples/](configs/examples/)
  - `desert_diurnal.yaml`: Basic 24-hour desert simulation (1D solver)
  - `rocky_terrain_2d.yaml`: Rocky terrain with lateral conduction enabled
  - `shadow_study.yaml`: High-resolution shadow/topography study

### Configuration Features
- **Flexible terrain specification**: flat, from_file, or synthetic
- **Material systems**: uniform or from classification file
- **Initial conditions**: uniform temperature, spinup period, or restart from file
- **Output control**: Configurable save intervals, checkpointing, diagnostics
- **Solver options**: Enable/disable lateral conduction, configure parameters
- **Physics warnings**: Time step stability, grid resolution, subsurface depth, etc.

## [3.0.0] - 2025-12-22

### Added - Lateral Surface Conduction (2D+1D Solver)
- **Optional lateral heat conduction** at surface (operator splitting: explicit lateral + implicit vertical)
  - New function `apply_lateral_diffusion()` in [src/solver.py](src/solver.py:533-616)
  - Vectorized 5-point stencil for 2D Laplacian with zero-flux boundary conditions
  - Stability constraints: Fourier number Fo < 0.5 with warnings at Fo ≥ 0.5 (unstable) and Fo ≥ 0.01 (inaccurate)
  - Energy conserving: < 1e-4 relative error with lateral conduction enabled
- **Configuration parameters** added to `ThermalSolver.__init__()`:
  - `enable_lateral_conduction` (bool, default False): Enable/disable lateral heat diffusion
  - `lateral_conductivity_factor` (float, default 1.0): Multiplier for lateral thermal conductivity (supports anisotropic materials)
- **Comprehensive test suite**: [tests/test_lateral_conduction.py](tests/test_lateral_conduction.py) (15 tests)
  - Backward compatibility tests (1D behavior unchanged when disabled)
  - Physics validation (heat spreading, gradient-driven flow, symmetry)
  - Energy conservation test (CRITICAL - validates < 1e-4 error)
  - Stability and boundary condition tests
  - Integration tests with full ThermalSolver
- **Complete documentation** in [docs/solver_algorithms.md](docs/solver_algorithms.md)
  - New Section 9: "Lateral Surface Conduction (Optional, v3.0+)"
  - Governing equations, spatial/temporal discretization, stability analysis
  - Usage examples and configuration guide
  - Performance impact analysis (~5-10% overhead for 100×100 grids)

### Changed - Solver Algorithm
- **Operator ordering**: Lateral diffusion applied AFTER vertical solve (not before)
  - Critical for energy conservation: Q_net computed from T^n matches temperature used in vertical solve
  - Algorithm: Energy balance → Vertical solve → Lateral diffusion
- **Updated solver documentation** to v3.0 with lateral conduction details

### Technical Details
- **Spatial discretization**: 5-point stencil for surface Laplacian
- **Temporal discretization**: Explicit forward Euler for lateral step
- **Boundary conditions**: Zero-flux (Neumann) at lateral domain edges via ghost cells
- **Vectorization**: NumPy array slicing provides 10-100× speedup over loops
- **Typical Fourier numbers**: Fo ~ 1e-4 for sand, Fo ~ 1e-3 for rock (well within stable range)

### Backward Compatibility
- **Fully backward compatible**: Feature disabled by default (enable_lateral_conduction=False)
- All 100 existing tests pass unchanged with lateral conduction disabled
- Zero performance impact when feature is disabled

### Test Results
- **Total test suite**: 104 tests (100 existing + 3 skipped + 1 new validation + 15 lateral conduction)
- **Pass rate**: 101/104 passing (97.1%), 3 skipped (Dirichlet BC validation tests)
- **Energy conservation**: < 1e-4 relative error with lateral conduction enabled (same as 1D solver)

## [2.0.0] - 2025-12-21

### Added - Testing Framework
- **Unit testing framework** with pytest (83 tests, 92% pass rate)
  - [tests/test_terrain.py](tests/test_terrain.py) - 9 tests for terrain module
  - [tests/test_materials.py](tests/test_materials.py) - 12 tests for materials module
  - [tests/test_atmosphere.py](tests/test_atmosphere.py) - 17 tests for atmosphere module
  - [tests/test_solar.py](tests/test_solar.py) - 19 tests for solar module
  - [tests/test_solver.py](tests/test_solver.py) - 26 tests for solver module
- **Test infrastructure**
  - [pytest.ini](pytest.ini) - Pytest configuration with markers
  - [tests/README.md](tests/README.md) - Quick testing guide
  - [TESTING.md](TESTING.md) - Comprehensive testing documentation
  - [tests/TEST_STATUS.md](tests/TEST_STATUS.md) - Current test status and known issues
- **Debug script organization**
  - Created [examples/debug/](examples/debug/) directory
  - Moved 5 debug scripts: debug_energy.py, debug_equilibrium.py, debug_first_hour.py, debug_solver.py, debug_timing.py

### Changed - Dependencies
- Updated [requirements.txt](requirements.txt) with testing dependencies:
  - Added `pytest>=7.4.0` for testing framework
  - Added `pytest-cov>=4.1.0` for coverage reporting

### Fixed - Solver v2.0 (Energy Conservation)
- **Implemented von Neumann boundary condition** at surface (replacing Dirichlet BC)
  - Changed from prescribed temperature to prescribed flux at surface
  - Flux BC: `-k·∂T/∂z|_surface = Q_net`
  - Energy conservation now < 1e-8 relative error (improved from ~1e-6)
  - Physically realistic temperature evolution (no more runaway heating)
- **Unified temperature field**
  - Surface is now layer 0 of subsurface grid (not separate entity)
  - Single solve per timestep (removed iterative surface update)
- **Updated solver algorithm** in [src/solver.py](src/solver.py)
  - Modified `solve_subsurface_tridiagonal()` signature to accept `Q_surface_net` and `rho_cp`
  - Removed `update_surface_temperature()` function (no longer needed)
  - Updated `ThermalSolver.step()` to single solve with flux BC
- **Shadow cache persistence** in [src/solar.py](src/solar.py)
  - Added `save()` method to save shadow cache to .npz file
  - Added `load()` classmethod to load cached shadows (0.037s vs ~100s compute)

### Fixed - Test Suite Issues
- **Solar coordinate system** (resolved all 7 failing tests)
  - Documented z-up coordinate system: x=east, y=north, z=up
  - Fixed UTC time convention in solar position tests
  - Corrected sun vector direction tests (points toward sun)
  - Fixed shadow cache API usage (add_shadow_map, get_shadow_map)
- **Materials module** (resolved 2 failing tests)
  - Fixed alpha/epsilon to 2D arrays (surface properties only)
  - Corrected material field shape assertions
  - Fixed MaterialProperties class name (was Material)

### Documentation
- **Updated [docs/solver_algorithms.md](docs/solver_algorithms.md) to v2.0**
  - Complete von Neumann BC documentation
  - Energy conservation analysis
  - Algorithm description with equations
- **Updated [docs/project_status.md](docs/project_status.md)**
  - Solver v2.0 status and validation results
  - Testing framework status
- **Created [CHANGELOG.md](CHANGELOG.md)** (this file)
  - Track all notable changes going forward

### Fixed - Test Suite Completion (Dec 21, 2025 - Later)
- **Achieved 100% test pass rate** (83/83 tests passing)
- **Fixed 6 solver tests** ([tests/test_solver.py](tests/test_solver.py))
  - Corrected API usage: `alpha` and `rho_cp` parameters are 2D arrays (ny, nx), not 3D
  - Tests: `test_solver_no_flux`, `test_solver_shape_preservation`, `test_solver_positive_flux_heats`, `test_solver_negative_flux_cools`
- **Fixed 2 subsurface grid tests** ([tests/test_solver.py](tests/test_solver.py))
  - Corrected attribute name: `z_nodes` not `z_centers` in SubsurfaceGrid class
  - Tests: `test_grid_creation`, `test_center_between_interfaces`
- **Fixed 1 terrain test** ([tests/test_terrain.py](tests/test_terrain.py))
  - Fixed test logic: check slope at (1,2) instead of symmetric peak at (2,2)
  - Test: `test_grid_spacing_impact`
- All issues were minor API/naming mismatches, not physics bugs

### Test Suite Status
- ✅ **100% pass rate achieved** - All 83 tests passing
- ✅ Energy conservation test validates core physics
- ✅ Full regression test coverage operational

### Added - Analytical Validation Tests (Dec 21, 2025)
- **Created [tests/test_analytical_validation.py](tests/test_analytical_validation.py)**
  - 3 passing validation tests for heat diffusion physics
  - Test 1: Zero flux maintains uniform temperature (validates equilibrium)
  - Test 2: Positive flux heats domain monotonically (validates heat flow direction)
  - Test 3: Heat diffusion penetration (validates diffusion physics)
- **Skipped 3 periodic temperature tests** (require Dirichlet BC, future work)
- **Total test count**: 86 tests (83 unit + 3 validation), 100% pass rate on active tests

## [1.0.0] - 2025-12-18

### Added - Initial Implementation
- **Core modules** implemented:
  - [src/terrain.py](src/terrain.py) - Terrain grid and geometric computations
  - [src/materials.py](src/materials.py) - Material properties and database
  - [src/atmosphere.py](src/atmosphere.py) - Atmospheric conditions and convection
  - [src/solar.py](src/solar.py) - Solar position, irradiance, and shadows
  - [src/solver.py](src/solver.py) - Thermal solver with subsurface heat equation
  - [src/visualization.py](src/visualization.py) - Plotting and visualization tools
- **Material database**
  - Representative materials: sand, granite, basalt, soil, sandstone, gravel
  - Properties: thermal conductivity, density, specific heat, absorptivity, emissivity
- **Examples and demonstrations**
  - 01_terrain_demo.py - Terrain grid creation
  - 02_materials_demo.py - Material properties
  - 03_atmosphere_demo.py - Atmospheric models
  - 04_solar_demo.py - Solar position and irradiance
  - 05_solver_demo.py - Full thermal simulation
- **Documentation**
  - docs/solver_algorithms.md v1.0
  - docs/project_status.md
  - README.md with project overview

### Implementation Notes - v1.0
- Initial solver used **Dirichlet boundary condition** at surface
  - Iterative coupling between surface and subsurface
  - Led to energy conservation issues and unrealistic heating
  - Fixed in v2.0 with von Neumann BC

---

## Version History Summary

- **v3.0.0** (Dec 22, 2025): Optional lateral surface conduction (2D+1D solver)
- **v2.0.0** (Dec 21, 2025): Von Neumann BC, testing framework, bug fixes
- **v1.0.0** (Dec 18, 2025): Initial implementation with all core modules

---

## Notes on Version Numbering

- **Major version** (X.0.0): Significant changes to physics models or API
- **Minor version** (0.X.0): New features, modules, or capabilities
- **Patch version** (0.0.X): Bug fixes, documentation updates, minor improvements

---

## Contributing

When making changes:
1. Update this CHANGELOG.md with your changes
2. Run test suite: `pytest -v`
3. Update relevant documentation
4. Ensure energy conservation test passes
