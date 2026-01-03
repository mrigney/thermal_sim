# Configuration System Documentation

**Thermal Terrain Simulator v3.0+**

This document describes the YAML-based configuration system for running thermal simulations.

## Overview

The configuration system provides a user-friendly way to set up and run thermal simulations without writing Python code. It features:

- **YAML format**: Human-readable configuration files
- **Three-level validation**: Schema, physics, and runtime checks
- **Sensible defaults**: Minimal required parameters
- **Clear warnings**: Physics-based guidance for parameter selection
- **Flexible**: Supports multiple terrain, material, and initial condition types

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run an Example Simulation

```bash
python run_simulation.py --config configs/examples/desert_diurnal.yaml
```

### 3. Create Your Own Configuration

Copy an example config and modify for your use case:

```bash
cp configs/examples/desert_diurnal.yaml configs/my_simulation.yaml
# Edit my_simulation.yaml
python run_simulation.py --config configs/my_simulation.yaml
```

## Command-Line Interface

### Basic Usage

```bash
python run_simulation.py --config <path_to_config.yaml>
```

### Options

- `--config, -c`: Path to YAML configuration file (**required**)
- `--output, -o`: Override output directory from config
- `--verbose, -v`: Enable verbose progress logging
- `--validate-only`: Validate configuration and exit (don't run simulation)
- `--version`: Show version information

### Examples

**Validate configuration without running:**
```bash
python run_simulation.py --config my_sim.yaml --validate-only
```

**Override output directory:**
```bash
python run_simulation.py --config my_sim.yaml --output output/run_042
```

**Verbose logging with progress:**
```bash
python run_simulation.py --config my_sim.yaml --verbose
```

## Configuration File Structure

A complete YAML configuration file has 9 sections:

### 1. Simulation

Defines the simulation time span and time step.

```yaml
simulation:
  name: "my_simulation"           # Descriptive name
  start_time: "2025-06-21T06:00:00"  # ISO 8601 format
  end_time: "2025-06-22T06:00:00"    # OR use duration_hours
  duration_hours: 24              # Alternative to end_time
  time_step: 120                  # Time step in seconds
```

**Required fields:**
- `start_time`: Simulation start (ISO 8601 datetime string)
- Either `end_time` OR `duration_hours`

**Optional fields:**
- `name`: Defaults to "thermal_sim"
- `time_step`: Defaults to 120 seconds

**Notes:**
- ISO 8601 format: `YYYY-MM-DDTHH:MM:SS`
- Time step affects stability (warnings issued if too large)
- Smaller time step = more accurate but slower

### 2. Site

Geographic location for solar position calculations.

```yaml
site:
  latitude: 35.0      # Degrees North (negative for South)
  longitude: 106.0    # Degrees West (positive = West, negative = East)
  altitude: 1500      # Meters above sea level
```

**Required fields:**
- `latitude`: Must be in [-90, 90]
- `longitude`: Positive = West, negative = East

**Optional fields:**
- `altitude`: Defaults to 0.0 m

### 3. Terrain

Defines the spatial grid and elevation.

```yaml
terrain:
  type: "flat"        # "flat", "from_file", or "synthetic"
  nx: 50              # Number of grid cells in x
  ny: 50              # Number of grid cells in y
  dx: 1.0             # Grid spacing in x (meters)
  dy: 1.0             # Grid spacing in y (meters)

  # For type="flat" (optional):
  flat_elevation: 0.0

  # For type="from_file" (required if type="from_file"):
  elevation_file: "data/elevation.npy"
```

**Required fields:**
- `nx`, `ny`: Grid dimensions
- `dx`, `dy`: Grid spacing in meters

**Type-specific fields:**
- `flat`: Optionally specify `flat_elevation` (default 0.0)
- `from_file`: Must specify `elevation_file` (NumPy .npy file)
- `synthetic`: No additional parameters (generates synthetic hill)

**Notes:**
- Fine grids (dx < 0.5 m) are computationally expensive
- Coarse grids (dx > 10 m) may miss important features
- Warnings issued for extreme grid spacings

### 4. Materials

Defines surface material properties.

```yaml
materials:
  type: "uniform"              # "uniform" or "from_classification"
  default_material: "sand"     # Built-in material name

  # For type="from_classification" (optional):
  classification_file: "data/materials.npy"

  # Custom materials (optional):
  custom_materials:
    - name: "volcanic_ash"
      conductivity: 0.5          # W/(m·K)
      density: 1200              # kg/m³
      specific_heat: 900         # J/(kg·K)
      absorptivity: 0.70         # Shortwave (0-1)
      emissivity: 0.85           # Longwave (0-1)
```

**Built-in materials:**
- `sand`: Dry desert sand (low k, low α)
- `granite`: Dense crystalline rock (high k)
- `basalt`: Volcanic rock (moderate k)
- `soil`: Typical soil (moderate k)
- `sandstone`: Sedimentary rock (low-moderate k)
- `gravel`: Loose gravel (low k, high α)

**Type-specific:**
- `uniform`: Uses `default_material` everywhere
- `from_classification`: Loads 2D material ID array from `classification_file`

**Custom materials:**
- Define new materials with thermal and optical properties
- Added to material database before loading
- Can be referenced in `default_material` or classification file

### 5. Atmosphere

Atmospheric temperature, wind, and sky conditions.

```yaml
atmosphere:
  temperature:
    model: "diurnal"           # Currently only "diurnal" supported
    mean_kelvin: 298.15        # Mean temperature (25°C)
    amplitude_kelvin: 12.0     # Diurnal amplitude (±12K)
  wind:
    model: "diurnal"
    mean_speed: 3.0            # Mean wind speed (m/s)
    amplitude: 1.5             # Wind variation (m/s)
  sky_temperature_model: "idso"  # "idso" or "swinbank"
```

**Required fields:**
- `temperature.model`: Only "diurnal" currently supported
- `temperature.mean_kelvin`: Mean air temperature
- `temperature.amplitude_kelvin`: Half-range of diurnal variation
- `wind.model`: Only "diurnal" currently supported
- `wind.mean_speed`: Mean wind speed (m/s)
- `wind.amplitude`: Wind speed variation (m/s)

**Optional fields:**
- `sky_temperature_model`: Defaults to "idso"
  - `idso`: Idso & Jackson (1969) clear-sky model
  - `swinbank`: Swinbank (1963) clear-sky model

### 6. Subsurface

Vertical subsurface grid configuration.

```yaml
subsurface:
  z_max: 0.5           # Maximum depth (meters)
  n_layers: 20         # Number of vertical layers
  stretch_factor: 1.2  # Layer spacing growth factor
```

**Optional fields (all have defaults):**
- `z_max`: Defaults to 0.5 m
- `n_layers`: Defaults to 20
- `stretch_factor`: Defaults to 1.2

**Notes:**
- Layers are stretched geometrically from surface
- `stretch_factor = 1.0`: Uniform spacing
- `stretch_factor > 1.0`: Finer near surface, coarser at depth
- Recommended `z_max ≥ 0.5 m` for diurnal cycles (warning issued if < 0.3 m)

### 7. Solver

Solver algorithm options.

```yaml
solver:
  enable_lateral_conduction: false   # Enable 2D+1D solver
  lateral_conductivity_factor: 1.0   # Lateral k multiplier
  shadow_timestep_minutes: 60.0      # Shadow cache time resolution
```

**Optional fields (all have defaults):**
- `enable_lateral_conduction`: Defaults to `false` (1D-only solver)
- `lateral_conductivity_factor`: Defaults to 1.0 (isotropic material)
- `shadow_timestep_minutes`: Defaults to 60.0 minutes (hourly shadow cache)

**Notes:**
- `enable_lateral_conduction = true`: Enables lateral heat diffusion at surface
- `lateral_conductivity_factor < 1.0`: Reduces lateral conductivity (anisotropic)
- `shadow_timestep_minutes`: Controls time resolution of pre-computed shadow maps
  - Lower values (e.g., 30): More accurate shadow interpolation, slower cache population
  - Higher values (e.g., 120): Faster cache population, coarser temporal resolution
  - Recommended: 30-60 minutes for most applications
- Stability warnings issued if time step too large for lateral conduction
- See [solver_algorithms.md](solver_algorithms.md) for physics details

### 8. Initial Conditions

Starting temperature field for simulation.

```yaml
initial_conditions:
  type: "uniform"              # "uniform", "spinup", or "from_file"

  # For type="uniform":
  temperature_kelvin: 298.0

  # For type="spinup":
  spinup_days: 1

  # For type="from_file":
  initial_state_file: "data/initial_state.npz"
```

**Type-specific fields:**
- `uniform`: Requires `temperature_kelvin` (default 298.0 K)
- `spinup`: Requires `spinup_days` (default 1 day)
  - Runs simulation for specified days before actual start
  - Allows system to reach quasi-equilibrium
- `from_file`: Requires `initial_state_file` (NPZ file with T_surface, T_subsurface)

### 9. Output

Output file locations and frequencies.

```yaml
output:
  directory: "output/my_sim"           # Output directory
  save_interval_seconds: 3600          # Save every hour
  save_temperature_fields: true        # Save full temperature fields
  save_energy_diagnostics: true        # Save energy/temperature time series
  checkpoint_interval_seconds: 7200    # Checkpoint every 2 hours (optional)
```

**Required fields:**
- `directory`: Output directory path (created if doesn't exist)

**Optional fields:**
- `save_interval_seconds`: Defaults to 3600 (1 hour)
- `save_temperature_fields`: Defaults to `true`
- `save_energy_diagnostics`: Defaults to `true`
- `checkpoint_interval_seconds`: Defaults to `null` (no checkpoints)

**Output files:**
- `fields/temperature_YYYYMMDD_HHMMSS.npz`: Temperature snapshots
- `diagnostics/timeseries.json`: Energy and temperature statistics
- `checkpoints/checkpoint_YYYYMMDD_HHMMSS.npz`: Restart files
- `config.yaml`: Copy of configuration used
- `shadow_cache.npz`: Computed shadow maps (reusable)

## Three-Level Validation

The configuration system validates settings at three levels:

### Level 1: Schema Validation

**What:** Checks structure and data types

**When:** During configuration loading (before any computation)

**Failures:** Critical errors, simulation aborts

**Examples:**
- Missing required fields
- Wrong data type (e.g., string instead of number)
- Invalid datetime format

### Level 2: Physics Validation

**What:** Checks for questionable physics settings

**When:** After schema validation, before initialization

**Failures:** Warnings only (simulation continues)

**Examples:**
- Time step may be unstable (Fourier number too large)
- Grid spacing very fine or very coarse
- Subsurface depth too shallow for diurnal cycle
- Output interval may miss important features

**Philosophy:** Warn user but trust domain expertise

### Level 3: Runtime Validation

**What:** Checks critical runtime requirements

**When:** Before starting simulation

**Failures:** Critical errors for issues that prevent execution

**Examples:**
- Input files don't exist
- Output directory can't be created
- Grid too large for available memory

## Examples

### Example 1: Basic Desert Simulation

**File:** `configs/examples/desert_diurnal.yaml`

24-hour simulation of flat desert terrain with uniform sand material.

**Key features:**
- 1D-only solver (fast)
- 50×50 grid, 1 m spacing
- Uniform initial temperature
- Hourly output

**Use case:** Learning the simulator, quick tests

### Example 2: Rocky Terrain with Lateral Conduction

**File:** `configs/examples/rocky_terrain_2d.yaml`

48-hour simulation of synthetic rocky terrain with lateral heat flow.

**Key features:**
- 2D+1D solver (lateral conduction enabled)
- 100×100 grid, 0.5 m spacing
- Granite (high conductivity)
- 1-day spinup period

**Use case:** Studying lateral heat redistribution, topographic effects

### Example 3: Shadow Study

**File:** `configs/examples/shadow_study.yaml`

High-resolution simulation focusing on shadow and topographic effects.

**Key features:**
- 200×200 grid, 0.25 m spacing (very fine)
- Winter solstice (low sun angle)
- Frequent output (10 min intervals)
- 2-day spinup for stable initial state

**Use case:** Shadow analysis, detailed spatial patterns

## Programmatic Configuration Generation

For parameter studies or batch processing, generate configs programmatically:

```python
import yaml
from pathlib import Path

# Base configuration template
base_config = {
    'simulation': {
        'name': 'parameter_study',
        'start_time': '2025-06-21T00:00:00',
        'duration_hours': 24,
        'time_step': 120
    },
    'site': {'latitude': 35.0, 'longitude': 106.0, 'altitude': 0},
    'terrain': {
        'type': 'flat',
        'nx': 50, 'ny': 50,
        'dx': 1.0, 'dy': 1.0
    },
    'materials': {
        'type': 'uniform',
        'default_material': 'sand'
    },
    'atmosphere': {
        'temperature': {
            'model': 'diurnal',
            'mean_kelvin': 298.15,
            'amplitude_kelvin': 10.0
        },
        'wind': {
            'model': 'diurnal',
            'mean_speed': 3.0,
            'amplitude': 1.5
        }
    },
    'subsurface': {},  # Use defaults
    'solver': {'enable_lateral_conduction': False},
    'initial_conditions': {'type': 'uniform'},
    'output': {
        'directory': 'output/base',
        'save_interval_seconds': 3600
    }
}

# Generate configs for different time steps
for dt in [60, 120, 240, 480]:
    config = base_config.copy()
    config['simulation']['time_step'] = dt
    config['simulation']['name'] = f'timestep_study_dt{dt}'
    config['output']['directory'] = f'output/timestep_study/dt_{dt}'

    output_path = Path(f'configs/timestep_dt{dt}.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created {output_path}")
```

Then run batch:
```bash
for config in configs/timestep_*.yaml; do
    python run_simulation.py --config "$config"
done
```

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'yaml'"**

Install PyYAML:
```bash
pip install PyYAML
```

**"FileNotFoundError: Terrain file not found"**

Check path in configuration is correct and file exists:
```yaml
terrain:
  elevation_file: "path/to/elevation.npy"  # Must exist
```

**Warning: "Time step may be UNSTABLE"**

Reduce time step or increase grid spacing:
```yaml
simulation:
  time_step: 60  # Reduce from 120
```

**Warning: "Shallow subsurface depth"**

Increase subsurface depth for better physics:
```yaml
subsurface:
  z_max: 0.5  # Increase from 0.3
```

**Simulation very slow**

- Reduce grid size: `nx`, `ny`
- Increase grid spacing: `dx`, `dy`
- Increase time step: `time_step` (check stability warnings)
- Disable lateral conduction if not needed

### Getting Help

1. **Validate first**: Use `--validate-only` to check configuration
2. **Start simple**: Begin with an example config
3. **Check warnings**: Physics validation warnings provide guidance
4. **Verbose mode**: Use `--verbose` to see detailed progress

## See Also

- [configs/README.md](../configs/README.md) - Quick reference guide
- [solver_algorithms.md](solver_algorithms.md) - Physics and algorithms
- [project_status.md](project_status.md) - Current implementation status
- [TESTING.md](../TESTING.md) - Testing and validation
