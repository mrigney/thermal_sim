# Configuration Files

This directory contains YAML configuration files for running thermal simulations.

## Quick Start

Run a simulation using one of the example configs:

```bash
# Basic desert simulation (1D solver)
python run_simulation.py --config configs/examples/desert_diurnal.yaml

# Rocky terrain with lateral conduction (2D+1D solver)
python run_simulation.py --config configs/examples/rocky_terrain_2d.yaml

# High-resolution shadow study
python run_simulation.py --config configs/examples/shadow_study.yaml
```

## Configuration File Structure

A complete configuration file has the following sections:

### Simulation
```yaml
simulation:
  name: "my_simulation"
  start_time: "2025-06-21T06:00:00"  # ISO 8601 format
  duration_hours: 24                 # OR use end_time
  time_step: 120                     # seconds
```

### Site Location
```yaml
site:
  latitude: 35.0     # degrees North
  longitude: 106.0   # degrees West (positive = West)
  altitude: 1500     # meters above sea level
```

### Terrain
```yaml
terrain:
  type: "flat"       # "flat", "from_file", or "synthetic"
  nx: 50             # grid cells in x
  ny: 50             # grid cells in y
  dx: 1.0            # grid spacing in x (meters)
  dy: 1.0            # grid spacing in y (meters)

  # For type="flat":
  flat_elevation: 0.0

  # For type="from_file":
  # elevation_file: "data/elevation.npy"
```

### Materials
```yaml
materials:
  type: "uniform"              # "uniform" or "from_classification"
  default_material: "sand"     # Name from built-in database

  # For type="from_classification":
  # classification_file: "data/materials.npy"

  # Optional custom materials:
  custom_materials:
    - name: "my_material"
      conductivity: 2.5        # W/(m·K)
      density: 2500            # kg/m³
      specific_heat: 800       # J/(kg·K)
      absorptivity: 0.85       # shortwave (0-1)
      emissivity: 0.90         # longwave (0-1)
```

Built-in materials: `sand`, `granite`, `basalt`, `soil`, `sandstone`, `gravel`

### Atmosphere
```yaml
atmosphere:
  temperature:
    model: "diurnal"
    mean_kelvin: 298.15        # 25°C
    amplitude_kelvin: 12.0     # ±12K variation
  wind:
    model: "diurnal"
    mean_speed: 3.0            # m/s
    amplitude: 1.5             # m/s
  sky_temperature_model: "idso"  # or "swinbank"
```

### Subsurface Grid
```yaml
subsurface:
  z_max: 0.5           # depth (meters)
  n_layers: 20         # number of vertical layers
  stretch_factor: 1.2  # layer spacing growth factor (≥1.0)
```

### Solver Options
```yaml
solver:
  enable_lateral_conduction: false   # Enable 2D+1D solver
  lateral_conductivity_factor: 1.0   # Lateral conductivity multiplier
```

**Note:** Lateral conduction requires sufficient grid resolution and appropriate time step. The solver will warn if stability constraints are violated.

### Initial Conditions
```yaml
initial_conditions:
  type: "uniform"              # "uniform", "spinup", or "from_file"

  # For type="uniform":
  temperature_kelvin: 298.0

  # For type="spinup":
  # spinup_days: 1

  # For type="from_file":
  # initial_state_file: "data/initial_state.npz"
```

### Output
```yaml
output:
  directory: "output/my_sim"
  save_interval_seconds: 3600        # Save every hour
  save_temperature_fields: true
  save_energy_diagnostics: true
  checkpoint_interval_seconds: 7200  # Checkpoint every 2 hours (optional)

  # Automatic visualization (optional)
  generate_plots: true               # Auto-generate plots (default: false)
  plot_format: "png"                 # png, pdf, or svg
  plot_dpi: 150                      # Resolution for raster formats
  plot_types:                        # Which plots to generate
    - "surface_temperature"          # 2D temperature field snapshots
    - "diagnostics_timeseries"       # Temperature evolution over time
    - "subsurface_profile"           # Vertical temperature profile
```

**Available plot types:**
- `surface_temperature`: 2D heatmap of surface temperature at each save interval
- `diagnostics_timeseries`: Line plot showing mean/min/max temperature evolution (updated each save)
- `subsurface_profile`: Vertical temperature profile at domain center

## Validation

The configuration system performs three levels of validation:

1. **Schema validation**: Checks structure and data types
2. **Physics validation**: Issues warnings for questionable settings (e.g., time step too large)
3. **Runtime validation**: Checks file existence, memory requirements

Validate a configuration without running:
```bash
python run_simulation.py --config my_config.yaml --validate-only
```

## Tips

### Time Step Selection
- Start with `dt = 120` seconds (2 minutes)
- Smaller time steps for:
  - Fine grids (small dx, dy)
  - Lateral conduction enabled
  - High thermal conductivity materials
- The solver warns if stability constraints are violated

### Grid Resolution
- **Coarse (1-5 m)**: Regional studies, computational efficiency
- **Medium (0.5-1 m)**: General purpose, good balance
- **Fine (0.1-0.5 m)**: Shadow effects, detailed features, lateral conduction

### Subsurface Depth
- **Rule of thumb**: Heat penetration depth ~ sqrt(α·period)
- For diurnal (24 hr) cycles:
  - Sand (α~1e-6): ~0.15 m
  - Rock (α~1e-5): ~0.5 m
- **Recommendation**: `z_max ≥ 0.5 m` for daily cycles

### Output Intervals
- **High temporal resolution**: 600-1800 s (10-30 min)
- **Standard**: 3600 s (1 hour)
- **Diagnostics only**: 7200+ s (2+ hours)

## Example Workflows

### Quick Test Run
```yaml
simulation:
  duration_hours: 2
  time_step: 120
terrain:
  nx: 20
  ny: 20
  dx: 2.0
output:
  save_interval_seconds: 600
```

### Production Run
```yaml
simulation:
  duration_hours: 168  # 1 week
  time_step: 120
terrain:
  nx: 200
  ny: 200
  dx: 0.5
output:
  save_interval_seconds: 3600
  checkpoint_interval_seconds: 14400
```

### Sensitivity Study
Create multiple configs programmatically:
```python
import yaml
from pathlib import Path

base_config = {...}  # Load base configuration

# Vary time step
for dt in [60, 120, 240]:
    config = base_config.copy()
    config['simulation']['time_step'] = dt
    config['output']['directory'] = f'output/dt_{dt}'

    with open(f'configs/sensitivity_dt_{dt}.yaml', 'w') as f:
        yaml.dump(config, f)
```

## See Also

- [../docs/solver_algorithms.md](../docs/solver_algorithms.md) - Solver physics and algorithms
- [../TESTING.md](../TESTING.md) - Testing and validation
- [../examples/](../examples/) - Python API examples
