# Enhanced Visualization Guide

## Overview

This guide explains the new enhanced visualization capabilities added to the thermal terrain simulator, including energy flux diagnostics and multi-panel summary plots.

## New Features

### 1. Energy Flux Diagnostics

The simulator now tracks and saves detailed energy flux information:

**Saved Fluxes** (in `diagnostics/timeseries.json`):
- `Q_solar_mean`: Mean solar radiation absorbed [W/m²]
- `Q_atm_mean`: Mean atmospheric downwelling radiation [W/m²]
- `Q_emission_mean`: Mean surface thermal emission [W/m²]
- `Q_convection_mean`: Mean convective heat transfer [W/m²]
- `Q_net_mean`: Mean net energy flux into surface [W/m²]

Plus min/max values for key fluxes.

### 2. New Plot Types

Added three new plot types to the YAML configuration:

#### `terrain_elevation`
- **Description**: Shows terrain elevation map
- **Created**: Once at start of simulation
- **Panels**: Single elevation contour map with statistics
- **Use case**: Understand terrain geometry

#### `hourly_summary`
- **Description**: Comprehensive 4-panel diagnostic plot
- **Created**: At each save interval (typically hourly)
- **Panels**:
  1. **Surface Temperature** - Contour plot of temperature field [°C]
  2. **Shadow Map** - Shows which areas are shadowed
  3. **Energy Fluxes** - Bar chart of mean flux components [W/m²]
  4. **Subsurface Profile** - Temperature vs depth at center point
- **Use case**: Detailed hourly diagnostics with all key information

#### `diagnostics_timeseries` (Enhanced)
- **Description**: Time series plots with temperature AND energy fluxes
- **Created**: Updated at each save interval
- **Panels**:
  - **Top**: Surface temperature evolution (mean, min, max)
  - **Bottom**: Energy flux time series (solar, atm, emission, convection, net)
- **Use case**: Track energy balance evolution over time

## Configuration

### YAML Configuration Example

```yaml
output:
  directory: "output/my_simulation"
  save_interval_seconds: 3600  # Hourly
  save_temperature_fields: true
  save_energy_diagnostics: true  # MUST be true for flux plots

  # Visualization settings
  generate_plots: true
  plot_format: "png"
  plot_dpi: 150
  plot_types:
    - "terrain_elevation"       # Terrain map
    - "hourly_summary"           # 4-panel hourly diagnostics
    - "diagnostics_timeseries"   # Enhanced time series
```

### Plot Type Options

| Plot Type | Description | Frequency | File Pattern |
|-----------|-------------|-----------|--------------|
| `terrain_elevation` | Terrain map | Once | `terrain_elevation.png` |
| `surface_temperature` | Simple T field | Each interval | `surface_temp_YYYYMMDD_HHMMSS.png` |
| `hourly_summary` | 4-panel diagnostic | Each interval | `hourly_summary_YYYYMMDD_HHMMSS.png` |
| `diagnostics_timeseries` | T + flux evolution | Each interval (updated) | `diagnostics_timeseries.png` |

## Understanding the Energy Flux Panel

The energy flux bar chart shows the domain-averaged fluxes:

### Positive Fluxes (Energy INTO Surface)
- **Solar** (Gold): Shortwave radiation from sun
  - Direct + diffuse components
  - Reduced in shadowed areas
  - Zero at night

- **Atmospheric** (Sky Blue): Longwave radiation from atmosphere
  - Depends on air temperature and humidity
  - Present day and night
  - Typically 200-400 W/m²

### Negative Fluxes (Energy OUT OF Surface)
- **Emission** (Dark Red): Longwave radiation emitted by surface
  - Follows Stefan-Boltzmann law (∝ T⁴)
  - Always negative (energy loss)
  - Typically -400 to -600 W/m²

- **Convection** (Light Coral): Sensible heat transfer to air
  - Depends on wind speed and T_surface - T_air
  - Can be positive or negative
  - Typically ±50 W/m²

### Net Flux (Black)
- **Net** = Solar + Atmospheric + Emission + Convection
- Positive: Surface heating
- Negative: Surface cooling
- Drives temperature changes

## Interpreting Results

### Example: Daytime Heating

Typical daytime fluxes (noon, summer):
```
Solar:        +800 W/m²  (strong direct + diffuse)
Atmospheric:  +350 W/m²  (longwave from atmosphere)
Emission:     -550 W/m²  (surface radiates)
Convection:   -100 W/m²  (surface warmer than air)
-----------------------------------
Net:          +500 W/m²  (HEATING)
```

### Example: Nighttime Cooling

Typical nighttime fluxes:
```
Solar:          0 W/m²  (sun below horizon)
Atmospheric:  +300 W/m²  (longwave from atmosphere)
Emission:     -450 W/m²  (surface radiates)
Convection:    +50 W/m²  (air warmer than surface)
-----------------------------------
Net:          -100 W/m²  (COOLING)
```

### Diagnosing Small Temperature Ranges

If you observe a **small diurnal temperature range** (e.g., only 4-5K), check:

1. **Material thermal inertia**
   - High k×ρ×cp → small temperature swings
   - Granite: High inertia (small ΔT expected)
   - Sand: Low inertia (large ΔT expected)

2. **Net flux magnitude**
   - Low Q_net → small temperature changes
   - Check if solar flux is being absorbed (Q_solar should be ~50-90% of incident radiation)

3. **Subsurface heat storage**
   - Deep z_max → more thermal mass → smaller surface ΔT
   - Lateral conduction → smooths out spatial variations

4. **Atmospheric forcing**
   - Small T_amplitude in diurnal cycle → small surface ΔT
   - Strong winds → enhanced convection → damped ΔT

## Troubleshooting

### Issue: No flux data in diagnostics

**Cause**: `save_energy_diagnostics: false` in config

**Fix**: Set to `true` in YAML

### Issue: Hourly summary plots not generated

**Possible causes**:
1. `generate_plots: false` - Enable it
2. Missing from `plot_types` - Add `"hourly_summary"`
3. No flux data - Check that solver.latest_fluxes is populated

**Debug**: Check verbose output for error messages

### Issue: Fluxes seem physically unrealistic

**Checks**:
1. Material properties (absorptivity should be 0.5-0.9)
2. Sun elevation (should be >0 during day)
3. Atmospheric temperature (should be reasonable)
4. Emission = -ε×σ×T⁴ (should be -400 to -600 W/m² for T~300K)

## Example Workflow

### 1. Create Configuration

```yaml
# my_diagnostic_run.yaml
simulation:
  name: "energy_balance_test"
  start_time: "2025-06-21 00:00:00"
  end_time: "2025-06-22 00:00:00"  # 24 hours
  dt: 120

# ... terrain, materials, etc ...

output:
  save_interval_seconds: 3600  # Hourly snapshots
  generate_plots: true
  plot_types:
    - "terrain_elevation"
    - "hourly_summary"
    - "diagnostics_timeseries"
```

### 2. Run Simulation

```bash
python run_simulation.py --config my_diagnostic_run.yaml --verbose
```

### 3. Check Outputs

```bash
cd output/energy_balance_test/

# View diagnostics
cat diagnostics/timeseries.json

# View plots
ls plots/
# terrain_elevation.png
# hourly_summary_20250621_060000.png
# hourly_summary_20250621_070000.png
# ...
# diagnostics_timeseries.png
```

### 4. Analyze Energy Balance

1. **Open `diagnostics_timeseries.png`**
   - Check if net flux is reasonable
   - Verify diurnal cycle in fluxes
   - Ensure solar > 0 during day, = 0 at night

2. **Open several `hourly_summary_*.png` files**
   - Morning (e.g., 07:00): Heating begins, shadows present
   - Noon (e.g., 12:00): Maximum solar, minimal shadows
   - Evening (e.g., 18:00): Cooling begins, long shadows
   - Night (e.g., 00:00): No solar, net cooling

3. **Compare fluxes to temperature changes**
   - Large Q_net → rapid T changes
   - Small Q_net → slow T changes
   - Net should correlate with dT/dt

## Advanced Usage

### Custom Plot Analysis

You can also use the visualization functions directly in Python:

```python
from src.visualization_enhanced import plot_hourly_summary, plot_flux_timeseries
from src.terrain import create_synthetic_terrain
# ... load your data ...

# Create custom plot
plot_hourly_summary(
    terrain=my_terrain,
    temp_field=my_temp_field,
    fluxes=my_fluxes,
    shadow_map=my_shadows,
    subsurface_grid=my_grid,
    current_time=my_time,
    output_path=Path("my_custom_plot.png"),
    dpi=300
)
```

### Flux Analysis Scripts

Extract flux statistics:

```python
import json

# Load diagnostics
with open('output/my_sim/diagnostics/timeseries.json') as f:
    data = json.load(f)

# Extract daytime fluxes
daytime_solar = [d['Q_solar_mean'] for d in data if d['Q_solar_mean'] > 10]
print(f"Mean daytime solar: {np.mean(daytime_solar):.1f} W/m²")

# Check energy balance closure
net_flux = [d['Q_net_mean'] for d in data]
print(f"Net flux range: [{min(net_flux):.1f}, {max(net_flux):.1f}] W/m²")
```

## References

### Energy Balance Equation

The surface energy balance is:

```
Q_net = Q_solar + Q_atm + Q_emission + Q_convection + Q_ground
```

Where:
- Q_solar = (1-albedo) × (I_direct × cos(θ) × (1-shadow) + I_diffuse × SVF)
- Q_atm = ε × σ × T_sky⁴
- Q_emission = -ε × σ × T_surface⁴
- Q_convection = h × (T_air - T_surface)
- Q_ground = -k × dT/dz (handled by subsurface solver)

### Physical Ranges

Typical values for desert terrain in summer:

| Quantity | Daytime | Nighttime | Units |
|----------|---------|-----------|-------|
| Q_solar | 400-900 | 0 | W/m² |
| Q_atm | 300-400 | 250-350 | W/m² |
| Q_emission | -500 to -650 | -350 to -500 | W/m² |
| Q_convection | -150 to +50 | -50 to +100 | W/m² |
| Q_net | +200 to +600 | -200 to -100 | W/m² |
| T_surface | 30-55 | 10-25 | °C |

## Support

For questions or issues:
1. Check this guide
2. Review example config: `configs/examples/enhanced_viz_example.yaml`
3. Examine output verbose messages
4. Open an issue with diagnostic plots attached
