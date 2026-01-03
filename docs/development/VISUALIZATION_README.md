# Visualization Module Guide

## Overview

The `visualization.py` module provides comprehensive plotting and visualization tools for the thermal terrain simulator. It allows you to visualize terrain geometry, material distributions, temperature fields, and simulation results.

## Quick Start

### Basic Usage

```python
from visualization import TerrainVisualizer, quick_terrain_plot, quick_temp_plot
import matplotlib.pyplot as plt

# Quick terrain overview
fig = quick_terrain_plot(terrain, material_db)
plt.show()

# Quick temperature plot
fig, ax = quick_temp_plot(T_surface, terrain, units='C')
plt.show()
```

### Running the Demo

```bash
python 02_visualization_demo.py
```

This creates 8 example visualizations showing all capabilities.

## Main Features

### 1. Terrain Geometry Visualization

**Elevation with Hillshading**
```python
vis = TerrainVisualizer()
fig, ax = plt.subplots()
vis.plot_elevation(terrain, ax=ax, hillshade=True)
plt.show()
```

**Slope and Aspect**
```python
fig, axes = vis.plot_slope_aspect(terrain)
plt.show()
```

**Sky View Factor**
```python
fig, ax = plt.subplots()
vis.plot_sky_view_factor(terrain, ax=ax)
plt.show()
```

### 2. Material Distribution

```python
fig, ax = plt.subplots()
vis.plot_material_distribution(terrain, material_db, ax=ax)
plt.show()
```

Shows color-coded material classes with legend.

### 3. Temperature Fields

**Surface Temperature**
```python
fig, ax = plt.subplots()
vis.plot_temperature_field(T_surface, terrain, ax=ax,
                          units='C',  # or 'K'
                          temp_range=(10, 50))  # optional fixed range
plt.show()
```

**Subsurface Profiles**
```python
fig, ax = plt.subplots()
vis.plot_subsurface_profile(T_subsurface, z_nodes,
                            i=50, j=50,  # grid indices
                            ax=ax)
plt.show()
```

### 4. Shadow Maps

```python
fig, ax = plt.subplots()
vis.plot_shadow_map(shadow_map, terrain,
                   sun_az=180.0,  # azimuth in degrees
                   sun_el=45.0,   # elevation in degrees
                   ax=ax)
plt.show()
```

### 5. Time Series

```python
fig, ax = plt.subplots()
vis.plot_temperature_time_series(times, T_values,
                                location_label='Point (50, 50)',
                                units='C')
plt.show()
```

### 6. Comprehensive Overview

```python
# Creates 2x2 grid with elevation, slope, SVF, and materials
fig = vis.plot_terrain_overview(terrain, material_db)
plt.show()
```

## Animation Support

Create animation frames for time-evolving temperature fields:

```python
from visualization import create_animation_frames

# List of temperature fields at different times
T_sequence = [T_t0, T_t1, T_t2, ...]

create_animation_frames(T_sequence, terrain,
                       output_dir='./frames',
                       temp_range=(10, 50),  # fixed scale
                       units='C')
```

Then create animation with:
```bash
# Using ffmpeg
ffmpeg -r 10 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p animation.mp4
```

## Customization

### Color Maps

Change default colormaps by modifying `TerrainVisualizer` attributes:

```python
vis = TerrainVisualizer()
vis.cmap_terrain = 'gist_earth'  # terrain elevation
vis.cmap_thermal = 'plasma'      # temperature fields
vis.cmap_sky = 'cividis'         # sky view factors
```

### Figure Sizes

```python
vis = TerrainVisualizer(figsize=(14, 12))  # default size for single plots
```

### Temperature Units

All temperature plotting functions accept `units='C'` or `units='K'`:
- `'C'`: Displays in Celsius (converts from Kelvin internally)
- `'K'`: Displays in Kelvin (default)

## Example Workflow

### During Development/Testing

```python
# 1. Create and visualize terrain
terrain = TerrainGrid(100, 100, 0.5, 0.5)
terrain.create_synthetic_terrain('rolling_hills')
terrain.compute_normals()
terrain.compute_sky_view_factor()

vis = TerrainVisualizer()
fig = vis.plot_terrain_overview(terrain)
plt.savefig('terrain_check.png', dpi=150)

# 2. Check material assignment
material_db = MaterialDatabase()
material_db.load_from_json('materials.json')
# ... assign materials ...

fig, ax = plt.subplots()
vis.plot_material_distribution(terrain, material_db, ax=ax)
plt.savefig('materials_check.png', dpi=150)

# 3. Verify initial conditions
T_initial = initialize_temperature_field(...)
fig, ax = quick_temp_plot(T_initial, terrain, units='C')
plt.savefig('initial_conditions.png', dpi=150)
```

### After Simulation

```python
# 1. Load results
T_surface = load_temperature_field('output_t_1000.npy')

# 2. Visualize
vis = TerrainVisualizer()
fig, ax = plt.subplots(figsize=(12, 10))
vis.plot_temperature_field(T_surface, terrain, ax=ax, units='C',
                          title='Temperature at t=1000s')
plt.savefig('results_t_1000.png', dpi=150)

# 3. Compare multiple times
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
times = [0, 3600, 7200, 10800]  # 0, 1hr, 2hr, 3hr

for idx, t in enumerate(times):
    ax = axes[idx // 2, idx % 2]
    T = load_temperature_field(f'output_t_{t}.npy')
    vis.plot_temperature_field(T, terrain, ax=ax, units='C',
                              temp_range=(15, 45),  # fixed scale
                              title=f'T at t={t/3600:.1f}hr')
plt.savefig('temperature_evolution.png', dpi=150)
```

## Tips

1. **Use fixed temperature ranges** when comparing fields at different times:
   ```python
   temp_range = (T_min, T_max)  # same for all plots
   ```

2. **Save figures with high DPI** for publications:
   ```python
   plt.savefig('figure.png', dpi=300, bbox_inches='tight')
   ```

3. **Use hillshading** for elevation plots to see terrain features better:
   ```python
   vis.plot_elevation(terrain, hillshade=True)
   ```

4. **Plot subsurface at interesting locations**:
   ```python
   # Find hottest surface point
   j_hot, i_hot = np.unravel_index(np.argmax(T_surface), T_surface.shape)
   vis.plot_subsurface_profile(T_subsurface, z_nodes, i=i_hot, j=j_hot)
   ```

5. **Use the comprehensive overview** for quick terrain checks:
   ```python
   quick_terrain_plot(terrain, material_db)  # One function call
   ```

## Output Files

The demo script (`02_visualization_demo.py`) creates these example outputs:
- `output_terrain_overview.png` - 2x2 overview of terrain
- `output_temperature_field.png` - Surface temperature map
- `output_subsurface_profiles.png` - Vertical temperature profiles
- `output_elevation.png` - Hillshaded elevation
- `output_slope_aspect.png` - Slope and aspect maps
- `output_sky_view_factor.png` - Sky view factor
- `output_shadow_map.png` - Example shadow map
- `output_time_series.png` - Temperature evolution over time

## Integration with Simulation

The visualization module is designed to work seamlessly with simulation output:

```python
# In your simulation loop
if output_this_timestep:
    # Quick check
    quick_temp_plot(T_surface, terrain, units='C')
    plt.savefig(f'temp_t_{timestep}.png')
    plt.close()

    # Or save for later animation
    T_frames.append(T_surface.copy())

# After simulation
create_animation_frames(T_frames, terrain, output_dir='animation')
```
