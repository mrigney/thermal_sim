# Setup Guide - Thermal Terrain Simulator

## Initial Setup

### 1. Install Required Packages

First, install the required Python packages. From this directory, run:

```bash
pip install -r requirements.txt
```

**Note**: If you don't plan to use GPU acceleration immediately, you can skip CuPy installation for now:

```bash
# Install everything except CuPy
pip install numpy scipy matplotlib pandas netCDF4 h5py pysolar pyproj rasterio tqdm
```

You can add CuPy later when ready for GPU acceleration.

### 2. Verify Installation

Test that the basic modules work:

```bash
python 01_basic_setup.py
```

This should create a simple terrain and assign materials without errors.

### 3. Test Visualization

Run the visualization demo:

```bash
python 02_visualization_demo.py
```

This will:
- Create synthetic terrain (100x100 grid, 50m x 50m domain)
- Compute geometric properties
- Assign materials
- Generate synthetic temperature fields
- Create 8 visualization outputs

**Expected Output Files**:
- `output_terrain_overview.png` - Comprehensive terrain view
- `output_temperature_field.png` - Surface temperature map
- `output_subsurface_profiles.png` - Vertical temperature profiles
- `output_elevation.png` - Hillshaded elevation
- `output_slope_aspect.png` - Slope and aspect
- `output_sky_view_factor.png` - Sky view factors
- `output_shadow_map.png` - Shadow map example
- `output_time_series.png` - Temperature time series

## Quick Test Without Full Installation

If you want to test just the terrain module without all dependencies:

```bash
# Install minimal requirements
pip install numpy scipy matplotlib

# Run basic terrain demo
python 01_basic_setup.py
```

## Project Structure

```
thermal_sim/
├── terrain.py                    # Terrain geometry module ✅
├── materials.py                  # Material properties module ✅
├── visualization.py              # Visualization tools ✅
├── 01_basic_setup.py            # Basic terrain/materials demo ✅
├── 02_visualization_demo.py     # Visualization demo ✅
├── representative_materials.json # Material database ✅
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
├── PROJECT_STATUS.md             # Current status and roadmap
├── VISUALIZATION_README.md       # Visualization guide
└── SETUP_GUIDE.md               # This file

Next to implement:
├── solar.py                      # Solar radiation and shadows
├── atmosphere.py                 # Atmospheric conditions
└── solver.py                     # Heat equation solver
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, make sure you've installed requirements:
```bash
pip install -r requirements.txt
```

### CuPy Installation Issues

CuPy requires CUDA toolkit. If you have issues:
1. Skip CuPy for now (not needed for initial development)
2. Or install the version matching your CUDA installation:
   ```bash
   pip install cupy-cuda11x  # for CUDA 11.x
   pip install cupy-cuda12x  # for CUDA 12.x
   ```

### Matplotlib Display Issues

If running headless or via SSH:
1. Comment out `plt.show()` at the end of demo scripts
2. Or set matplotlib backend:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # before importing pyplot
   ```

### Path Issues

If Python can't find modules, make sure you're running from the project directory:
```bash
cd c:\Users\Matt Rigney\Documents\thermal_sim
python 02_visualization_demo.py
```

## Next Steps

After verifying the visualization works:

1. **Add your material database** (replace `representative_materials.json` or add new file)
2. **Test with your own terrain data** (modify examples to load DEM files)
3. **Continue development**:
   - Solar radiation module
   - Atmospheric conditions module
   - Heat equation solver

## Using the Visualization in Your Workflow

### During Development

```python
from visualization import quick_terrain_plot, quick_temp_plot
import matplotlib.pyplot as plt

# Quick terrain check
fig = quick_terrain_plot(terrain, material_db)
plt.show()

# Quick temperature check
fig, ax = quick_temp_plot(T_surface, terrain, units='C')
plt.show()
```

### For Publications/Reports

```python
from visualization import TerrainVisualizer

vis = TerrainVisualizer(figsize=(10, 8))

# High-resolution output
fig, ax = plt.subplots()
vis.plot_temperature_field(T_surface, terrain, ax=ax, units='C')
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

See `VISUALIZATION_README.md` for detailed usage examples.

## Getting Help

- Check `PROJECT_STATUS.md` for current implementation status
- See `VISUALIZATION_README.md` for visualization examples
- Review example scripts (`01_basic_setup.py`, `02_visualization_demo.py`)
- Material database format: `representative_materials.json`
