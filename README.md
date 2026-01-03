# Thermal Terrain Simulator

A high-fidelity thermal simulation tool for computing spatially-resolved temperature distributions on natural terrain for infrared scene generation applications.

## Project Overview

This simulator solves coupled surface energy balance and subsurface heat diffusion equations to predict thermal signatures of terrain clutter. It's designed for:
- High-resolution terrain (0.1m grid spacing)
- Kilometer-scale domains
- Multi-day simulations with diurnal cycles
- GPU acceleration for computational efficiency

## Quick Start

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the visualization demo:
```bash
cd examples
python 02_visualization_demo.py
```

This creates sample visualizations in the `outputs/` directory.

### Basic Usage

```python
from src.terrain import create_synthetic_terrain
from src.materials import MaterialDatabase
from src.visualization import quick_terrain_plot

# Create terrain
terrain = create_synthetic_terrain(100, 100, 0.5, 0.5, terrain_type='rolling_hills')
terrain.compute_normals()
terrain.compute_sky_view_factor_simple()

# Visualize
quick_terrain_plot(terrain)
```

## Project Structure

```
thermal_sim/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── terrain.py                # Terrain geometry (✅ complete)
│   ├── materials.py              # Material properties (✅ complete)
│   ├── visualization.py          # Visualization tools (✅ complete)
│   ├── solar.py                  # Solar radiation (✅ complete)
│   ├── atmosphere.py             # Atmospheric conditions (⏳ planned)
│   └── solver.py                 # Heat equation solver (⏳ planned)
│
├── examples/                     # Example scripts
│   ├── 01_basic_setup.py         # Basic terrain and materials demo
│   ├── 02_visualization_demo.py  # Visualization capabilities demo
│   └── 03_solar_demo.py          # Solar radiation demo
│
├── data/                         # Input data
│   └── materials/
│       └── representative_materials.json  # Default material database
│
├── outputs/                      # Output files
│   └── *.png                     # Visualization outputs
│
└── docs/                         # Documentation
    ├── PROJECT_STATUS.md         # Current status and roadmap
    ├── VISUALIZATION_README.md   # Visualization guide
    ├── VISUALIZATION_COMPLETE.md # Visualization completion notes
    ├── SOLAR_ALGORITHMS.md       # Solar radiation algorithms
    ├── SOLAR_MODULE_COMPLETE.md  # Solar module completion notes
    └── SETUP_GUIDE.md            # Setup instructions
```

## Features

### Implemented (Phase 1) ✅

**Terrain Module** ([src/terrain.py](src/terrain.py))
- DEM loading and synthetic terrain generation
- Surface normal, slope, and aspect computation
- Sky view factor calculation
- Multiple terrain types: flat, rolling hills, ridges, valleys

**Materials Module** ([src/materials.py](src/materials.py), [src/materials_db.py](src/materials_db.py))
- Material property database management (JSON and SQLite)
- **Depth-varying thermal properties** with SQLite database
- Spatial material field handling with depth interpolation
- Representative desert materials included
- Scientific provenance tracking and material versioning
- 9 materials available (6 legacy + 3 depth-varying)

**Visualization Module** ([src/visualization.py](src/visualization.py))
- Terrain geometry visualization (elevation, slope, aspect)
- Sky view factor plotting
- Material distribution maps
- Temperature field visualization
- Subsurface temperature profiles
- Time series plotting
- Shadow map visualization
- Animation frame generation

**Solar Radiation Module** ([src/solar.py](src/solar.py))
- Solar position calculation (azimuth, elevation)
- Sunrise/sunset time computation
- Extraterrestrial irradiance (solar constant correction)
- Clear-sky irradiance models (Ineichen-Perez and simplified)
- Surface irradiance on inclined terrain
- Shadow computation via ray marching
- Shadow caching for multi-day simulations

### Planned (Phase 2+) ⏳

- **Atmosphere Module**: Atmospheric conditions, convection coefficients, wind fields
- **Solver Module**: Heat equation solver with semi-implicit time stepping
- **GPU Acceleration**: CuPy/CUDA kernels for performance
- **I/O Module**: NetCDF/HDF5 output, checkpointing
- **Advanced Features**: Terrain-to-terrain radiation, wind fields, vegetation

## Physics Included

### Surface Energy Balance
```
ρc_p ∂T/∂t = Q_solar + Q_atmospheric + Q_emission + Q_convection +
              Q_cond_vertical + Q_cond_lateral
```

Components:
- **Solar radiation**: Direct + diffuse with shadow computation
- **Atmospheric longwave**: Sky radiation with view factors
- **Thermal emission**: Surface radiative cooling
- **Convection**: Wind-dependent heat transfer
- **Vertical conduction**: 1D subsurface heat diffusion
- **Lateral conduction**: 2D surface heat diffusion

### Numerical Methods

- **Semi-implicit time stepping** (IMEX)
- **Crank-Nicolson** for subsurface heat equation
- **ADI** (Alternating Direction Implicit) for lateral conduction
- Unconditionally stable schemes
- Time step: 60-120 seconds

## Material Properties

The simulator tracks spatially-varying material properties with optional depth variation:

**Surface Properties:**
- Solar absorptivity (α)
- Thermal emissivity (ε)
- Surface roughness

**Thermal Properties (can vary with depth):**
- Thermal conductivity (k)
- Density (ρ)
- Specific heat capacity (cp)

### Available Materials

**Legacy materials (uniform depth):**
1. Dry Sand
2. Granite
3. Basalt
4. Dry Soil
5. Sandstone
6. Gravel

**Depth-varying materials:**
7. Desert Sand (Depth-Varying) - Presley & Christensen (1997)
8. Basalt (Weathered to Fresh) - Christensen (1986)
9. Lunar Regolith Analog - Cremers (1975)

See [docs/materials_database.md](docs/materials_database.md) for complete documentation.

## Usage Examples

### Creating Terrain

```python
from src.terrain import create_synthetic_terrain

# Synthetic terrain
terrain = create_synthetic_terrain(
    nx=200, ny=200,        # 200x200 grid
    dx=0.5, dy=0.5,        # 0.5m spacing
    terrain_type='rolling_hills'
)

# Compute geometric properties
terrain.compute_normals()
terrain.compute_slope_aspect()
terrain.compute_sky_view_factor_simple()
```

### Working with Materials

```python
from src.materials import MaterialDatabase, MaterialField

# Load material database
material_db = MaterialDatabase()
material_db.load_from_json('data/materials/representative_materials.json')

# Assign materials to terrain
# (terrain.material_class is an integer array mapping to material IDs)
mat_field = MaterialField(ny, nx, nz, material_db)
mat_field.assign_from_classification(terrain.material_class)
```

### Visualization

```python
from src.visualization import TerrainVisualizer, quick_terrain_plot, quick_temp_plot
import matplotlib.pyplot as plt

# Quick terrain overview
fig = quick_terrain_plot(terrain, material_db)
plt.savefig('outputs/my_terrain.png', dpi=150)

# Temperature visualization
quick_temp_plot(T_surface, terrain, units='C')
plt.savefig('outputs/my_temperatures.png', dpi=150)

# Detailed custom plot
vis = TerrainVisualizer()
fig, ax = plt.subplots(figsize=(12, 10))
vis.plot_temperature_field(T_surface, terrain, ax=ax,
                          temp_range=(15, 45),  # fixed color scale
                          units='C',
                          title='Surface Temperature at Noon')
plt.savefig('outputs/detailed_temp.png', dpi=300)
```

## Documentation

- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Detailed project status, design decisions, and roadmap
- **[docs/VISUALIZATION_README.md](docs/VISUALIZATION_README.md)** - Complete visualization guide with examples
- **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Installation and setup instructions
- **[docs/VISUALIZATION_COMPLETE.md](docs/VISUALIZATION_COMPLETE.md)** - Visualization module completion notes

## Running Examples

### Basic Setup Example
```bash
cd examples
python 01_basic_setup.py
```

Demonstrates:
- Creating synthetic terrain
- Loading material database
- Assigning materials
- Computing thermal properties

### Visualization Demo
```bash
cd examples
python 02_visualization_demo.py
```

Creates 8 visualization outputs:
- Terrain overview (elevation, slope, SVF, materials)
- Temperature field
- Subsurface profiles
- Individual feature plots
- Shadow map demonstration
- Time series

All outputs saved to `outputs/` directory.

## Design Decisions

### Phase 1 Simplifications

1. **No terrain-to-terrain radiation**: Justified for open desert terrain where sky radiation dominates
2. **Uniform wind field**: Simple model to start, can enhance with mass-consistent solver later
3. **Pre-computed shadow caching**: Compute once per day, reuse for multiple days
4. **Python + NumPy**: Easy development, can refactor to C++/CUDA later for performance

### Future Enhancements

- Terrain-to-terrain radiation (ray tracing for valleys/canyons)
- Wind field modeling (WindNinja-style mass-consistent)
- Multi-GPU domain decomposition
- Vegetation and man-made objects

## Target Use Cases

- Desert terrain with rolling hills
- Open landscapes (high sky view factors)
- Occasional mountains or cliffs
- Natural terrain (not urban)

Typical domain sizes:
- Development/testing: 100m × 100m at 0.5m resolution
- Production: 1-10km × 1-10km at 0.1m resolution

## Performance Considerations

### Memory Estimates (10km × 10km at 0.1m)
- Grid points: 10¹⁰ points
- Surface temps: 40 GB
- Subsurface (20 layers): 800 GB
- Material properties: ~200 GB
- **Total**: ~1 TB per time snapshot

For large domains, will need:
- Multi-GPU domain decomposition
- Out-of-core processing
- Efficient I/O strategies

### Computational Cost
- Tridiagonal solves: O(N) per point, highly parallel
- Shadow computation: Most expensive, but cached
- On modern GPU: potentially 1-10 seconds per time step

## Contributing

This is an active development project. Key areas for contribution:
1. Solar radiation and shadow computation
2. Atmospheric modeling
3. Heat equation solver implementation
4. GPU kernel optimization
5. Validation against measured data

## License

[To be determined]

## Authors

Thermal Terrain Simulator Project
Started: December 2025

## Version History

- **v0.1.0** (December 2025) - Initial implementation
  - Terrain module complete
  - Materials module complete
  - Visualization module complete
  - Project structure established
