# Thermal Terrain Simulator

A high-fidelity thermal simulation tool for computing spatially-resolved surface and subsurface temperatures on natural terrain, designed for infrared scene generation applications.

## Overview

This simulator solves the coupled surface energy balance and subsurface heat diffusion equations to predict temperature distributions across terrain with complex topography. It accounts for:

- Solar radiation (direct and diffuse) with shadowing
- Atmospheric longwave radiation
- Surface thermal emission
- Convective heat transfer
- Vertical heat conduction into subsurface
- Lateral surface heat conduction
- Material property variations

## Features

- Multi-day simulation capability with diurnal cycles
- High-resolution terrain support (0.1m grid spacing)
- Multiple material types with spatially-varying properties
- Pre-computed shadow caching for efficiency
- GPU acceleration via CuPy
- Semi-implicit time stepping for stability

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Note: CuPy requires CUDA toolkit. Adjust the CuPy version in requirements.txt based on your CUDA version.

## Project Structure

```
thermal_terrain_sim/
├── src/                    # Source code
│   ├── terrain.py         # Terrain geometry and properties
│   ├── materials.py       # Material property management
│   ├── solar.py           # Solar radiation and shadows
│   ├── atmosphere.py      # Atmospheric conditions
│   ├── solver.py          # Heat equation solver
│   ├── io_utils.py        # File I/O utilities
│   └── kernels.py         # GPU kernel implementations
├── data/                   # Input data files
│   ├── dem/               # Digital elevation models
│   ├── materials/         # Material property databases
│   └── weather/           # Atmospheric forcing data
├── outputs/               # Simulation outputs
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for analysis
└── examples/              # Example scripts
```

## Quick Start

See `examples/` directory for example simulation scripts.

## Physics Background

### Surface Energy Balance
```
ρc_p ∂T/∂t = Q_solar + Q_atmospheric + Q_emission + Q_convection + Q_conduction
```

### Subsurface Heat Equation (1D at each surface point)
```
ρc_p ∂T/∂t = ∂/∂z(k ∂T/∂z)
```

### Lateral Surface Conduction
```
∂T/∂t = α ∇²T
```

## Development Status

**Phase 1 (Current):** Core physics implementation
- [x] Project structure
- [ ] Terrain module
- [ ] Material properties
- [ ] Solar radiation and shadows
- [ ] Heat solver (surface + subsurface)
- [ ] Basic validation cases

**Phase 2 (Future):** Enhanced features
- [ ] Terrain-to-terrain radiation
- [ ] GPU optimization
- [ ] Multi-GPU support
- [ ] Advanced atmospheric models

**Phase 3 (Future):** Extended capabilities
- [ ] Vegetation modeling
- [ ] Man-made objects
- [ ] Advanced validation

## License

TBD

## Contact

TBD
