# Thermal Terrain Simulator - Documentation

Complete documentation for the thermal terrain simulation framework with depth-varying materials, solar irradiance, and object integration.

## Quick Navigation

### For New Users
Start here if you're new to the simulator:

1. **[Setup Guide](user-guide/SETUP_GUIDE.md)** - Installation and first run
2. **[Configuration Guide](user-guide/configuration.md)** - YAML configuration reference
3. **[Materials Database](user-guide/materials_database.md)** - Material properties and database usage
4. **[Object Thermal Guide](user-guide/OBJECT_THERMAL_GUIDE.md)** - Adding thermal objects to simulations
5. **[Object Quick Reference](user-guide/OBJECT_QUICK_REFERENCE.md)** - Fast reference for object configuration

### For Developers
Technical documentation and implementation details:

- **[Testing Guide](development/TESTING.md)** - Running and writing tests
- **[Testing Summary](development/TESTING_SUMMARY.md)** - Test coverage and results
- **[Visualization Development](development/VISUALIZATION_README.md)** - Plot generation tools

### Technical Documentation
Deep dives into algorithms and physics:

- **[Solar Algorithms](technical/SOLAR_ALGORITHMS.md)** - Solar position, irradiance, and shadow computation
- **[Atmosphere Algorithms](technical/ATMOSPHERE_ALGORITHMS.md)** - Atmospheric temperature, wind, and sky radiation models
- **[Shadow Optimization](technical/SHADOW_OPTIMIZATION.md)** - Shadow cache system and performance
- **[Enhanced Visualization Guide](technical/ENHANCED_VISUALIZATION_GUIDE.md)** - Advanced plotting capabilities

### Project Management
Track project status and changes:

- **[CHANGELOG.md](CHANGELOG.md)** - Version history and feature additions
- **[Project Status](project_status.md)** - Current implementation status and roadmap

## Documentation Structure

```
docs/
├── README.md                          # This file - navigation guide
├── CHANGELOG.md                       # Version history
├── project_status.md                  # Project status and roadmap
│
├── user-guide/                        # End-user documentation
│   ├── SETUP_GUIDE.md                # Installation and first run
│   ├── configuration.md              # YAML configuration reference
│   ├── materials_database.md         # Materials database guide
│   ├── OBJECT_THERMAL_GUIDE.md       # Object integration guide
│   └── OBJECT_QUICK_REFERENCE.md     # Object configuration quick ref
│
├── technical/                         # Technical algorithm documentation
│   ├── SOLAR_ALGORITHMS.md           # Solar radiation calculations
│   ├── ATMOSPHERE_ALGORITHMS.md      # Atmospheric models
│   ├── SHADOW_OPTIMIZATION.md        # Shadow computation
│   └── ENHANCED_VISUALIZATION_GUIDE.md  # Advanced visualization
│
├── development/                       # Developer documentation
│   ├── TESTING.md                    # Testing guide
│   ├── TESTING_SUMMARY.md            # Test results
│   └── VISUALIZATION_README.md       # Visualization development
│
└── archive/                           # Historical documentation
    ├── implementation-notes/          # Completed implementation docs
    ├── bugfixes/                      # Bug fix documentation
    ├── discussions/                   # Design discussions
    └── daily-logs/                    # Development logs
```

## Key Features

### Depth-Varying Materials
The simulator supports materials with thermal properties (k, ρ, cp) that vary with depth:
- SQLite database with scientific citations
- Automatic interpolation to solver grid
- 9 materials including lunar regolith, desert sand, and basalt
- See [materials_database.md](user-guide/materials_database.md)

### Solar Radiation & Shadows
High-fidelity solar irradiance with shadow computation:
- Möller-Trumbore ray-triangle intersection
- Configurable shadow cache for performance
- Direct, diffuse, and reflected radiation
- See [SOLAR_ALGORITHMS.md](technical/SOLAR_ALGORITHMS.md)

### Thermal Objects
Add 3D objects to terrain simulations:
- Ground clamping for automatic placement
- Object-terrain and object-object shadows
- Thermal integration with subsurface solver
- See [OBJECT_THERMAL_GUIDE.md](user-guide/OBJECT_THERMAL_GUIDE.md)

### Atmospheric Models
Realistic atmospheric conditions:
- Diurnal temperature and wind cycles
- Multiple sky temperature models (Idso, Prata, Swinbank, Simple)
- Humidity and cloud cover support
- See [ATMOSPHERE_ALGORITHMS.md](technical/ATMOSPHERE_ALGORITHMS.md)

## Common Workflows

### Running Your First Simulation

1. Install dependencies: See [SETUP_GUIDE.md](user-guide/SETUP_GUIDE.md)
2. Choose a demo configuration from `configs/examples/`:
   - `depth_varying_demo.yaml` - Depth-varying materials
   - `lunar_regolith_demo.yaml` - Lunar surface simulation
   - `legacy_materials_demo.yaml` - Uniform materials
3. Run: `python main.py configs/examples/depth_varying_demo.yaml`
4. Check output in `output/depth_varying_demo/`

### Configuring Materials

**Uniform materials (legacy JSON):**
```yaml
materials:
  type: "uniform"
  default_material: "Dry Sand"
  use_sqlite_database: false
```

**Depth-varying materials (SQLite):**
```yaml
materials:
  type: "uniform"
  default_material: "Desert Sand (Depth-Varying)"
  use_sqlite_database: true
  sqlite_database_path: "data/materials/materials.db"
```

See [configuration.md](user-guide/configuration.md) for full reference.

### Adding Objects

```yaml
objects:
  - name: "boulder"
    mesh_file: "data/objects/boulder.obj"
    position: [50.0, 50.0, null]  # null = auto ground clamp
    material: "Granite"
```

See [OBJECT_QUICK_REFERENCE.md](user-guide/OBJECT_QUICK_REFERENCE.md) for details.

## Physics Implementation

### Surface Energy Balance
```
ρcp ∂T/∂t = Q_solar + Q_atmospheric + Q_emission + Q_convection + Q_conduction
```

### Subsurface Heat Equation
```
ρcp ∂T/∂t = ∂/∂z[k(z) ∂T/∂z]
```
- Depth-varying thermal conductivity k(z)
- Crank-Nicolson implicit solver
- Harmonic mean for interface conductivity

### Lateral Conduction
```
∂T/∂t = α ∇²T
```
- Optional lateral heat transfer
- Coupled with vertical diffusion

See [technical/](technical/) folder for detailed algorithm documentation.

## Archive

Historical implementation notes and completed work are in [archive/](archive/):
- **[implementation-notes/](archive/implementation-notes/)** - Depth-varying materials, object integration, etc.
- **[bugfixes/](archive/bugfixes/)** - Bug fix documentation
- **[discussions/](archive/discussions/)** - Design discussions (e.g., materials database design)
- **[daily-logs/](archive/daily-logs/)** - Development logs

## Getting Help

1. Check the appropriate guide above
2. Review [CHANGELOG.md](CHANGELOG.md) for recent changes
3. Check [project_status.md](project_status.md) for known issues
4. Review archived implementation notes for specific features

## Contributing

When adding new features:
1. Update relevant user-guide documentation
2. Add technical documentation to technical/ if implementing new algorithms
3. Update CHANGELOG.md
4. Update project_status.md if affecting roadmap
5. Add implementation notes to archive/ when complete
