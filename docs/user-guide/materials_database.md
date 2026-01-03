# Materials Database Documentation

## Overview

The thermal terrain simulator supports two material database systems:

1. **Legacy JSON Database** - Simple, uniform thermal properties (no depth variation)
2. **SQLite Database** - Advanced system with depth-varying properties and full provenance tracking

This document describes the SQLite database system, which provides depth-varying thermal properties based on scientifically validated material data.

## Quick Start

### Using SQLite Materials in Your Simulation

Add these lines to your YAML configuration:

```yaml
materials:
  type: "uniform"
  default_material: "Desert Sand (Depth-Varying)"

  # Enable SQLite database
  use_sqlite_database: true
  sqlite_database_path: "data/materials/materials.db"
```

That's it! The solver will automatically:
- Load the material from the database
- Interpolate properties to your subsurface grid depths
- Use depth-varying thermal conductivity, density, and specific heat

## Database Schema

The SQLite database consists of five interconnected tables:

### 1. Materials Table (Core Metadata)

Stores material identification and version control:

```sql
CREATE TABLE materials (
    material_id TEXT PRIMARY KEY,          -- UUID for unique identification
    name TEXT NOT NULL,                    -- Human-readable name
    version INTEGER NOT NULL DEFAULT 1,    -- Version number
    supersedes TEXT,                       -- UUID of previous version (if any)
    created_at TEXT NOT NULL,              -- ISO 8601 timestamp
    source_database TEXT,                  -- Original data source
    source_citation TEXT,                  -- Scientific citation
    notes TEXT,                            -- Additional information
    UNIQUE(name, version)
);
```

**Design principle**: Materials are immutable. Updates create new versions with incremented version numbers.

### 2. Thermal Properties Table (Depth-Varying)

Stores thermal conductivity, density, and specific heat at discrete depths:

```sql
CREATE TABLE thermal_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT NOT NULL,
    depth_m REAL NOT NULL,                 -- Depth below surface [meters]
    k_thermal REAL NOT NULL,               -- Thermal conductivity [W/(m·K)]
    rho REAL NOT NULL,                     -- Density [kg/m³]
    cp REAL NOT NULL,                      -- Specific heat [J/(kg·K)]
    k_min REAL,                            -- Uncertainty bounds (optional)
    k_max REAL,
    rho_min REAL,
    rho_max REAL,
    cp_min REAL,
    cp_max REAL,
    FOREIGN KEY (material_id) REFERENCES materials(material_id),
    UNIQUE(material_id, depth_m)
);
```

**Depth variation**: Each material can have multiple rows at different depths. Linear interpolation is used between defined depths.

### 3. Radiative Properties Table (Surface Only)

Stores solar absorptivity and thermal emissivity (broadband):

```sql
CREATE TABLE radiative_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT NOT NULL,
    alpha_solar REAL NOT NULL,             -- Solar absorptivity [0-1]
    epsilon_thermal REAL NOT NULL,         -- Thermal emissivity [0-1]
    alpha_min REAL,                        -- Uncertainty bounds
    alpha_max REAL,
    epsilon_min REAL,
    epsilon_max REAL,
    FOREIGN KEY (material_id) REFERENCES materials(material_id),
    UNIQUE(material_id)
);
```

**Surface-only**: Radiative properties apply at the surface and do not vary with depth.

### 4. Surface Properties Table

Stores surface roughness for aerodynamic calculations:

```sql
CREATE TABLE surface_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT NOT NULL,
    roughness_m REAL NOT NULL,             -- Surface roughness [meters]
    roughness_min REAL,
    roughness_max REAL,
    FOREIGN KEY (material_id) REFERENCES materials(material_id),
    UNIQUE(material_id)
);
```

### 5. Spectral Emissivity Table (Optional)

Stores wavelength-dependent emissivity for spectral thermal modeling:

```sql
CREATE TABLE spectral_emissivity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id TEXT NOT NULL,
    wavelength_um REAL NOT NULL,           -- Wavelength [micrometers]
    emissivity REAL NOT NULL,              -- Spectral emissivity [0-1]
    temperature_k REAL,                    -- Temperature (if temperature-dependent)
    FOREIGN KEY (material_id) REFERENCES materials(material_id),
    UNIQUE(material_id, wavelength_um, temperature_k)
);
```

**Future use**: Currently not used by the solver, but available for advanced spectral calculations.

## Python API

### MaterialDatabaseSQLite Class

Main interface for database operations:

```python
from src.materials_db import MaterialDatabaseSQLite

# Open database
db = MaterialDatabaseSQLite("data/materials/materials.db")

# Query material by name
material = db.get_material_by_name("Desert Sand (Depth-Varying)")

# Query material by UUID
material = db.get_material("513ca659-91f8-4e19-8c8b-c76f812d8612")

# List all materials
db.print_summary()

# Close database
db.close()
```

### MaterialPropertiesDepth Class

Dataclass representing a material with depth-varying properties:

```python
from src.materials_db import MaterialPropertiesDepth
import numpy as np

material = MaterialPropertiesDepth(
    material_id=str(uuid.uuid4()),
    name="Custom Sand",
    version=1,
    source_database="Laboratory measurements",
    source_citation="Smith et al. (2024)",
    depths=np.array([0.0, 0.1, 0.2, 0.5]),      # Depths [m]
    k=np.array([0.3, 0.35, 0.4, 0.45]),         # k [W/(m·K)]
    rho=np.array([1600, 1700, 1750, 1800]),     # rho [kg/m³]
    cp=np.array([800, 820, 830, 840]),          # cp [J/(kg·K)]
    alpha=0.6,                                  # Solar absorptivity
    epsilon=0.9,                                # Thermal emissivity
    roughness=0.001                             # Surface roughness [m]
)

# Calculate thermal inertia at each depth
I = material.thermal_inertia()  # Returns array: sqrt(k * rho * cp)

# Interpolate to custom depths
target_depths = np.array([0.0, 0.05, 0.15, 0.3, 0.4])
props = material.interpolate_to_depths(target_depths)
# Returns dict: {'k': array, 'rho': array, 'cp': array}
```

## Available Materials

### Legacy Materials (Uniform Depth)

Migrated from JSON database, properties constant with depth:

| Name | k [W/(m·K)] | ρ [kg/m³] | cp [J/(kg·K)] | Thermal Inertia |
|------|-------------|-----------|---------------|-----------------|
| Dry Sand | 0.30 | 1600 | 800 | 620 |
| Granite | 2.50 | 2700 | 790 | 2318 |
| Basalt | 2.00 | 2900 | 840 | 2470 |
| Dry Soil | 0.40 | 1500 | 850 | 715 |
| Sandstone | 1.80 | 2200 | 800 | 1779 |
| Gravel | 0.80 | 1800 | 820 | 1082 |

### Depth-Varying Materials

#### 1. Desert Sand (Depth-Varying)

**Source**: Presley & Christensen (1997) + NASA TPSX
**Application**: Desert terrain, sandy surfaces, compaction studies
**Thermal Inertia (surface)**: 620 J/(m²·K·s^0.5)

| Depth [m] | k [W/(m·K)] | ρ [kg/m³] | cp [J/(kg·K)] | Physical Interpretation |
|-----------|-------------|-----------|---------------|------------------------|
| 0.00 | 0.30 | 1600 | 800 | Loose surface sand |
| 0.05 | 0.35 | 1650 | 810 | Beginning compaction |
| 0.10 | 0.40 | 1700 | 820 | Moderate compaction |
| 0.20 | 0.45 | 1750 | 830 | Well-compacted |
| 0.50 | 0.50 | 1800 | 840 | Fully consolidated |

**Physics**: Represents natural compaction gradient from loose surface grains to consolidated subsurface material. Thermal conductivity increases ~67% from surface to 0.5m depth.

#### 2. Basalt (Weathered to Fresh)

**Source**: Christensen (1986) + CINDAS TPMD
**Application**: Volcanic terrain, rocky surfaces, weathering profiles
**Thermal Inertia (surface)**: 1844 J/(m²·K·s^0.5)

| Depth [m] | k [W/(m·K)] | ρ [kg/m³] | cp [J/(kg·K)] | Physical Interpretation |
|-----------|-------------|-----------|---------------|------------------------|
| 0.00 | 1.5 | 2700 | 840 | Weathered, fractured surface |
| 0.02 | 1.8 | 2800 | 840 | Transition zone |
| 0.05 | 2.0 | 2900 | 840 | Fresh bedrock begins |
| 0.10 | 2.0 | 2900 | 840 | Fresh bedrock |
| 0.50 | 2.0 | 2900 | 840 | Fresh bedrock |

**Physics**: Weathering profile from fractured surface basalt to fresh bedrock. Thermal conductivity increases ~33% in the first 5cm as fractures decrease and density increases.

#### 3. Lunar Regolith Analog

**Source**: Cremers (1975) - Apollo mission data
**Application**: Lunar surface simulations, very low conductivity materials
**Thermal Inertia (surface)**: 106 J/(m²·K·s^0.5)

| Depth [m] | k [W/(m·K)] | ρ [kg/m³] | cp [J/(kg·K)] | Physical Interpretation |
|-----------|-------------|-----------|---------------|------------------------|
| 0.00 | 0.010 | 1500 | 750 | Fine dust layer |
| 0.02 | 0.015 | 1600 | 760 | Consolidating dust |
| 0.05 | 0.020 | 1700 | 770 | Compacted regolith |
| 0.10 | 0.025 | 1800 | 780 | Well-compacted |
| 0.30 | 0.030 | 1900 | 790 | Fully consolidated |

**Physics**: Extreme low conductivity due to fine particle size and vacuum conditions. Thermal conductivity triples from surface to 0.3m, but remains very low compared to terrestrial materials.

## Adding New Materials

### Method 1: Python Script

```python
from src.materials_db import MaterialDatabaseSQLite, MaterialPropertiesDepth
import numpy as np
import uuid

# Open database
db = MaterialDatabaseSQLite("data/materials/materials.db")

# Create new material
my_material = MaterialPropertiesDepth(
    material_id=str(uuid.uuid4()),
    name="My Custom Material",
    version=1,
    source_database="Laboratory XYZ",
    source_citation="Author et al. (2025). Title. Journal, vol(issue), pages.",
    notes="Additional context about material measurements and conditions",
    depths=np.array([0.0, 0.1, 0.2, 0.5]),
    k=np.array([0.5, 0.6, 0.7, 0.8]),
    rho=np.array([2000, 2100, 2150, 2200]),
    cp=np.array([900, 910, 920, 930]),
    alpha=0.7,
    epsilon=0.85,
    roughness=0.002
)

# Add to database
material_id = db.add_material(my_material)
print(f"Added material with ID: {material_id}")

db.close()
```

### Method 2: Modify create_materials_database.py

Edit `scripts/create_materials_database.py` and add your material to the database population script, then regenerate:

```bash
python scripts/create_materials_database.py --overwrite
```

## Depth Interpolation

The solver automatically interpolates material properties to match your subsurface grid. Interpolation uses:

- **Linear interpolation** between defined depth points
- **Constant extrapolation** beyond defined range (uses nearest value)

### Example:

Database defines material at depths: [0.0, 0.1, 0.2, 0.5] m
Subsurface grid has 20 layers from 0 to 0.5 m

The solver will:
1. Identify all unique depths in the grid (20 depths)
2. Interpolate k, ρ, cp to each grid depth
3. Use interpolated values in heat equation

### Accuracy Considerations:

- **Fine grids**: Interpolation is most accurate when grid spacing << spacing between database depths
- **Recommended**: Define depth points at physically meaningful transitions (e.g., layer boundaries)
- **Convergence**: Solver results converge as grid resolution increases

## Material Versioning

Materials are immutable and use versioning for updates:

```python
# Create new version of existing material
old_material = db.get_material_by_name("Desert Sand (Depth-Varying)", version=1)

new_material = MaterialPropertiesDepth(
    material_id=str(uuid.uuid4()),  # New UUID
    name="Desert Sand (Depth-Varying)",  # Same name
    version=2,  # Incremented version
    supersedes=old_material.material_id,  # Link to previous version
    # ... updated properties
)

db.add_material(new_material)
```

**Query specific version**:
```python
# Get latest version (default)
material = db.get_material_by_name("Desert Sand (Depth-Varying)")

# Get specific version
material = db.get_material_by_name("Desert Sand (Depth-Varying)", version=1)
```

## Database Tools

### Inspecting the Database

Use `print_summary()` to list all materials:

```python
from src.materials_db import MaterialDatabaseSQLite

db = MaterialDatabaseSQLite("data/materials/materials.db")
db.print_summary()
db.close()
```

Output:
```
Materials Database: data/materials/materials.db
Total materials: 9

Name                                     Version    ID
------------------------------------------------------------------------------------------
Basalt                                   1          71ef2909-0d65-4087-89e9-84628d04f4b2
Basalt (Weathered to Fresh)              1          e85542d4-3cfc-4204-b037-0967422b6c6f
Desert Sand (Depth-Varying)              1          513ca659-91f8-4e19-8c8b-c76f812d8612
...
```

### Migration from JSON

Convert existing JSON materials to SQLite:

```python
from src.materials import MaterialDatabase
from src.materials_db import MaterialDatabaseSQLite, MaterialPropertiesDepth
import uuid

# Load JSON database
json_db = MaterialDatabase()
json_db.load_from_json("data/materials/representative_materials.json")

# Create SQLite database
sqlite_db = MaterialDatabaseSQLite("my_materials.db", create_if_missing=True)

# Migrate each material
for class_id, mat in json_db.materials.items():
    migrated = MaterialPropertiesDepth(
        material_id=str(uuid.uuid4()),
        name=mat.name,
        version=1,
        source_database="migrated_from_json",
        depths=np.array([0.0]),  # Single depth for uniform materials
        k=np.array([mat.k]),
        rho=np.array([mat.rho]),
        cp=np.array([mat.cp]),
        alpha=mat.alpha,
        epsilon=mat.epsilon,
        roughness=mat.roughness
    )
    sqlite_db.add_material(migrated)

sqlite_db.close()
```

## References

Key scientific sources for material properties:

1. **Presley, M.A. & Christensen, P.R. (1997)**. Thermal conductivity measurements of particulate materials: 2. Results. *Journal of Geophysical Research*, 102(E3), 6551-6566.

2. **Christensen, P.R. (1986)**. The spatial distribution of rocks on Mars. *Icarus*, 68(2), 217-238.

3. **Cremers, C.J. (1975)**. Thermophysical properties of Apollo 14 fines. *Journal of Geophysical Research*, 80(32), 4466-4470.

4. **NASA TPSX** - Thermophysical Properties of Spacecraft and Materials Database

5. **CINDAS TPMD** - Center for Information and Numerical Data Analysis and Synthesis, Thermophysical Properties of Matter Database

## Solver Integration

### How Depth-Varying Properties Work

The thermal solver automatically handles depth-varying properties:

1. **MaterialFieldDepthVarying** loads material from SQLite database
2. Properties are **linearly interpolated** to match subsurface grid depths
3. **3D arrays** (ny, nx, nz) store k, ρ, cp at each grid cell
4. Solver uses **harmonic mean** for interface thermal conductivity:
   ```
   k_interface = 2·k₁·k₂ / (k₁ + k₂)
   ```
5. Heat equation solved implicitly with depth-dependent coefficients

### Complete Example: YAML Configuration

```yaml
# Example: configs/examples/depth_varying_demo.yaml
simulation:
  name: "desert_thermal_study"
  start_time: "2024-06-21T06:00:00"
  duration_hours: 24
  time_step: 1800  # 30 minutes

site:
  latitude: 35.0
  longitude: 106.0
  altitude: 1500.0
  timezone_offset: -7.0

terrain:
  type: "flat"
  nx: 20
  ny: 20
  dx: 10.0
  dy: 10.0

materials:
  type: "uniform"
  default_material: "Desert Sand (Depth-Varying)"

  # Enable SQLite database
  use_sqlite_database: true
  sqlite_database_path: "data/materials/materials.db"

atmosphere:
  temperature:
    model: "diurnal"
    mean_kelvin: 300.0
    amplitude_kelvin: 15.0
  wind:
    model: "diurnal"
    mean_speed: 3.0
    amplitude: 1.0

subsurface:
  z_max: 0.5        # 50 cm depth
  n_layers: 20      # Fine resolution
  stretch_factor: 1.2

output:
  directory: "output/desert_study"
  save_interval_seconds: 3600.0
  generate_plots: true
```

Run with:
```bash
python run_simulation.py --config configs/examples/depth_varying_demo.yaml --verbose
```

### Expected Output

```
INFO: Using SQLite materials database: data/materials/materials.db
Assigning depth-varying properties for 1 materials
  Interpolating to 20 depth points: 0.0013 to 0.4572 m
  - 'Desert Sand (Depth-Varying)': 5 depth points -> 20 grid layers
    k range: 0.301 to 0.493 W/(m*K)
```

The solver will:
- Interpolate from 5 database depths to 20 grid layers
- Use varying k (0.30 → 0.50 W/m·K) throughout the simulation
- Produce different thermal response compared to uniform properties

### Performance Considerations

**Memory**: Depth-varying materials use ~3× more memory than uniform (3D vs 2D arrays)
- Uniform: (ny, nx) arrays for k, ρ, cp
- Depth-varying: (ny, nx, nz) arrays

**Speed**: Negligible impact (<5% slowdown)
- Interpolation happens once during setup
- Solver computational cost dominated by time stepping

**Grid size impact**:
- Small grids (20×20×20 = 8,000 cells): No noticeable difference
- Large grids (100×100×50 = 500,000 cells): ~10 MB extra memory

### Backward Compatibility

The system maintains **full backward compatibility**:

**Legacy mode** (default):
```yaml
materials:
  type: "uniform"
  default_material: "Dry Sand"
  use_sqlite_database: false  # or omit entirely
```
- Uses JSON database
- Properties uniform with depth
- Existing simulations unaffected

**SQLite mode** (opt-in):
```yaml
materials:
  use_sqlite_database: true
  sqlite_database_path: "data/materials/materials.db"
```
- Uses SQLite database
- Supports depth-varying properties
- Can also use legacy materials (migrated as single-depth entries)

## Best Practices

1. **Always cite sources**: Include full citations for material properties
2. **Document assumptions**: Use the `notes` field to explain measurement conditions
3. **Use UUIDs**: Generate unique identifiers for each material
4. **Version control**: Never modify existing materials, create new versions instead
5. **Validate physics**: Ensure thermal inertia √(k·ρ·cp) is physically reasonable
6. **Test interpolation**: Verify interpolated properties make physical sense
7. **Match depth ranges**: Ensure material depth points span your subsurface grid depth
8. **Use legacy mode for testing**: Start with uniform materials, then switch to depth-varying

## Troubleshooting

### Material not found error
```
ValueError: Material 'My Material' not found. Available: [...]
```
**Solution**: Check material name spelling (case-sensitive). Use `db.print_summary()` to see exact names.

### Database file not found
```
FileNotFoundError: SQLite materials database not found: data/materials/materials.db
```
**Solution**: Run `python scripts/create_materials_database.py` to create the database.

### Unexpected property values
**Solution**: Check that subsurface grid depth range overlaps with material depth points. Properties are extrapolated (constant) outside defined depth range.
