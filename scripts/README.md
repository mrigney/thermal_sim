# Scripts Directory

This directory contains utility scripts for managing the thermal terrain simulator.

## create_materials_database.py

Creates and populates the SQLite materials database with both legacy and depth-varying materials.

### Usage

**Create new database:**
```bash
python scripts/create_materials_database.py
```

**Overwrite existing database without prompting:**
```bash
python scripts/create_materials_database.py --overwrite
```

**Specify custom output path:**
```bash
python scripts/create_materials_database.py --output path/to/custom.db
```

### What It Does

The script performs two main tasks:

1. **Migrates legacy materials from JSON** - Converts 6 materials from `representative_materials.json` to SQLite format as single-depth entries for backward compatibility

2. **Adds depth-varying materials** - Populates database with 3 scientifically validated materials that have properties varying with depth:
   - Desert Sand (Depth-Varying) - Based on Presley & Christensen (1997)
   - Basalt (Weathered to Fresh) - Based on Christensen (1986)
   - Lunar Regolith Analog - Based on Cremers (1975) Apollo data

### Output

Creates `data/materials/materials.db` containing:
- 9 total materials (6 legacy + 3 depth-varying)
- Full provenance tracking (source citations, notes)
- Material versioning system
- Ready to use in simulations

### Example Output

```
======================================================================
Creating Materials Database
======================================================================
Output: data/materials/materials.db

PART 1: Migrating legacy materials from JSON
----------------------------------------------------------------------
  [1] Migrated: Dry Sand
      k=0.300 W/(m*K), rho=1600 kg/m^3, cp=800 J/(kg*K)
  [2] Migrated: Granite
      k=2.500 W/(m*K), rho=2700 kg/m^3, cp=790 J/(kg*K)
  ...

[OK] Migrated 6 legacy materials

PART 2: Adding depth-varying materials
----------------------------------------------------------------------
  [7] Added: Desert Sand (Depth-Varying)
      Depths: [0.   0.05 0.1  0.2  0.5 ]
      k range: 0.30 - 0.50 W/(m*K)
      Thermal inertia (surface): 620 J/(m^2*K*s^0.5)
      Source: Presley, M.A. & Christensen, P.R. (1997)...
  ...

[OK] Added 3 depth-varying materials

======================================================================
DATABASE CREATION COMPLETE
======================================================================

Total materials: 9
  - Legacy (uniform depth): 6
  - Depth-varying: 3

Database file: C:\...\data\materials\materials.db
Database size: 64.0 KB
```

### Adding Your Own Materials

To add custom materials to the database:

1. Edit `create_materials_database.py`
2. Add your material definition in the "PART 2" section
3. Regenerate database with `--overwrite` flag

Example:
```python
my_material = MaterialPropertiesDepth(
    material_id=str(uuid.uuid4()),
    name="My Custom Sand",
    version=1,
    source_database="Lab measurements",
    source_citation="Smith et al. (2025). Title. Journal.",
    notes="Measured under conditions X, Y, Z",
    depths=np.array([0.0, 0.1, 0.2, 0.5]),
    k=np.array([0.35, 0.40, 0.45, 0.50]),
    rho=np.array([1650, 1700, 1750, 1800]),
    cp=np.array([810, 820, 830, 840]),
    alpha=0.65,
    epsilon=0.88,
    roughness=0.002
)
mat_id = db.add_material(my_material)
```

## Future Scripts

Planned utilities (not yet implemented):

- `query_material.py` - Interactive tool to inspect database materials
- `add_material.py` - Interactive CLI for adding new materials
- `validate_database.py` - Check database integrity and physics
- `export_materials.py` - Export materials to CSV/JSON formats
