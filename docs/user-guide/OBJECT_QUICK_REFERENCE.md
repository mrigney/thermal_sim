# 3D Object Thermal Simulation - Quick Reference

## Add Objects to Config (YAML)

```yaml
objects:
  - name: "building_1"
    mesh_file: "examples/simple_building.obj"    # Relative to data/objects/
    location: [30.0, 50.0, 0.0]                  # [x, y, z] in meters
    material: "granite"                          # From material database
    thickness: 0.25                              # meters (for 1D solver)
    rotation: [0.0, 0.0, 45.0]                  # [rx, ry, rz] Euler angles (degrees)
    enabled: true
    ground_clamped: false                        # false: use z-coord, true: auto-place on terrain
```

## Output Structure

```
output/simulation_name/
├── terrain/
│   ├── surface_temperature_0000.npy
│   ├── surface_temperature_0001.npy
│   └── ...
├── objects/
│   ├── building_1/
│   │   ├── geometry.obj                    # Once per object
│   │   ├── face_temperature_0000.npy       # Per timestep
│   │   ├── subsurface_temperature_0000.npy
│   │   ├── face_solar_flux_0000.npy
│   │   ├── face_shadow_fraction_0000.npy
│   │   └── ...
│   └── metadata.json
└── diagnostics/
    └── energy_balance.csv
```

## Load and Analyze (Python)

```python
import numpy as np

# Load object temperatures for timestep 10
temps = np.load('output/my_sim/objects/building_1/face_temperature_0010.npy')
flux = np.load('output/my_sim/objects/building_1/face_solar_flux_0010.npy')
shadows = np.load('output/my_sim/objects/building_1/face_shadow_fraction_0010.npy')

# Convert to Celsius
temps_C = temps - 273.15

print(f"Temperature range: {temps_C.min():.1f} to {temps_C.max():.1f} °C")
print(f"Mean solar flux: {flux.mean():.1f} W/m²")
print(f"Shadowed faces: {(shadows > 0.5).sum()} / {len(shadows)}")
```

## Common Materials

| Material | k (W/m·K) | Use Case | Thickness |
|----------|-----------|----------|-----------|
| granite | 2.5 | Buildings, stone | 0.20-0.30 m |
| basalt | 1.7 | Dark rock | 0.10-0.25 m |
| sandstone | 1.3 | Light stone | 0.15-0.25 m |
| granite (metal proxy) | 2.5 | Vehicles, equipment | 0.002-0.02 m |

## Location Guidelines

```yaml
# Manual placement (ground_clamped: false - default)
location: [50.0, 75.0, 0.0]    # Specify exact z-coordinate
location: [30.0, 40.0, 1.5]    # Elevated 1.5m above datum
location: [100.0, 100.0, 3.0]  # Water tower at 3m

# Automatic ground clamping (ground_clamped: true)
location: [50.0, 75.0, 0.0]    # z ignored - auto-placed on terrain at (x,y)
ground_clamped: true            # Queries terrain elevation at (x, y) location
```

## Rotation Examples

```yaml
rotation: [0.0, 0.0, 0.0]      # North-aligned (no rotation)
rotation: [0.0, 0.0, 45.0]     # Rotated 45° clockwise
rotation: [0.0, 0.0, 90.0]     # East-facing
rotation: [0.0, 0.0, 180.0]    # South-facing
rotation: [0.0, 0.0, -30.0]    # 30° counterclockwise
```

## Numerical Stability

**Thermal diffusion number**: r = κΔt/Δz² < 0.5 (recommended)

**If seeing NaN temperatures:**
- Increase `nz` (number of subsurface layers): 10 → 20
- Reduce time step: 60s → 30-45s
- Default config should use `nz=20` for objects

**Example subsurface config:**
```yaml
subsurface:
  z_max: 0.5           # 50 cm depth
  n_layers: 20         # 20 layers (good stability)
  stretch_factor: 1.2

solver:
  shadow_timestep_minutes: 60.0  # Shadow cache time resolution (default: 60)
```

## Shadow Configuration

**Shadow time resolution** (`shadow_timestep_minutes`):
- **30 minutes**: Higher accuracy, slower cache population (recommended for shadow studies)
- **60 minutes**: Balanced (default)
- **120 minutes**: Faster cache, coarser temporal resolution

## Shadow Types Implemented

1. **Object → Terrain**: Objects cast shadows on ground (most important for IR)
2. **Terrain → Object**: Terrain blocks sun from object faces
3. **Self-shadowing**: Object blocks its own faces (backface culling)

## Physics Per Face

```
Energy balance:
Q_net = Q_solar + Q_longwave_in - Q_longwave_out + Q_convection

Heat conduction (1D through thickness):
ρcp ∂T/∂t = ∂/∂z(k ∂T/∂z)
```

## Example Configs

**See**: `configs/examples/urban_objects_demo.yaml`
- 6 objects demonstrating all features
- Buildings, vehicles, elevated objects
- Multiple materials and thicknesses
- Rotation examples

## Documentation

**Complete guide**: [docs/OBJECT_THERMAL_GUIDE.md](docs/OBJECT_THERMAL_GUIDE.md) (50 pages)
- Configuration reference
- Output formats
- Post-processing workflows
- Blender import guide
- Troubleshooting

**Implementation status**: [docs/OBJECT_MODULE_COMPLETE.md](docs/OBJECT_MODULE_COMPLETE.md)

## Tests

**Run integration test:**
```bash
python test_object_thermal_integration.py
```

**Run shadow visualization:**
```bash
python test_simple_shadow_viz.py
```

## Blender Workflow (Basic)

```python
# In Blender Python console:
import bpy
import numpy as np

# 1. Import geometry
bpy.ops.import_scene.obj(filepath="output/my_sim/objects/building_1/geometry.obj")

# 2. Load temperatures
temps = np.load("output/my_sim/objects/building_1/face_temperature_0010.npy")
temps_C = temps - 273.15

# 3. Map to vertex colors
obj = bpy.context.active_object
mesh = obj.data
if not mesh.vertex_colors:
    mesh.vertex_colors.new()
color_layer = mesh.vertex_colors.active

T_min, T_max = temps_C.min(), temps_C.max()
for poly in mesh.polygons:
    T_norm = (temps_C[poly.index] - T_min) / (T_max - T_min)
    r, g, b = T_norm, 0.0, 1.0 - T_norm  # Blue to red
    for loop_idx in poly.loop_indices:
        color_layer.data[loop_idx].color = (r, g, b, 1.0)
```

## Key Features

✅ OBJ mesh loading
✅ YAML configuration
✅ Full shadow coupling (3 types)
✅ Per-face thermal solution
✅ Blender-compatible output
✅ Separate terrain/object files
✅ Material database integration
✅ Rotation support

## Known Limitations

- No lateral conduction within objects (faces independent)
- Simplified sky view factors (heuristic, not ray traced)
- No object-to-object radiation
- No contact conduction with terrain

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "OBJ file not found" | Check path in `mesh_file` (relative to `data/objects/`) |
| "Object falls through terrain" | Set `z = 0.0` for ground level |
| NaN temperatures | Increase `nz` to 20 or reduce timestep |
| Shadows look wrong | Check sun elevation > 0 and object in terrain bounds |
| All faces same temp | Normal if object small or high conductivity |

---

**Status**: ✅ Fully operational (Dec 28, 2025)
**Version**: v2.1.0
