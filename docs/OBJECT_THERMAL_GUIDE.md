# 3D Object Thermal Simulation Guide

**Thermal Terrain Simulator - Object Module**
**Version**: 1.0.0
**Date**: December 28, 2025

## Overview

The object thermal simulation capability allows you to place and thermally model 3D objects (buildings, vehicles, equipment, etc.) on terrain. Objects are defined by OBJ mesh files and positioned via YAML configuration. Each object face receives an independent 1D thermal solution, with full shadow coupling to terrain and other objects.

**Key Capabilities:**
- Load 3D objects from OBJ Wavefront format
- Position and rotate objects on terrain
- Per-face thermal solution with subsurface heat conduction
- Comprehensive shadow computation (object→terrain, terrain→object, self-shadowing)
- Separate output files for terrain and objects (Blender-compatible)
- Material and thickness specification for each object

## Quick Start

### 1. Prepare Object Geometry

Create or obtain an OBJ mesh file for your object:

```bash
# Example: Simple cube
data/objects/examples/cube_1m.obj
```

The simulator includes example meshes:
- `cube_1m.obj` - 1m cube (basic test object)
- `box_2x1x1m.obj` - Elongated box (vehicle proxy)
- `simple_building.obj` - Basic building shape

### 2. Configure Your Simulation

Add objects to your YAML configuration:

```yaml
simulation:
  name: "my_simulation"
  start_time: "2025-06-21T06:00:00"
  duration_hours: 12
  time_step: 60

# ... terrain, materials, atmosphere sections ...

objects:
  - name: "building_1"
    mesh_file: "examples/simple_building.obj"
    location: [30.0, 50.0, 0.0]      # [x, y, z] in terrain coordinates
    material: "granite"               # Material from database
    thickness: 0.25                   # 25 cm walls
    rotation: [0.0, 0.0, 0.0]        # [rx, ry, rz] Euler angles
    enabled: true

  - name: "vehicle"
    mesh_file: "examples/box_2x1x1m.obj"
    location: [60.0, 30.0, 0.5]
    material: "granite"               # Metal-like properties
    thickness: 0.02                   # 2 cm metal shell
    rotation: [0.0, 0.0, 90.0]       # Rotated 90° about z-axis
    enabled: true
```

### 3. Run Simulation

```bash
python run_simulation.py --config configs/my_simulation.yaml
```

### 4. View Results

The output structure separates terrain and object data:

```
output/my_simulation/
├── terrain/
│   ├── surface_temperature_0000.npy
│   ├── surface_temperature_0001.npy
│   └── ...
├── objects/
│   ├── building_1/
│   │   ├── geometry.obj                    # Geometry (once)
│   │   ├── face_temperature_0000.npy       # Per-timestep data
│   │   ├── face_temperature_0001.npy
│   │   ├── face_solar_flux_0000.npy
│   │   ├── face_shadow_fraction_0000.npy
│   │   └── subsurface_temperature_0000.npy
│   ├── vehicle/
│   │   └── ...
│   └── metadata.json                        # Object metadata
└── diagnostics/
    └── energy_balance.csv
```

## Configuration Reference

### Object Configuration Fields

Each object in the `objects` list requires:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for this object |
| `mesh_file` | string | Yes | Path to OBJ file (relative to `data/objects/`) |
| `location` | [x, y, z] | Yes | Position in terrain coordinates (meters) |
| `material` | string | Yes | Material name from material database |
| `thickness` | float | Yes | Object thickness for 1D thermal solver (meters) |
| `rotation` | [rx, ry, rz] | No | Euler angles in degrees (default: [0, 0, 0]) |
| `enabled` | bool | No | Enable/disable this object (default: true) |
| `ground_clamped` | bool | No | Auto-place on terrain surface (default: false) |

### Location Specification

The `location` field places the object in terrain coordinates:
- **x**: East-west position (meters, origin at terrain corner)
- **y**: North-south position (meters, origin at terrain corner)
- **z**: Vertical position (meters above terrain)

**Manual Placement (ground_clamped: false, default):**
- `z = 0.0`: Object sits on terrain surface at elevation 0
- `z > 0.0`: Object elevated above terrain datum
- `z < 0.0`: Object partially buried (advanced usage)

**Automatic Ground Clamping (ground_clamped: true):**
When `ground_clamped: true`, the simulator automatically queries the terrain elevation at the (x, y) location and places the object on the terrain surface, ignoring the z-coordinate in the location field.

**Example locations:**
```yaml
# Manual placement
location: [50.0, 75.0, 0.0]       # At terrain datum (z=0)
ground_clamped: false

location: [30.0, 40.0, 1.5]       # Elevated 1.5m above datum
ground_clamped: false

# Ground clamping (auto-placed on terrain)
location: [100.0, 100.0, 0.0]     # z ignored - placed on terrain at (100, 100)
ground_clamped: true              # Queries terrain elevation automatically
```

### Rotation Specification

The `rotation` field applies Euler angle rotations:
- **rx**: Rotation about x-axis (degrees, roll)
- **ry**: Rotation about y-axis (degrees, pitch)
- **rz**: Rotation about z-axis (degrees, yaw)

**Rotation order**: rx → ry → rz (applied sequentially)

**Common rotation examples:**
```yaml
rotation: [0.0, 0.0, 0.0]      # No rotation (north-aligned)
rotation: [0.0, 0.0, 45.0]     # Rotated 45° clockwise (northeast)
rotation: [0.0, 0.0, 90.0]     # Rotated 90° (east-facing)
rotation: [0.0, 0.0, 180.0]    # Rotated 180° (south-facing)
rotation: [0.0, 0.0, -30.0]    # Rotated 30° counterclockwise
```

### Material and Thickness

Objects use materials from the material database with specified thickness:

**Material selection:**
```yaml
material: "granite"     # Stone building (k=2.5 W/(m·K))
material: "basalt"      # Dark volcanic rock (high absorptivity)
material: "sandstone"   # Light-colored rock (low absorptivity)
```

**Thickness guidelines:**
```yaml
thickness: 0.25    # 25 cm - thick walls (buildings, bunkers)
thickness: 0.10    # 10 cm - moderate walls (sheds, containers)
thickness: 0.02    # 2 cm - thin metal (vehicles, equipment)
thickness: 0.003   # 3 mm - very thin sheet metal
```

**Physical meaning:**
- Thickness defines the depth of the 1D subsurface grid through the object
- Thicker objects have greater thermal inertia (slower heating/cooling)
- Thin objects respond quickly to solar heating and atmospheric cooling

## Output File Format

### Object Metadata (`objects/metadata.json`)

Contains information about all objects:

```json
{
  "n_objects": 2,
  "objects": [
    {
      "name": "building_1",
      "n_faces": 12,
      "n_vertices": 8,
      "location": [30.0, 50.0, 0.0],
      "rotation": [0.0, 0.0, 0.0],
      "material": "granite",
      "thickness": 0.25,
      "bounds": {
        "x_min": 29.0,
        "x_max": 31.0,
        "y_min": 49.0,
        "y_max": 51.0,
        "z_min": 0.0,
        "z_max": 2.0
      }
    }
  ]
}
```

### Geometry File (`objects/{name}/geometry.obj`)

Standard OBJ Wavefront format containing mesh geometry:
- Saved once at simulation start
- Includes vertices, faces, and normals
- Compatible with Blender, ParaView, MeshLab

### Temperature Fields (`objects/{name}/face_temperature_NNNN.npy`)

NumPy array with shape `(n_faces,)` containing surface temperature of each face in Kelvin.

**Loading in Python:**
```python
import numpy as np

# Load face temperatures for timestep 10
temps = np.load('objects/building_1/face_temperature_0010.npy')

# Convert to Celsius
temps_celsius = temps - 273.15

print(f"Temperature range: {temps_celsius.min():.1f} to {temps_celsius.max():.1f} °C")
print(f"Mean temperature: {temps_celsius.mean():.1f} °C")
```

### Subsurface Temperatures (`objects/{name}/subsurface_temperature_NNNN.npy`)

NumPy array with shape `(n_faces, nz)` containing subsurface temperature profile through object thickness.

**Loading in Python:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Load subsurface temperatures
T_sub = np.load('objects/building_1/subsurface_temperature_0010.npy')

# Plot temperature profile for face 0 (e.g., south-facing wall)
face_idx = 0
plt.plot(T_sub[face_idx, :] - 273.15, label=f'Face {face_idx}')
plt.xlabel('Depth index')
plt.ylabel('Temperature (°C)')
plt.title('Subsurface Temperature Profile')
plt.legend()
plt.show()
```

### Solar Flux (`objects/{name}/face_solar_flux_NNNN.npy`)

NumPy array with shape `(n_faces,)` containing total solar flux (W/m²) on each face.

**Physical meaning:**
- Includes direct beam + diffuse sky radiation
- Accounts for face orientation (cos θ factor)
- Reduced by shadow fraction
- Zero for faces not visible to sun or sky

### Shadow Fraction (`objects/{name}/face_shadow_fraction_NNNN.npy`)

NumPy array with shape `(n_faces,)` containing shadow fraction for each face (0 = fully sunlit, 1 = fully shadowed).

**Shadow sources:**
- Terrain blocking sun (terrain → object shadowing)
- Self-shadowing (object blocking itself)
- Other objects (future enhancement)

## Physics Details

### Per-Face Thermal Solution

Each triangular face receives an independent 1D thermal solution:

**Energy balance at face surface:**
```
Q_net = Q_solar + Q_longwave_in - Q_longwave_out + Q_convection
```

Where:
- **Q_solar** = α · (I_direct · cos θ · (1 - shadow) + I_diffuse · SVF)
- **Q_longwave_in** = ε · σ · T_sky⁴ · SVF
- **Q_longwave_out** = -ε · σ · T_surface⁴
- **Q_convection** = h · (T_air - T_surface)

**Subsurface heat conduction:**
```
ρcp ∂T/∂t = ∂/∂z(k ∂T/∂z)
```

Solved using Crank-Nicolson implicit scheme with:
- Upper BC: von Neumann flux boundary (Q_net)
- Lower BC: Zero flux (insulated back face)
- Time integration: Unconditionally stable
- Surface temperature emerges from solution

### Shadow Computation

Three types of shadow interactions:

**1. Object → Terrain shadows**
- Most important for IR scene generation
- Ray from terrain point to sun
- Möller-Trumbore triangle intersection test
- Updates terrain shadow map

**2. Terrain → Object shadows**
- Ray from object face centroid to sun
- Ray marching through terrain elevation
- Stores shadow fraction per face

**3. Object self-shadowing**
- Backface culling: faces with normal·sun < 0 are shadowed
- Ray from front-facing face to sun
- Intersection test with other faces of same object

### Sky View Factors

Simplified sky visibility based on face normal direction:

```python
if nz > 0:  # Upward-facing
    sky_view = 0.5 + 0.5 * nz    # 0.5 to 1.0
else:       # Downward or sideways
    sky_view = 0.5 * max(0, 1 + nz)  # 0.0 to 0.5
```

**Physical interpretation:**
- Horizontal upward-facing surface: SVF = 1.0 (full hemisphere)
- Vertical surface: SVF ≈ 0.5 (half hemisphere)
- Horizontal downward-facing surface: SVF = 0.0 (no sky visible)

**Note:** This is a heuristic approximation. Future versions may use ray tracing for exact sky visibility.

## Post-Processing Workflows

### Python Analysis

**Example 1: Time series of object temperature**

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load all timesteps for an object
obj_dir = Path('output/my_sim/objects/building_1')
timesteps = sorted(obj_dir.glob('face_temperature_*.npy'))

# Extract mean temperature at each timestep
times = []
mean_temps = []

for i, fpath in enumerate(timesteps):
    temps = np.load(fpath)
    mean_temps.append(temps.mean() - 273.15)  # Convert to Celsius
    times.append(i * 3600)  # Assuming hourly output

# Plot time series
plt.figure(figsize=(10, 5))
plt.plot(np.array(times) / 3600, mean_temps, linewidth=2)
plt.xlabel('Time (hours)')
plt.ylabel('Mean Object Temperature (°C)')
plt.title('Building Temperature Evolution')
plt.grid(True, alpha=0.3)
plt.show()
```

**Example 2: Compare object vs terrain temperature**

```python
import numpy as np
import matplotlib.pyplot as plt

# Load timestep 10
obj_temps = np.load('output/my_sim/objects/building_1/face_temperature_0010.npy')
terrain_temps = np.load('output/my_sim/terrain/surface_temperature_0010.npy')

obj_mean = obj_temps.mean() - 273.15
terrain_mean = terrain_temps.mean() - 273.15

print(f"Object mean temperature: {obj_mean:.1f} °C")
print(f"Terrain mean temperature: {terrain_mean:.1f} °C")
print(f"Temperature difference: {obj_mean - terrain_mean:.1f} °C")

# Histogram comparison
plt.figure(figsize=(10, 5))
plt.hist(obj_temps - 273.15, bins=20, alpha=0.5, label='Object faces', density=True)
plt.hist(terrain_temps.flatten() - 273.15, bins=20, alpha=0.5, label='Terrain', density=True)
plt.xlabel('Temperature (°C)')
plt.ylabel('Probability Density')
plt.legend()
plt.title('Temperature Distribution: Object vs Terrain')
plt.show()
```

**Example 3: Solar flux and shadow analysis**

```python
import numpy as np
import matplotlib.pyplot as plt

# Load solar flux and shadow data
flux = np.load('output/my_sim/objects/building_1/face_solar_flux_0010.npy')
shadows = np.load('output/my_sim/objects/building_1/face_shadow_fraction_0010.npy')

# Identify sunlit vs shadowed faces
sunlit = shadows < 0.5
shadowed = shadows >= 0.5

print(f"Sunlit faces: {sunlit.sum()} / {len(flux)}")
print(f"Mean flux (sunlit): {flux[sunlit].mean():.1f} W/m²")
print(f"Mean flux (shadowed): {flux[shadowed].mean():.1f} W/m²")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(flux, bins=20)
axes[0].set_xlabel('Solar Flux (W/m²)')
axes[0].set_ylabel('Number of Faces')
axes[0].set_title('Solar Flux Distribution')

axes[1].scatter(range(len(flux)), flux, c=shadows, cmap='RdYlBu_r', s=50)
axes[1].set_xlabel('Face Index')
axes[1].set_ylabel('Solar Flux (W/m²)')
axes[1].set_title('Solar Flux vs Shadow Fraction')
axes[1].colorbar(label='Shadow Fraction')

plt.tight_layout()
plt.show()
```

### Blender Visualization

**Import object geometry and animate temperatures:**

1. **Import geometry:**
   ```python
   # In Blender Python console:
   import bpy

   # Import OBJ file
   bpy.ops.import_scene.obj(filepath="output/my_sim/objects/building_1/geometry.obj")
   ```

2. **Load face temperatures:**
   ```python
   import numpy as np

   # Load temperature data for timestep 10
   temps = np.load("output/my_sim/objects/building_1/face_temperature_0010.npy")
   temps_celsius = temps - 273.15
   ```

3. **Map temperatures to vertex colors:**
   ```python
   obj = bpy.context.active_object
   mesh = obj.data

   # Create vertex color layer
   if not mesh.vertex_colors:
       mesh.vertex_colors.new()

   color_layer = mesh.vertex_colors.active

   # Map temperatures to colors (simple red-blue colormap)
   T_min, T_max = temps_celsius.min(), temps_celsius.max()

   for poly in mesh.polygons:
       face_idx = poly.index
       T_norm = (temps_celsius[face_idx] - T_min) / (T_max - T_min)

       # Simple blue (cold) to red (hot) colormap
       r = T_norm
       g = 0.0
       b = 1.0 - T_norm

       for loop_idx in poly.loop_indices:
           color_layer.data[loop_idx].color = (r, g, b, 1.0)
   ```

4. **Animate through timesteps:**
   - Create keyframes for each timestep
   - Update vertex colors in each frame
   - Render animation showing temperature evolution

**Advanced Blender workflow:**
- Use Geometry Nodes to procedurally map NumPy data
- Create custom shader for temperature visualization (infrared emissivity)
- Combine object and terrain in single scene
- Add lighting to simulate thermal camera view

## Example Configurations

### Example 1: Urban Scene with Buildings

```yaml
simulation:
  name: "urban_thermal"
  start_time: "2025-06-21T06:00:00"
  duration_hours: 24
  time_step: 60

site:
  latitude: 35.0
  longitude: -106.0
  altitude: 1500.0

terrain:
  type: "flat"
  nx: 200
  ny: 200
  dx: 0.5
  dy: 0.5

materials:
  type: "uniform"
  default_material: "dry sand"

objects:
  - name: "north_building"
    mesh_file: "examples/simple_building.obj"
    location: [50.0, 80.0, 0.0]
    material: "granite"
    thickness: 0.30
    rotation: [0.0, 0.0, 0.0]
    enabled: true

  - name: "south_building"
    mesh_file: "examples/simple_building.obj"
    location: [50.0, 20.0, 0.0]
    material: "sandstone"
    thickness: 0.25
    rotation: [0.0, 0.0, 180.0]
    enabled: true

  - name: "east_building"
    mesh_file: "examples/simple_building.obj"
    location: [80.0, 50.0, 0.0]
    material: "granite"
    thickness: 0.25
    rotation: [0.0, 0.0, 270.0]
    enabled: true
```

### Example 2: Vehicle Convoy

```yaml
objects:
  - name: "vehicle_1"
    mesh_file: "examples/box_2x1x1m.obj"
    location: [30.0, 50.0, 0.5]
    material: "granite"      # Metal-like properties
    thickness: 0.02
    rotation: [0.0, 0.0, 0.0]
    enabled: true

  - name: "vehicle_2"
    mesh_file: "examples/box_2x1x1m.obj"
    location: [35.0, 50.0, 0.5]
    material: "granite"
    thickness: 0.02
    rotation: [0.0, 0.0, 0.0]
    enabled: true

  - name: "vehicle_3"
    mesh_file: "examples/box_2x1x1m.obj"
    location: [40.0, 50.0, 0.5]
    material: "granite"
    thickness: 0.02
    rotation: [0.0, 0.0, 0.0]
    enabled: true
```

### Example 3: Shadow Study with Elevated Object

```yaml
objects:
  - name: "water_tower"
    mesh_file: "examples/cube_1m.obj"
    location: [50.0, 50.0, 5.0]   # Elevated 5m
    material: "granite"
    thickness: 0.10
    rotation: [0.0, 0.0, 0.0]
    enabled: true

  - name: "reference_ground"
    mesh_file: "examples/cube_1m.obj"
    location: [60.0, 50.0, 0.5]   # Ground level
    material: "basalt"
    thickness: 0.10
    enabled: true
```

## Troubleshooting

### Common Issues

**Problem: "OBJ file not found"**

Check that:
- OBJ file exists in `data/objects/` directory
- Path in `mesh_file` is correct (relative to `data/objects/`)
- File has `.obj` extension

**Problem: "Object falls through terrain"**

Check `location` z-value:
- `z = 0.0` places object on terrain surface
- Negative z may bury object
- Ensure object's bottom vertices have z ≥ 0 in local coordinates

**Problem: "Object shadows look wrong"**

Check:
- Sun elevation > 0 (sun is above horizon)
- Object location is within terrain bounds
- Terrain resolution is sufficient to capture shadow details

**Problem: "Object temperatures are NaN"**

Check numerical stability:
- Reduce time step or increase subsurface layers (nz)
- Ensure material properties are physical (k > 0, ρ > 0, cp > 0)
- Check thermal diffusion number: r = κΔt/Δz² < 0.5

**Problem: "All faces have same temperature"**

This may be expected if:
- Object is small (low spatial variation)
- Material has high thermal conductivity
- Simulation just started (not enough time for gradients to develop)

### Performance Considerations

**Shadow computation:**
- Most expensive part: O(n_terrain × n_objects × n_faces) per shadow map
- For large/complex objects, use simplified proxy meshes for shadow casting
- Future optimization: BVH spatial acceleration structure

**Memory usage:**
- Each object adds: `n_faces × (4 + 4 + 4 + nz×4)` bytes per timestep
- Example: 1000 faces, 20 layers, 100 timesteps = ~1 MB
- Multiple complex objects can add significant memory

**Thermal solver:**
- Per-face 1D solve is parallelizable
- CPU performance good up to ~10,000 total faces
- GPU acceleration planned for larger object counts

## Technical Reference

### Coordinate Systems

**Terrain coordinates:**
- Origin: Southwest corner of terrain grid
- x-axis: East (increasing grid i)
- y-axis: North (increasing grid j)
- z-axis: Up (elevation above datum)

**Object local coordinates:**
- Defined in OBJ file
- Transformed to world coordinates via rotation + translation

**Rotation convention:**
- Euler angles: [rx, ry, rz] in degrees
- Order: rx (roll) → ry (pitch) → rz (yaw)
- Right-hand rule: positive rotation is counterclockwise when viewed along axis

### Numerical Scheme

**Time integration:**
- Crank-Nicolson implicit scheme (unconditionally stable)
- Thomas algorithm for tridiagonal solve (O(nz) per face)
- Explicit treatment of radiation and convection

**Spatial discretization:**
- Finite volume method
- Geometric stretching (fine near surface, coarse at depth)
- Default: 10-20 layers through object thickness

**Stability criteria:**
- Thermal diffusion number: r = κΔt/Δz² (recommended r < 10)
- Time step: dt = 60-120 seconds typical
- Warnings issued if r > 50 or r < 0.1

### File Formats

**OBJ Wavefront format:**
```
# Vertices
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0

# Normals (optional)
vn 0.0 0.0 1.0

# Faces (1-indexed)
f 1 2 3
f 1 3 4
```

**NumPy .npy format:**
- Binary format for arrays
- Efficient, portable
- Load with `np.load()`, save with `np.save()`

## Future Enhancements

**Planned features:**

1. **Lateral heat conduction within objects**
   - Currently: Each face is thermally independent
   - Future: 3D heat conduction within solid objects

2. **Object-to-object radiative coupling**
   - Currently: Objects only exchange radiation with sky and terrain
   - Future: Objects radiate to each other

3. **Accurate sky view factors**
   - Currently: Simplified heuristic based on normal direction
   - Future: Ray tracing for exact hemisphere visibility

4. **BVH acceleration for shadows**
   - Currently: Brute-force triangle intersection
   - Future: Bounding Volume Hierarchy for O(log N) performance

5. **Contact conduction for grounded objects**
   - Currently: Objects thermally isolated from terrain subsurface
   - Future: Heat conduction through object-terrain contact

6. **FBX and glTF geometry import**
   - Currently: OBJ only
   - Future: Support more 3D formats from CAD tools

## See Also

- [configuration.md](configuration.md) - General configuration guide
- [solver_algorithms.md](solver_algorithms.md) - Thermal solver details
- [SOLAR_ALGORITHMS.md](SOLAR_ALGORITHMS.md) - Shadow computation algorithms
- [urban_objects_demo.yaml](../configs/examples/urban_objects_demo.yaml) - Example configuration

## Acknowledgments

**Shadow computation:**
- Möller-Trumbore algorithm (1997): Fast ray-triangle intersection
- Described in: Möller & Trumbore, "Fast, Minimum Storage Ray-Triangle Intersection", Journal of Graphics Tools

**Thermal modeling:**
- 1D thermal solver per face approach
- Inspired by building energy simulation methods (EnergyPlus, ESP-r)

## Support

For questions or issues:
1. Check this documentation
2. Review example configurations in `configs/examples/`
3. Examine test scripts: `test_object_thermal_integration.py`, `test_simple_shadow_viz.py`
4. Consult project status: [docs/project_status.md](project_status.md)
