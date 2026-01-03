# Material Classification and Boundary Handling Design

**Date**: January 3, 2026
**Status**: Design/Planning Phase
**Related Issue**: Spatially-varying materials with realistic boundary transitions

## Overview

This document outlines the design for two related capabilities:
1. **Material classification maps** - Define different materials at each grid point
2. **Diffuse material boundaries** - Handle realistic (non-sharp) transitions between materials

Both are important for realistic thermal modeling of natural terrain where material properties vary spatially and boundaries are gradual rather than abrupt.

---

## 1. Material Classification Maps

### Current State

**Good news**: The infrastructure already exists!

The codebase already supports material classification via:
- `MaterialField.assign_from_classification()` ([src/materials.py:163-197](../../../src/materials.py#L163-L197))
- Configuration: `materials.type = "from_classification"`
- Works with both JSON (legacy) and SQLite (depth-varying) databases

**How it works**:
```python
# Takes a 2D array (ny, nx) of material class IDs
material_class = np.array([
    [1, 1, 2, 2],  # Row 0: sand, sand, basalt, basalt
    [1, 2, 2, 3],  # Row 1: sand, basalt, basalt, granite
    [2, 2, 3, 3],  # Row 2: basalt, basalt, granite, granite
])

# Assigns properties from database for each class
materials.assign_from_classification(material_class)
```

The method:
1. Validates all class IDs exist in database
2. Creates masks for each unique class
3. Assigns surface properties (alpha, epsilon, roughness) - 2D
4. Assigns subsurface properties (k, rho, cp) - 3D with depth variation if using SQLite

### What's Missing

#### 1.1. File Format Readers

**Current limitation**: Classification must be provided as numpy array in memory.

**Needed file formats**:

| Format | Priority | Use Case | Implementation |
|--------|----------|----------|----------------|
| **GeoTIFF** | High | Remote sensing data, GIS integration | `rasterio` or `gdal` |
| **PNG/JPG** | High | Easy to create/visualize, artistic control | `PIL` or `imageio` |
| **NumPy (.npy)** | Medium | Already supported via code, need file loader | `np.load()` |
| **CSV** | Low | Simple text format, spreadsheet editing | `np.loadtxt()` |
| **ASCII Grid** | Low | GIS standard (ESRI .asc) | Custom parser |

**Proposed Implementation**:
```python
# In src/materials.py or new src/classification_io.py
def load_classification_map(filepath: str, nx: int, ny: int) -> np.ndarray:
    """
    Load material classification map from various file formats.

    Args:
        filepath: Path to classification file
        nx, ny: Expected grid dimensions

    Returns:
        2D integer array (ny, nx) of material class IDs

    Supported formats:
        - .tif, .tiff: GeoTIFF (single band, integer values)
        - .png, .jpg: Image (pixel values = class IDs or via colormap)
        - .npy: NumPy array
        - .csv, .txt: CSV/ASCII text
    """
    ext = Path(filepath).suffix.lower()

    if ext in ['.tif', '.tiff']:
        # Use rasterio for proper georeferencing
        import rasterio
        with rasterio.open(filepath) as src:
            data = src.read(1)  # Read first band

    elif ext in ['.png', '.jpg', '.jpeg']:
        # Load as image
        from PIL import Image
        img = Image.open(filepath)
        data = np.array(img)

        # If RGB, might need to map colors to class IDs
        if len(data.shape) == 3:
            # Option 1: Use colormap lookup
            # Option 2: Use just one channel (R)
            data = data[:, :, 0]

    elif ext == '.npy':
        data = np.load(filepath)

    elif ext in ['.csv', '.txt', '.asc']:
        data = np.loadtxt(filepath, delimiter=',', dtype=int)

    else:
        raise ValueError(f"Unsupported classification file format: {ext}")

    # Validate shape
    if data.shape != (ny, nx):
        raise ValueError(
            f"Classification map shape {data.shape} doesn't match "
            f"grid size ({ny}, {nx})"
        )

    return data.astype(int)
```

#### 1.2. Name-Based Material Lookup

**Current limitation**: Classification maps must use integer class IDs.

**Desired capability**: Use material names directly in classification data.

**Options**:

**Option A: Metadata file**
```yaml
# classification_metadata.yaml
colormap:
  0: "Desert Sand (Depth-Varying)"
  50: "Basalt (Weathered to Fresh)"
  100: "Lunar Regolith Analog"
  200: "Granite"
```

**Option B: String arrays** (for formats that support it)
```python
# classification.npy contains strings
material_names = np.array([
    ["Desert Sand", "Desert Sand", "Basalt"],
    ["Desert Sand", "Basalt", "Granite"],
])
```

**Option C: RGB to material mapping**
```yaml
# For PNG files with specific colors
color_to_material:
  [255, 200, 100]: "Desert Sand (Depth-Varying)"  # Sandy color
  [50, 50, 50]: "Basalt (Weathered to Fresh)"     # Dark gray
  [200, 200, 200]: "Granite"                       # Light gray
```

**Recommendation**: Support all three, with Option A being most flexible.

#### 1.3. Validation and Reporting

**Needed checks**:
```python
def validate_classification_map(material_class, material_db):
    """Validate classification map against database."""
    unique_classes = np.unique(material_class)

    # Check all materials exist
    missing = []
    for class_id in unique_classes:
        if class_id not in material_db.materials:
            missing.append(class_id)

    if missing:
        raise ValueError(f"Materials not found in database: {missing}")

    # Report statistics
    print(f"Classification map statistics:")
    print(f"  Total grid points: {material_class.size}")
    print(f"  Unique materials: {len(unique_classes)}")

    for class_id in unique_classes:
        count = np.sum(material_class == class_id)
        fraction = count / material_class.size
        material = material_db.get_material(class_id)
        print(f"    {material.name}: {count} pixels ({fraction*100:.1f}%)")
```

#### 1.4. Visualization Tools

**Create utility to preview classification maps**:
```python
# scripts/visualize_classification.py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_classification_map(material_class, material_db, output_file=None):
    """
    Plot material classification map with legend.

    Args:
        material_class: 2D array of material IDs
        material_db: Material database
        output_file: Optional output filename
    """
    unique_classes = np.unique(material_class)

    # Create colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    cmap = ListedColormap(colors)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(material_class, cmap=cmap, interpolation='nearest')

    # Add colorbar with material names
    cbar = plt.colorbar(im, ax=ax, ticks=unique_classes)
    labels = [material_db.get_material(c).name for c in unique_classes]
    cbar.ax.set_yticklabels(labels)

    ax.set_title('Material Classification Map')
    ax.set_xlabel('X (grid points)')
    ax.set_ylabel('Y (grid points)')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    else:
        plt.show()
```

### Configuration Updates

**Proposed YAML configuration**:
```yaml
materials:
  type: "from_classification"
  classification_file: "data/classifications/terrain_materials.tif"

  # Optional: Metadata for name/color mapping
  classification_metadata: "data/classifications/terrain_materials.yaml"

  # Database options
  use_sqlite_database: true
  sqlite_database_path: "data/materials/materials.db"
```

---

## 2. Diffuse Material Boundaries

### Physical Motivation

**Problem**: Sharp material transitions create unrealistic thermal signatures.

In nature, material boundaries are typically gradual due to:
- Mixing (e.g., sand grading into gravel)
- Weathering (e.g., exposed bedrock to fractured to soil)
- Biological activity (e.g., vegetation/bare soil transitions)
- Transport processes (e.g., wind, water depositing sediments)

**Typical transition widths**: 0.5 - 5 meters depending on process

### Approach Comparison

#### Option A: Gaussian Smoothing

**Description**: Apply spatial filter to material properties after classification.

**Pros**:
- Simple to implement (~20 lines of code)
- Fast (scipy.ndimage.gaussian_filter)
- User-controllable transition width
- Physically reasonable for most natural gradients

**Cons**:
- Not based on specific physical model
- Same smoothing applied to all boundaries
- Can smooth across domain boundaries if not careful

**Implementation**:
```python
from scipy.ndimage import gaussian_filter

def smooth_material_boundaries(material_field, smoothing_length_m, dx, dy):
    """
    Apply Gaussian smoothing to material properties.

    Args:
        material_field: MaterialField or MaterialFieldDepthVarying
        smoothing_length_m: Transition width in meters
        dx, dy: Grid spacing in meters
    """
    # Convert smoothing length to pixels
    sigma_x = smoothing_length_m / dx
    sigma_y = smoothing_length_m / dy

    # Smooth each property (don't smooth in depth direction)
    # Surface properties
    material_field.alpha = gaussian_filter(
        material_field.alpha,
        sigma=(sigma_y, sigma_x)
    )
    material_field.epsilon = gaussian_filter(
        material_field.epsilon,
        sigma=(sigma_y, sigma_x)
    )
    material_field.roughness = gaussian_filter(
        material_field.roughness,
        sigma=(sigma_y, sigma_x)
    )

    # Subsurface properties (smooth in x,y but not z)
    for iz in range(material_field.nz):
        material_field.k[:, :, iz] = gaussian_filter(
            material_field.k[:, :, iz],
            sigma=(sigma_y, sigma_x)
        )
        material_field.rho[:, :, iz] = gaussian_filter(
            material_field.rho[:, :, iz],
            sigma=(sigma_y, sigma_x)
        )
        material_field.cp[:, :, iz] = gaussian_filter(
            material_field.cp[:, :, iz],
            sigma=(sigma_y, sigma_x)
        )
```

**Complexity**: Low
**Effort**: 1-2 hours
**Accuracy**: Good for most applications

#### Option B: Distance-Based Blending

**Description**: For boundary pixels, compute weighted average based on distance to pure material regions.

**Pros**:
- Respects actual material distribution
- Can handle complex geometries naturally
- More physically motivated than simple smoothing

**Cons**:
- More computationally expensive (requires distance transforms)
- Need to define "boundary region" width
- Still somewhat arbitrary

**Implementation sketch**:
```python
from scipy.ndimage import distance_transform_edt

def blend_material_boundaries(material_field, material_class, blend_width_m, dx, dy):
    """
    Blend material properties at boundaries using distance-weighted averaging.

    Args:
        material_field: MaterialField instance
        material_class: 2D array of material IDs
        blend_width_m: Width of blending zone in meters
        dx, dy: Grid spacing
    """
    # Identify boundary pixels (where neighbors differ)
    boundaries = detect_boundaries(material_class)

    # For each material, compute distance field
    unique_materials = np.unique(material_class)
    distance_fields = {}

    for mat_id in unique_materials:
        # Distance to nearest pixel of this material
        mask = (material_class == mat_id)
        dist = distance_transform_edt(~mask) * dx  # Convert to meters
        distance_fields[mat_id] = dist

    # For each boundary pixel, compute weighted average
    blend_width_pixels = blend_width_m / dx

    for j in range(material_field.ny):
        for i in range(material_field.nx):
            if not boundaries[j, i]:
                continue

            # Compute weights (inverse distance)
            weights = {}
            total_weight = 0.0

            for mat_id in unique_materials:
                d = distance_fields[mat_id][j, i]
                if d < blend_width_m:
                    w = 1.0 / (1.0 + d)
                    weights[mat_id] = w
                    total_weight += w

            # Normalize weights
            for mat_id in weights:
                weights[mat_id] /= total_weight

            # Blend properties
            k_blend = 0.0
            rho_blend = 0.0
            # ... etc for all properties

            for mat_id, w in weights.items():
                material = material_db.get_material(mat_id)
                k_blend += w * material.k
                rho_blend += w * material.rho
                # ... etc

            material_field.k[j, i, :] = k_blend
            material_field.rho[j, i, :] = rho_blend
            # ... etc
```

**Complexity**: Medium
**Effort**: 4-6 hours
**Accuracy**: Better than Gaussian for complex geometries

#### Option C: Effective Medium Theory

**Description**: Use physics-based mixing models for composite materials.

**Available Models**:

1. **Arithmetic Mean (Parallel layers)**:
   ```
   k_eff = f₁·k₁ + f₂·k₂ + ...
   ```
   - Heat flow parallel to layers
   - Upper bound for mixtures

2. **Harmonic Mean (Series layers)**:
   ```
   k_eff = 1 / (f₁/k₁ + f₂/k₂ + ...)
   ```
   - Heat flow perpendicular to layers
   - Lower bound for mixtures
   - Already used in solver for depth interfaces!

3. **Geometric Mean (Well-mixed)**:
   ```
   k_eff = k₁^f₁ · k₂^f₂ · ...
   ```
   - Good for randomly mixed materials
   - Common in geophysics

4. **Hashin-Shtrikman Bounds**:
   - Tightest bounds on effective properties
   - Accounts for microstructure

5. **Maxwell-Garnett** (Inclusions in matrix):
   ```
   k_eff = k_m · (k_i + 2k_m + 2f_i(k_i - k_m)) / (k_i + 2k_m - f_i(k_i - k_m))
   ```
   - One material (i) dispersed in matrix (m)
   - Good for particle suspensions

**Pros**:
- Physically rigorous
- Different models for different scenarios
- Published literature values for validation

**Cons**:
- Requires understanding material mixing physics
- Need to know volume fractions, not just distances
- More complex to implement correctly

**Implementation**:
```python
def effective_conductivity(materials_dict, mixing_model='geometric'):
    """
    Compute effective thermal conductivity for material mixture.

    Args:
        materials_dict: {material_id: volume_fraction}
        mixing_model: 'arithmetic', 'harmonic', 'geometric', 'hs_upper', 'hs_lower'

    Returns:
        k_effective: Effective thermal conductivity
    """
    if mixing_model == 'arithmetic':
        k_eff = sum(f * material.k for material, f in materials_dict.items())

    elif mixing_model == 'harmonic':
        k_eff = 1.0 / sum(f / material.k for material, f in materials_dict.items())

    elif mixing_model == 'geometric':
        k_eff = np.prod([material.k**f for material, f in materials_dict.items()])

    elif mixing_model == 'hs_upper':
        # Hashin-Shtrikman upper bound
        # Implementation requires sorting by conductivity
        pass

    # ... similar for rho, cp (usually arithmetic mean is fine)

    return k_eff
```

**Complexity**: High
**Effort**: 8-12 hours (including research and validation)
**Accuracy**: Best for known mixing scenarios

#### Option D: Subgrid Parameterization

**Description**: Explicitly store material fractions at each grid point.

**Data structure**:
```python
# Instead of single material ID per pixel
class MaterialFractions:
    def __init__(self, ny, nx, nz):
        # Dictionary of fraction arrays for each material
        self.fractions = {}  # {material_id: np.ndarray(ny, nx)}

    def add_material(self, material_id, fraction_map):
        self.fractions[material_id] = fraction_map

    def get_effective_properties(self, j, i, mixing_model='geometric'):
        """Compute effective properties at grid point (j, i)"""
        materials_at_point = {}
        for mat_id, frac_map in self.fractions.items():
            if frac_map[j, i] > 0:
                materials_at_point[mat_id] = frac_map[j, i]

        return effective_properties(materials_at_point, mixing_model)
```

**Input format**: Multiple layers in GeoTIFF or separate files per material
```
sand_fraction.tif    -> values 0.0-1.0
basalt_fraction.tif  -> values 0.0-1.0
granite_fraction.tif -> values 0.0-1.0
```

**Pros**:
- Most accurate representation
- Flexible - can use any mixing model
- Natural for remote sensing data (e.g., spectral unmixing)

**Cons**:
- Requires fractional material data (not always available)
- Increased memory usage
- More complex input data preparation

**Complexity**: High
**Effort**: 10-15 hours
**Accuracy**: Best if data available

---

## Recommendations

### Phase 1: Enable Classification Maps (Priority: High)

**Scope**: Make existing functionality usable with real-world data.

**Tasks**:
1. Add file format readers (GeoTIFF, PNG, NumPy)
   - Create `src/classification_io.py` module
   - Support integer and RGB classification maps
2. Add name-based material lookup
   - Optional metadata file (YAML)
   - Fallback to integer IDs
3. Update configuration parsing
   - Add `classification_metadata` option
4. Create validation and reporting
   - Check all materials exist
   - Print statistics
5. Create visualization tool
   - `scripts/visualize_classification.py`
6. Create example classification maps
   - Simple 3-material test case
   - Add to `data/classifications/`
7. Update documentation
   - User guide section on classification maps
   - Example YAML configurations

**Estimated effort**: 4-6 hours
**Deliverables**:
- Working classification map loader
- Example maps and configs
- Documentation

### Phase 2: Add Boundary Smoothing (Priority: Medium)

**Scope**: Implement Gaussian smoothing as default method.

**Tasks**:
1. Implement Gaussian smoothing function
   - Add to `MaterialField` and `MaterialFieldDepthVarying`
   - Optional method called after classification
2. Add configuration options
   ```yaml
   materials:
     smooth_boundaries: true
     smoothing_length_m: 2.0
     smoothing_method: "gaussian"  # Future: "distance", "effective_medium"
   ```
3. Add boundary detection
   - Helper function to identify boundary pixels
   - Option to exclude certain boundaries from smoothing
4. Testing
   - Verify smooth transitions
   - Check energy conservation
5. Documentation
   - Technical guide on boundary handling
   - Physical interpretation of smoothing parameter

**Estimated effort**: 3-4 hours
**Deliverables**:
- Gaussian smoothing implementation
- Configuration options
- Documentation

### Phase 3: Advanced Mixing Models (Priority: Low)

**Scope**: Add effective medium theory for specialized applications.

**Tasks**:
1. Research and implement mixing models
   - Arithmetic, harmonic, geometric means
   - Hashin-Shtrikman bounds (optional)
2. Add `mixing_model` configuration option
3. Validate against literature
4. Document when to use each model

**Estimated effort**: 6-8 hours
**Deliverables**:
- Multiple mixing model options
- Validation cases
- Technical documentation

### Phase 4: Subgrid Parameterization (Priority: Future)

**Scope**: Support explicit material fractions (if needed).

**Trigger**: User has fractional material data from remote sensing

**Estimated effort**: 10-12 hours

---

## Configuration Examples

### Example 1: Simple Classification Map

```yaml
simulation:
  name: "multi_material_demo"
  start_time: "2024-06-21T06:00:00"
  duration_hours: 24
  time_step: 1800

site:
  latitude: 35.0
  longitude: 106.0
  altitude: 1500.0
  timezone_offset: -7.0

terrain:
  type: "flat"
  nx: 50
  ny: 50
  dx: 10.0
  dy: 10.0
  flat_elevation: 0.0

materials:
  type: "from_classification"
  classification_file: "data/classifications/desert_terrain.tif"
  use_sqlite_database: true
  sqlite_database_path: "data/materials/materials.db"

# ... rest of config
```

### Example 2: Classification with Smoothing

```yaml
materials:
  type: "from_classification"
  classification_file: "data/classifications/desert_terrain.png"

  # Material name mapping
  classification_metadata: "data/classifications/desert_terrain.yaml"

  # Database
  use_sqlite_database: true
  sqlite_database_path: "data/materials/materials.db"

  # Boundary smoothing
  smooth_boundaries: true
  smoothing_length_m: 2.5  # 2.5 meter transition zones
  smoothing_method: "gaussian"
```

### Example 3: Advanced - Effective Medium

```yaml
materials:
  type: "from_classification"
  classification_file: "data/classifications/mixed_terrain.tif"
  use_sqlite_database: true

  # Advanced boundary handling
  smooth_boundaries: true
  smoothing_method: "effective_medium"
  mixing_model: "geometric"  # or "harmonic", "arithmetic"
  smoothing_length_m: 1.5
```

---

## File Structure

### New Files to Create

```
thermal_sim/
├── src/
│   └── classification_io.py          # NEW: Classification map readers
│
├── scripts/
│   ├── visualize_classification.py   # NEW: Visualization tool
│   └── create_classification_map.py  # NEW: Helper to create test maps
│
├── data/
│   └── classifications/               # NEW: Classification map data
│       ├── README.md                  # Guide for creating maps
│       ├── simple_3material.tif       # Example map
│       ├── simple_3material.yaml      # Metadata
│       └── desert_scene.tif           # More complex example
│
└── docs/
    └── user-guide/
        └── classification_maps.md     # NEW: User documentation
```

### Modified Files

```
src/materials.py          # Add smoothing methods
src/config.py            # Add smoothing config options
src/runner.py            # Call classification loader
docs/user-guide/configuration.md  # Document new options
```

---

## Testing Strategy

### Unit Tests

1. **Classification loading**:
   - Load from each file format
   - Validate shape checking
   - Test name-to-ID mapping

2. **Boundary smoothing**:
   - Verify Gaussian kernel application
   - Check edge handling
   - Verify depth dimension not smoothed

3. **Effective medium**:
   - Test each mixing model
   - Compare to analytical solutions
   - Verify volume fraction normalization

### Integration Tests

1. **Full simulation with classification map**:
   - Create simple 2-material test case
   - Run 24-hour simulation
   - Verify temperature is continuous across boundary (if smoothed)

2. **Energy conservation**:
   - Verify smoothing doesn't break energy balance
   - Check against uniform material case

### Validation Tests

1. **Thermal response comparison**:
   - Sharp boundary vs smoothed boundary
   - Verify smoothing reduces artifacts
   - Compare to analytical solutions if available

---

## Open Questions

1. **File formats**: Which formats are most important? (GeoTIFF seems essential, PNG useful)

2. **Boundary smoothing default**: Should smoothing be ON or OFF by default?

3. **Smoothing length**: What's a good default? 1-2 grid cells? Or physical distance like 2m?

4. **Sharp transitions**: Do we need ability to mark certain boundaries as sharp (e.g., road/terrain)?

5. **Validation**: Do we have test cases with known material distributions to validate against?

6. **Performance**: For large grids, is smoothing performance acceptable? (Gaussian filtering is O(N) so should be fine)

7. **Mixing models**: Which effective medium model should be default? (Geometric mean seems most general)

---

## References

### Material Mixing Models

1. **Hashin, Z. & Shtrikman, S. (1962)**. "A variational approach to the theory of the effective magnetic permeability of multiphase materials." *Journal of Applied Physics*, 33(10), 3125-3131.

2. **Clauser, C. & Huenges, E. (1995)**. "Thermal Conductivity of Rocks and Minerals." *Rock Physics & Phase Relations: A Handbook of Physical Constants*, AGU Reference Shelf 3.

3. **Côté, J. & Konrad, J.M. (2005)**. "A generalized thermal conductivity model for soils and construction materials." *Canadian Geotechnical Journal*, 42(2), 443-458.

4. **Carson, J.K. et al. (2005)**. "Thermal conductivity bounds for isotropic, porous materials." *International Journal of Heat and Mass Transfer*, 48(11), 2150-2158.

### Remote Sensing for Material Classification

5. **Gillespie, A.R. et al. (1998)**. "Temperature and Emissivity Separation from Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) Images." *IEEE Transactions on Geoscience and Remote Sensing*, 36(4), 1113-1126.

6. **Ramsey, M.S. & Christensen, P.R. (1998)**. "Mineral abundance determination: Quantitative deconvolution of thermal emission spectra." *JGR: Solid Earth*, 103(B1), 577-596.

---

## Implementation Priority

| Phase | Priority | Effort | Value | When |
|-------|----------|--------|-------|------|
| Classification Map Loading | **HIGH** | 4-6h | High | Now |
| Gaussian Smoothing | **MEDIUM** | 3-4h | Medium | After Phase 1 |
| Effective Medium Models | **LOW** | 6-8h | Low-Med | When needed |
| Subgrid Fractions | **FUTURE** | 10-12h | Low | If requested |

**Recommended next step**: Implement Phase 1 (Classification Map Loading) to enable immediate use of spatially-varying materials with existing infrastructure.
