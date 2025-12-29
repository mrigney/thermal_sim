# Solar Radiation Algorithms

## Overview

This document describes the algorithms implemented in the solar radiation module ([src/solar.py](../src/solar.py)) for computing solar position, irradiance, and shadows on terrain.

## Table of Contents

1. [Solar Position Calculation](#solar-position-calculation)
2. [Extraterrestrial Irradiance](#extraterrestrial-irradiance)
3. [Clear Sky Irradiance Models](#clear-sky-irradiance-models)
4. [Surface Irradiance](#surface-irradiance)
5. [Shadow Computation](#shadow-computation)
6. [Shadow Caching Strategy](#shadow-caching-strategy)

---

## Solar Position Calculation

### Algorithm: Simplified Solar Position (Michalsky 1988)

**Purpose**: Calculate solar azimuth and elevation angles for any date, time, and location.

**Accuracy**: ±0.01° (sufficient for thermal simulation and shadow casting)

**Inputs**:
- Latitude φ [degrees, -90 to 90, north positive]
- Longitude λ [degrees, -180 to 180, east positive]
- Date and time (UTC or local with timezone offset)

**Outputs**:
- Solar azimuth α [degrees, 0-360, measured clockwise from north]
- Solar elevation h [degrees, -90 to 90, positive above horizon]

### Mathematical Formulation

#### Step 1: Julian Day Number

Convert calendar date to Julian Day (JD):

```
For date: year Y, month M, day D, time H (decimal hours)

If M ≤ 2:
    Y = Y - 1
    M = M + 12

A = floor(Y / 100)
B = 2 - A + floor(A / 4)

JD = floor(365.25 × (Y + 4716)) + floor(30.6001 × (M + 1)) + D + H/24 + B - 1524.5
```

#### Step 2: Julian Century

Julian centuries from J2000.0 epoch:

```
T = (JD - 2451545.0) / 36525.0
```

#### Step 3: Solar Longitude and Anomaly

Geometric mean longitude of sun:
```
L₀ = (280.46646 + 36000.76983×T + 0.0003032×T²) mod 360°
```

Geometric mean anomaly:
```
M = (357.52911 + 35999.05029×T - 0.0001537×T²) mod 360°
```

#### Step 4: Equation of Center

Sun's equation of center (accounts for elliptical orbit):
```
C = (1.914602 - 0.004817×T - 0.000014×T²) × sin(M)
  + (0.019993 - 0.000101×T) × sin(2M)
  + 0.000289 × sin(3M)
```

True longitude:
```
λ_true = L₀ + C
```

Apparent longitude (corrected for nutation):
```
Ω = 125.04 - 1934.136×T
λ_sun = λ_true - 0.00569 - 0.00478 × sin(Ω)
```

#### Step 5: Solar Declination

Obliquity of ecliptic:
```
ε = 23.439291 - 0.0130042×T - 0.00000164×T² + 0.000000504×T³
```

Solar declination:
```
δ = arcsin(sin(ε) × sin(λ_sun))
```

#### Step 6: Equation of Time

```
y = tan²(ε/2)

EoT = 4 × [y×sin(2L₀) - 2e×sin(M) + 4e×y×sin(M)×cos(2L₀)
          - 0.5×y²×sin(4L₀) - 1.25×e²×sin(2M)]
```

where e = Earth's orbital eccentricity ≈ 0.01671

EoT is in degrees, convert to minutes by multiplying by 4.

#### Step 7: Hour Angle

True solar time:
```
TST = H × 60 + EoT + 4×λ    [minutes]
```

Hour angle:
```
ω = (TST / 4) - 180°
```

#### Step 8: Solar Position

Solar zenith angle:
```
cos(θz) = sin(φ) × sin(δ) + cos(φ) × cos(δ) × cos(ω)
```

Solar elevation:
```
h = 90° - θz
```

Solar azimuth (measured clockwise from north):
```
cos(α) = (sin(δ) - sin(φ) × cos(θz)) / (cos(φ) × sin(θz))

If ω > 0 (afternoon): α = 360° - arccos(cos(α))
If ω ≤ 0 (morning):   α = arccos(cos(α))
```

### Sun Vector

Convert to Cartesian unit vector pointing toward sun:

```
x = sin(α) × cos(h)    (East component)
y = cos(α) × cos(h)    (North component)
z = sin(h)             (Up component)
```

**References**:
- Michalsky, J.J. (1988). "The Astronomical Almanac's algorithm for approximate solar position (1950-2050)." *Solar Energy*, 40(3), 227-235.
- Meeus, J. (1998). *Astronomical Algorithms*, 2nd Ed. Willmann-Bell, Inc.

---

## Extraterrestrial Irradiance

### Solar Constant Correction

The solar constant varies throughout the year due to Earth's elliptical orbit.

**Solar constant**: I_sc = 1361 W/m² (Kopp & Lean, 2011)

**Distance correction factor**:

```
Γ = 2π × (n - 1) / 365    [day angle, n = day of year]

r_factor = 1.000110 + 0.034221×cos(Γ) + 0.001280×sin(Γ)
         + 0.000719×cos(2Γ) + 0.000077×sin(2Γ)
```

**Extraterrestrial irradiance**:
```
I₀ = I_sc × r_factor
```

Typical range: 1322 - 1412 W/m²

**References**:
- Kopp, G. & Lean, J.L. (2011). "A new, lower value of total solar irradiance." *Geophysical Research Letters*, 38, L01706.
- Spencer, J.W. (1971). "Fourier series representation of the position of the sun." *Search*, 2(5), 172.

---

## Clear Sky Irradiance Models

Two clear-sky models are implemented for calculating direct beam and diffuse irradiance.

### Model 1: Simplified Model (Meinel & Meinel 1976)

**Suitable for**: Quick calculations, educational purposes

#### Direct Beam Irradiance

Atmospheric transmittance:

```
Pressure ratio: p_r = exp(-h_alt / 8400)    [h_alt in meters]

Air mass (Kasten & Young 1989):
AM = p_r / (sin(h) + 0.50572 × (h + 6.07995)^(-1.6364))

Beam transmittance:
τ_b = 0.56 × (exp(-0.65 × AM) + exp(-0.095 × AM))

Direct irradiance:
I_direct = I₀ × τ_b
```

#### Diffuse Irradiance

```
Diffuse transmittance:
τ_d = 0.271 - 0.294 × τ_b

Diffuse irradiance (on horizontal):
I_diffuse = I₀ × τ_d × sin(h)
```

### Model 2: Ineichen-Perez Model (Default)

**Suitable for**: High-fidelity simulations, research

More sophisticated model accounting for aerosols, water vapor, and altitude.

#### Linke Turbidity Factor

Represents atmospheric clarity:
- TL = 2.0 : Very clear, dry atmosphere
- TL = 3.0 : Clear sky
- TL = 5.0 : Moderately turbid
- TL = 8.0 : Very turbid (dust, pollution)

Estimated from visibility V [km]:
```
TL = 2.0 + (50 - V) / 10.0
TL ∈ [1.5, 8.0]
```

#### Altitude Corrections

```
f_h1 = exp(-h_alt / 8000)     [for beam transmittance]
f_h2 = exp(-h_alt / 1250)     [for diffuse component]
```

#### Direct Beam Irradiance

Relative optical air mass:
```
AM = 1 / (sin(h) + 0.50572 × (h + 6.07995)^(-1.6364))
```

Beam transmittance:
```
τ_b = exp(-0.8662 × TL × AM × f_h1)
```

Direct beam irradiance:
```
I_direct = I₀ × τ_b
```

#### Diffuse Irradiance

Diffuse fraction:
```
F_d = 1 - τ_b
```

Diffuse irradiance:
```
I_diffuse = I₀ × sin(h) × F_d × f_h2
```

**Typical values**:
- Clear day at sea level, h=60°: I_direct ≈ 900 W/m², I_diffuse ≈ 100 W/m²
- Turbid atmosphere: I_direct decreases, I_diffuse increases

**References**:
- Ineichen, P. & Perez, R. (2002). "A new airmass independent formulation for the Linke turbidity coefficient." *Solar Energy*, 73(3), 151-157.
- Kasten, F. & Young, A.T. (1989). "Revised optical air mass tables and approximation formula." *Applied Optics*, 28(22), 4735-4738.

---

## Surface Irradiance

### Total Irradiance on Inclined Surface

For a surface with normal vector **n** and sky view factor SVF:

```
I_total = I_direct_component + I_diffuse_component + I_reflected
```

#### Direct Component

```
cos(θ_i) = n · s    [s = sun unit vector]

I_direct_component = I_direct × max(0, cos(θ_i))    if sunlit
                   = 0                                if shadowed
```

#### Diffuse Component (Isotropic Sky Model)

Assumes uniform radiance from sky hemisphere:

```
I_diffuse_component = I_diffuse × SVF
```

where SVF = sky view factor (0 = fully obstructed, 1 = open sky)

#### Ground-Reflected Component

Currently neglected (typically <5% for natural terrain):

```
I_reflected ≈ ρ_ground × I_global × (1 - SVF)
```

where ρ_ground ≈ 0.2 for typical terrain.

**Note**: Future enhancement could include anisotropic sky models (e.g., Perez model) for better accuracy on tilted surfaces.

**References**:
- Liu, B.Y.H. & Jordan, R.C. (1960). "The interrelationship and characteristic distribution of direct, diffuse and total solar radiation." *Solar Energy*, 4(3), 1-19.

---

## Shadow Computation

### Algorithm: Horizon Angle Ray Marching

**Purpose**: Determine which terrain points are shadowed by terrain features

**Method**: For each grid point, cast a ray toward the sun and check if terrain blocks the sun.

### Procedure

For each point (i, j) on terrain:

1. **Get sun direction**:
   ```
   Sun azimuth α, elevation h
   Ray direction in grid: (dx_ray, dy_ray) = (sin(α), cos(α))
   Vertical slope: tan(h)
   ```

2. **March along ray**:
   ```
   For steps s = 1, 2, 3, ... until boundary or max distance:

       Position along ray:
       i_ray = i + s × step_size × dx_ray
       j_ray = j + s × step_size × dy_ray

       Interpolate terrain elevation z_terrain at (i_ray, j_ray)

       Compute sun ray height:
       distance = s × step_size × grid_spacing
       z_sun = z_here + distance × tan(h)

       If z_terrain > z_sun:
           Point is shadowed
           Break
   ```

3. **Bilinear interpolation** for sub-grid terrain elevation:
   ```
   Given fractional indices (i_ray, j_ray):

   i₀ = floor(i_ray), i₁ = ceil(i_ray)
   j₀ = floor(j_ray), j₁ = ceil(j_ray)
   fx = i_ray - i₀
   fy = j_ray - j₀

   z_interp = (1-fx)(1-fy)×z[j₀,i₀] + fx(1-fy)×z[j₀,i₁]
            + (1-fx)fy×z[j₁,i₀] + fx×fy×z[j₁,i₁]
   ```

### Parameters

- **step_size**: Fraction of grid cell for ray marching (default: 0.5)
  - Smaller = more accurate but slower
  - Larger = faster but may miss shadows

- **max_distance**: Maximum shadow casting distance [meters]
  - Limits computational cost
  - Typical: 500-2000 m for rolling terrain
  - None = check entire domain (expensive!)

### Complexity

- **Time**: O(N × M) where N = grid points, M = ray marching steps
- **Memory**: O(N) for shadow map storage
- **Typical**: For 1000×1000 grid with 200 ray steps: ~200M operations

### Limitations

1. **Self-shadowing accuracy**: Sub-grid terrain features not captured
2. **Ray marching artifacts**: Aliasing on shadow boundaries
3. **Computational cost**: Expensive for large domains

**Future enhancements**:
- GPU-accelerated ray tracing
- Adaptive step size
- Shadow boundary anti-aliasing
- Hierarchical terrain representation (octree/quadtree)

**References**:
- Corripio, J.G. (2003). "Vectorial algebra algorithms for calculating terrain parameters from DEMs and solar radiation modelling in mountainous terrain." *International Journal of Geographical Information Science*, 17(1), 1-23.

---

## Shadow Caching Strategy

### Motivation

Shadow computation is expensive (~1-10 seconds per shadow map for large domains). However, solar position changes slowly from day to day:

- Solar declination: ~0.25°/day (varies seasonally)
- At 40°N latitude: ~0.5° change in azimuth/elevation per day

Therefore, we can pre-compute shadows and reuse them.

### Strategy

**Pre-computation**:
1. Compute shadow maps at discrete time intervals (e.g., every 15 minutes)
2. Store shadow maps for one day
3. Reuse for multiple consecutive days
4. Recalculate when sun position drift exceeds threshold (~1-2°)

**Storage requirements**:

For one day at 15-minute intervals:
```
Number of shadow maps: 96
Grid size: 10,000 × 10,000 = 10⁸ points
Storage per map: 10⁸ bits = 12.5 MB
Total per day: 96 × 12.5 MB = 1.2 GB
```

Manageable for modern systems. Can store 7 days in ~8.4 GB.

### Cache Structure

```python
class ShadowCache:
    times: List[datetime]           # Timestamps
    sun_positions: List[(az, el)]   # Solar positions
    shadow_maps: List[ndarray]      # Boolean arrays
```

### Usage

**Pre-compute for one day**:
```python
cache = ShadowCache()
cache.compute_daily_shadows(
    terrain_elevation, dx, dy,
    latitude, longitude, date,
    time_step_minutes=15
)
```

**Retrieve during simulation**:
```python
shadow_map, az, el = cache.get_shadow_map(current_time, interpolate=True)
```

### Reuse Policy

Recalculate shadow maps when:
1. Starting a new simulation
2. Sun position differs by > 1° from cached values
3. Every N days (N = 3-7 depending on season)

During equinoxes (March/September): More frequent updates
During solstices (June/December): Less frequent updates

### Benefits

- **Speed**: 1000× faster than computing every time step
- **Accuracy**: Minimal error (<1° position change acceptable)
- **Memory**: Manageable (~1-2 GB per day)

**Trade-offs**:
- Initial computation time: ~1-5 minutes for large domain
- Memory usage: 1-10 GB for multi-day cache
- Small errors near shadow boundaries during reuse

---

## Algorithm Performance Summary

| Component | Algorithm | Accuracy | Speed | Notes |
|-----------|-----------|----------|-------|-------|
| **Solar Position** | Michalsky 1988 | ±0.01° | <1 ms | Excellent for 1950-2050 |
| **Irradiance** | Ineichen-Perez | ±10% | <1 ms | Clear sky only |
| **Shadows** | Ray marching | ~1 m | 1-10 s | Grid-dependent |
| **Shadow Cache** | Pre-computation | ±1° | 1000× speedup | Reusable 3-7 days |

## Validation Recommendations

To validate the solar module:

1. **Solar Position**: Compare with NOAA Solar Calculator or pyephem
2. **Irradiance**: Compare with measured data or PVLIB-Python
3. **Shadows**: Visual inspection, compare with GIS shadow tools
4. **Energy Conservation**: Integrate I_direct + I_diffuse over day, compare to known values

## Future Enhancements

1. **Cloud cover models** for partially cloudy conditions
2. **Spectral irradiance** for wavelength-dependent radiation
3. **Terrain-reflected radiation** (inter-reflection between slopes)
4. **Atmospheric refraction** for low sun angles
5. **GPU-accelerated shadow computation** using OptiX or CUDA
6. **Adaptive shadow resolution** based on terrain roughness

---

## References

### Solar Position
- Michalsky, J.J. (1988). Solar Energy, 40(3), 227-235.
- Reda, I. & Andreas, A. (2004). NREL Technical Report NREL/TP-560-34302.

### Irradiance Models
- Ineichen, P. & Perez, R. (2002). Solar Energy, 73(3), 151-157.
- Bird, R.E. & Hulstrom, R.L. (1981). SERI Technical Report SERI/TR-642-761.
- Kasten, F. & Young, A.T. (1989). Applied Optics, 28(22), 4735-4738.

### Shadow Computation
- Corripio, J.G. (2003). International Journal of GIS, 17(1), 1-23.
- Kumar, L. et al. (1997). International Journal of GIS, 11(5), 475-497.

### General References
- Duffie, J.A. & Beckman, W.A. (2013). *Solar Engineering of Thermal Processes*, 4th Ed.
- Iqbal, M. (1983). *An Introduction to Solar Radiation*. Academic Press.
