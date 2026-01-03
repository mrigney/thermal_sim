# Atmospheric Conditions Module - Algorithm Documentation

**Module**: `src/atmosphere.py`
**Author**: Thermal Terrain Simulator Team
**Date**: December 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Physical Background](#physical-background)
3. [Sky Temperature Models](#sky-temperature-models)
4. [Convective Heat Transfer](#convective-heat-transfer)
5. [Wind Profile Models](#wind-profile-models)
6. [Vapor Pressure Calculations](#vapor-pressure-calculations)
7. [Implementation Details](#implementation-details)
8. [References](#references)

---

## 1. Overview

The atmosphere module provides atmospheric state management for thermal terrain simulations. It handles:

- **Air temperature** (constant or time-varying)
- **Wind speed** with height adjustment
- **Relative humidity** and vapor pressure
- **Cloud cover** effects
- **Sky temperature** for longwave radiation
- **Convective heat transfer coefficients**

### Role in Energy Balance

Atmospheric conditions affect the surface energy balance through:

1. **Longwave radiation**: Q_atm = ε_surface · σ · SVF · **T_sky⁴**
2. **Convection**: Q_conv = **h_conv** · (T_air - T_surface)

where T_sky and h_conv depend on atmospheric state.

---

## 2. Physical Background

### 2.1 Atmospheric Radiation

The atmosphere is **not** a blackbody. It emits longwave radiation primarily from:
- Water vapor (H₂O) - dominant contributor
- Carbon dioxide (CO₂)
- Ozone (O₃) in stratosphere

**Key concept**: Effective sky temperature (T_sky) represents the temperature of an equivalent blackbody that would emit the same downward longwave radiation as the actual atmosphere.

**Typical values**:
- Clear desert sky: T_sky ≈ T_air - 30 to 40 K
- Humid conditions: T_sky ≈ T_air - 10 to 20 K
- Overcast sky: T_sky ≈ T_air (clouds are nearly blackbodies)

### 2.2 Convective Heat Transfer

Heat transfer between surface and air occurs through:
- **Forced convection**: Wind-driven turbulent mixing (dominant when U > 2 m/s)
- **Free convection**: Buoyancy-driven (when surface much warmer than air)
- **Mixed regime**: Both mechanisms active

The convective heat transfer coefficient h_conv parameterizes this complex process.

---

## 3. Sky Temperature Models

### 3.1 Simple Offset Model

**Formula**:
```
T_sky = T_air - ΔT
```

where ΔT ≈ 20 K is a constant offset.

**Pros**:
- Extremely simple
- Computationally fast

**Cons**:
- No physics basis
- Ignores humidity effects
- Poor accuracy (errors up to 50%)

**Use case**: Quick estimates only, not recommended for production.

---

### 3.2 Swinbank (1963) Model

**Formula**:
```
T_sky = 0.0552 · T_air^1.5
```

**Derivation**:
Empirical fit to clear-sky radiation measurements in England.

**Example**:
For T_air = 300 K:
```
T_sky = 0.0552 · (300)^1.5
      = 0.0552 · 5196.15
      = 286.8 K
ΔT = 300 - 286.8 = 13.2 K
```

**Pros**:
- Simple single-parameter model
- Reasonable for temperate climates

**Cons**:
- No humidity dependence (implicitly assumes moderate moisture)
- Derived for maritime climate (not ideal for deserts)
- Sky depression often underestimated in arid regions

**Accuracy**: ±5-10 K in temperate regions, larger errors in extreme climates

---

### 3.3 Brunt (1932) Model

**Formula**:

Step 1: Calculate clear-sky atmospheric emissivity
```
ε_clear = 0.52 + 0.065 · √(e_a / 100)
```
where e_a is vapor pressure in Pa (divide by 100 to convert to mbar).

Step 2: Apply cloud correction
```
ε_cloud = ε_clear · (1 + 0.22 · N²)
```
where N is cloud fraction [0, 1].

Step 3: Calculate effective sky temperature
```
T_sky = ε_cloud^(1/4) · T_air
```

**Physical basis**:
- ε_clear accounts for water vapor absorption/emission
- Higher vapor pressure → higher emissivity → warmer sky
- Cloud correction: clouds are nearly blackbodies (ε ≈ 1)

**Example**:
For T_air = 300 K, RH = 30%, N = 0 (clear):

```
e_sat = 611.2 · exp(17.67 · 26.85 / (26.85 + 243.5))
      = 3564 Pa

e_a = 0.3 · 3564 = 1069 Pa = 10.69 mbar

ε_clear = 0.52 + 0.065 · √10.69
        = 0.52 + 0.065 · 3.27
        = 0.732

ε_cloud = 0.732 · (1 + 0) = 0.732

T_sky = (0.732)^0.25 · 300
      = 0.919 · 300
      = 275.7 K

ΔT = 300 - 275.7 = 24.3 K
```

**Pros**:
- Physics-based
- Accounts for humidity
- Handles cloud cover
- Widely validated

**Cons**:
- Requires humidity data
- Moderate computational cost
- Coefficients are empirical

**Accuracy**: ±3-5 K in most conditions

---

### 3.4 Idso-Jackson (1969) Model

**Formula**:

Step 1: Calculate clear-sky emissivity
```
ε_clear = 1 - 0.261 · exp(-7.77 × 10⁻⁴ · (273 - T_air)²)
```

Step 2: Apply cloud correction (same as Brunt)
```
ε_cloud = ε_clear · (1 + 0.22 · N²)
```

Step 3: Calculate sky temperature
```
T_sky = ε_cloud^(1/4) · T_air
```

**Physical basis**:
Derived from atmospheric radiation measurements in Arizona desert. The exponential form better captures emissivity variation with temperature in low-humidity conditions.

**Example**:
For T_air = 300 K, N = 0:

```
ε_clear = 1 - 0.261 · exp(-7.77e-4 · (273 - 300)²)
        = 1 - 0.261 · exp(-7.77e-4 · 729)
        = 1 - 0.261 · exp(-0.566)
        = 1 - 0.261 · 0.568
        = 1 - 0.148
        = 0.852

T_sky = (0.852)^0.25 · 300
      = 0.961 · 300
      = 288.3 K

ΔT = 300 - 288.3 = 11.7 K
```

**Comparison with Brunt**:
- Idso typically gives **higher** T_sky (warmer) than Brunt
- Better for arid/desert climates
- Brunt better for humid regions

**Pros**:
- Excellent for desert conditions
- Based on dry climate measurements
- Good physical form

**Cons**:
- No explicit humidity dependence (implicit in temperature)
- Less accurate in very humid conditions

**Accuracy**: ±2-4 K in arid climates

---

### 3.5 Model Selection Guidelines

| Condition | Recommended Model | Typical ΔT |
|-----------|-------------------|------------|
| Desert, clear sky | Idso-Jackson | 25-40 K |
| Temperate, clear | Brunt | 15-25 K |
| Humid, clear | Brunt | 10-20 K |
| Any, overcast | Brunt with N=1 | 0-5 K |
| Quick estimate | Swinbank | 10-20 K |

---

## 4. Convective Heat Transfer

### 4.1 Physical Regimes

**Forced convection** (wind-driven):
```
Q_conv ~ U^n · (T_air - T_surface)
```
where n ≈ 0.6-1.0 depending on flow regime.

**Free convection** (buoyancy):
```
Q_conv ~ (T_surface - T_air)^(5/4)
```

**Mixed regime**: More complex, combination of both.

### 4.2 Dimensionless Numbers

**Reynolds number** (inertia vs viscous):
```
Re = ρ · U · L / μ
```

**Grashof number** (buoyancy vs viscous):
```
Gr = g · β · ΔT · L³ / ν²
```

**Criterion for regime**:
- Forced dominant: Gr/Re² << 1
- Free dominant: Gr/Re² >> 1
- Mixed: Gr/Re² ≈ 1

For outdoor terrain with U > 1 m/s, typically **forced convection dominates**.

---

## 4.3 McAdams Correlation (1954)

**Formula**:
```
h_conv = 5.7 + 3.8 · U    [W/(m²·K)]
```

where U is wind speed in m/s.

**Derivation**:
Empirical fit to turbulent forced convection data for flat plates.

**Examples**:
| Wind Speed | h_conv |
|------------|--------|
| 0 m/s | 5.7 W/(m²·K) |
| 2 m/s | 13.3 W/(m²·K) |
| 5 m/s | 24.7 W/(m²·K) |
| 10 m/s | 43.7 W/(m²·K) |

**Pros**:
- Simple and robust
- Widely used benchmark
- Good for moderate winds

**Cons**:
- No natural convection term
- Independent of surface orientation
- No length scale dependence

**Valid range**: U > 1 m/s (forced convection regime)

---

## 4.4 Jurges Correlation (1924)

**Formula**:
```
h_conv = 2.8 + 3.0 · U    [W/(m²·K)]
```

**Application**:
Developed for building energy calculations, slightly more conservative than McAdams.

**Comparison with McAdams**:
- Lower natural convection term (2.8 vs 5.7)
- Lower wind coefficient (3.0 vs 3.8)
- Results in 10-30% lower h_conv

**Use case**: Building surfaces, conservative estimates

---

## 4.5 Watmuff Correlation (1977)

**Formula**:

For low wind (U < 1 m/s, mixed convection):
```
h_conv = 2.8 + 3.0 · U
```

For forced convection (U ≥ 1 m/s):
```
h_conv = 8.6 · U^0.6 / L^0.4
```

where L is characteristic length [m].

**Physical basis**:
- Developed for flat-plate solar collectors
- Accounts for length scale effects
- Smooth transition between regimes

**Example**:
For U = 5 m/s, L = 1 m:
```
h_conv = 8.6 · (5)^0.6 / (1)^0.4
       = 8.6 · 2.627 / 1
       = 22.6 W/(m²·K)
```

**Length scale effects**:
| U = 5 m/s | L = 0.1 m | L = 1 m | L = 10 m |
|-----------|-----------|---------|----------|
| h_conv | 35.8 | 22.6 | 14.2 W/(m²·K) |

Smaller features have higher h_conv (thinner boundary layer).

**Pros**:
- Best for outdoor thermal applications
- Includes length scale physics
- Validated for heated surfaces

**Cons**:
- Requires characteristic length estimate
- More complex than McAdams

**Recommendation**: **Use Watmuff for terrain simulations**

---

## 4.6 Typical Values Summary

| Condition | h_conv [W/(m²·K)] |
|-----------|-------------------|
| Calm air, indoor | 5-10 |
| Light breeze (2 m/s) | 10-15 |
| Moderate wind (5 m/s) | 20-30 |
| Strong wind (10 m/s) | 35-50 |
| Very strong wind (20 m/s) | 60-100 |

For reference:
- Radiation: h_rad ≈ 4εσT³ ≈ 5-7 W/(m²·K) for typical surfaces
- At low wind, convection and radiation are comparable
- At high wind, convection dominates

---

## 5. Wind Profile Models

### 5.1 Logarithmic Wind Profile

**Formula**:
```
U(z) / U(z_ref) = ln(z / z₀) / ln(z_ref / z₀)
```

where:
- U(z) = wind speed at height z
- U(z_ref) = wind speed at reference height z_ref
- z₀ = surface roughness length

**Physical basis**:
Valid in the **surface layer** (lowest 10-100m of atmosphere) under **neutral stability** (no strong heating/cooling).

**Assumptions**:
1. Constant shear stress with height
2. Neutral atmospheric stability
3. Flat, homogeneous terrain
4. z > 10·z₀ (above roughness sublayer)

**Example**:
Given U(2m) = 5 m/s over short grass (z₀ = 0.01 m), find U(10m):

```
U(10) = U(2) · ln(10 / 0.01) / ln(2 / 0.01)
      = 5 · ln(1000) / ln(200)
      = 5 · 6.908 / 5.298
      = 6.52 m/s
```

Wind increases ~30% from 2m to 10m height.

### 5.2 Surface Roughness Length (z₀)

Representative values:

| Surface Type | z₀ [m] | Description |
|--------------|--------|-------------|
| Open water | 0.0001 | Very smooth |
| Sand, snow | 0.001 | Smooth |
| Mowed grass | 0.01 | Short vegetation |
| Crops, long grass | 0.1 | Tall vegetation |
| Shrubs | 0.5 | Low obstacles |
| Forest, urban | 1-2 | Large obstacles |

**Rule of thumb**: z₀ ≈ h/10 where h is obstacle height.

### 5.3 Limitations

The log profile is **NOT valid** when:
- Strong surface heating (unstable, convective boundary layer)
- Strong surface cooling (stable, suppressed turbulence)
- Complex terrain (hills, valleys)
- Non-uniform surface

For these cases, more sophisticated models are needed (Monin-Obukhov similarity theory, mass-consistent models, CFD).

---

## 6. Vapor Pressure Calculations

### 6.1 Saturation Vapor Pressure

**Tetens Formula** (1930):
```
e_sat = 611.2 · exp(17.67 · T_C / (T_C + 243.5))    [Pa]
```

where T_C is temperature in °C.

**Alternative (August-Roche-Magnus)**:
```
e_sat = 611.2 · exp(17.27 · T_C / (T_C + 237.3))    [Pa]
```

Both are accurate to ±0.1% for -40°C < T < 50°C.

**Example**:
For T = 300 K (26.85°C):
```
e_sat = 611.2 · exp(17.67 · 26.85 / (26.85 + 243.5))
      = 611.2 · exp(17.67 · 0.0993)
      = 611.2 · exp(1.754)
      = 611.2 · 5.78
      = 3534 Pa
```

### 6.2 Actual Vapor Pressure

Given relative humidity RH [0-1]:
```
e_a = RH · e_sat(T)
```

**Example**:
For RH = 60%, T = 300 K:
```
e_a = 0.6 · 3534 = 2120 Pa = 21.2 mbar
```

### 6.3 Dewpoint Temperature

Temperature at which air becomes saturated (RH = 100%):

```
T_dew = 243.5 · ln(e_a / 611.2) / (17.67 - ln(e_a / 611.2))    [°C]
```

**Example**:
For e_a = 2120 Pa:
```
T_dew = 243.5 · ln(2120 / 611.2) / (17.67 - ln(2120 / 611.2))
      = 243.5 · 1.242 / (17.67 - 1.242)
      = 243.5 · 1.242 / 16.428
      = 18.4°C
```

This is the temperature at which dew/frost will form on surfaces.

---

## 7. Implementation Details

### 7.1 AtmosphericConditions Class

**Design philosophy**:
- Flexible input: constants, callables, or (future) file-based
- Lazy evaluation: compute derived quantities on demand
- Clean interface: separate physics from application

**Key methods**:
```python
get_air_temperature(time) -> float
get_wind_speed(time, height=None) -> float
get_relative_humidity(time) -> float
get_vapor_pressure(time) -> float
get_sky_temperature(time, model='brunt') -> float
get_convection_coefficient(time, roughness, correlation='watmuff') -> float
get_atmospheric_state(time) -> dict  # All quantities
```

### 7.2 Time-Varying Inputs

**Constant**:
```python
atm = AtmosphericConditions(T_air=300.0, wind_speed=5.0)
```

**Callable**:
```python
def T_diurnal(t):
    hour = t.hour + t.minute/60.0
    return 295.0 + 10.0 * np.sin(np.pi * (hour - 6) / 12)

atm = AtmosphericConditions(T_air=T_diurnal, wind_speed=5.0)
```

**Helper functions**:
```python
T_func = create_diurnal_temperature(T_mean=300, T_amp=12)
wind_func = create_diurnal_wind(wind_mean=5.0, wind_amp=2.0)
atm = AtmosphericConditions(T_air=T_func, wind_speed=wind_func)
```

### 7.3 Integration with Energy Balance

In the solver, the atmosphere module provides:

```python
# At each time step
T_air = atmosphere.get_air_temperature(current_time)
T_sky = atmosphere.get_sky_temperature(current_time, model='idso')
h_conv = atmosphere.get_convection_coefficient(current_time)

# Compute heat fluxes
Q_atm = epsilon_surface * SIGMA * SVF * T_sky**4
Q_conv = h_conv * (T_air - T_surface)

# Update temperature field
dT/dt = (Q_solar + Q_atm - Q_emission + Q_conv + Q_cond) / (rho * cp)
```

---

## 8. References

### Sky Temperature Models

1. **Swinbank, W. C.** (1963). Long-wave radiation from clear skies. *Quarterly Journal of the Royal Meteorological Society*, 89(381), 339-348.

2. **Brunt, D.** (1932). Notes on radiation in the atmosphere. *Quarterly Journal of the Royal Meteorological Society*, 58(247), 389-420.

3. **Idso, S. B., & Jackson, R. D.** (1969). Thermal radiation from the atmosphere. *Journal of Geophysical Research*, 74(23), 5397-5403.

4. **Duffie, J. A., & Beckman, W. A.** (2013). *Solar Engineering of Thermal Processes* (4th ed.). Wiley. Chapter 2.

### Convective Heat Transfer

5. **McAdams, W. H.** (1954). *Heat Transmission* (3rd ed.). McGraw-Hill.

6. **Jurges, W.** (1924). Der Wärmeübergang an einer ebenen Wand. *Gesundh. Ing. Beih. Reihe* 1, Nr. 19.

7. **Watmuff, J. H., Charters, W. W. S., & Proctor, D.** (1977). Solar and wind induced external coefficients for solar collectors. *Revue Internationale d'Héliotechnique*, 2, 56.

8. **Incropera, F. P., et al.** (2011). *Fundamentals of Heat and Mass Transfer* (7th ed.). Wiley. Chapter 7 & 9.

### Atmospheric Boundary Layer

9. **Stull, R. B.** (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers.

10. **Kaimal, J. C., & Finnigan, J. J.** (1994). *Atmospheric Boundary Layer Flows*. Oxford University Press.

### Vapor Pressure

11. **Tetens, O.** (1930). Über einige meteorologische Begriffe. *Zeitschrift für Geophysik*, 6, 297-309.

12. **Buck, A. L.** (1981). New equations for computing vapor pressure and enhancement factor. *Journal of Applied Meteorology*, 20(12), 1527-1532.

---

## Appendix: Validation Data

### A.1 Sky Temperature Comparisons

For T_air = 300 K:

| RH | Brunt T_sky | Idso T_sky | Simple (ΔT=20K) | Swinbank |
|----|-------------|------------|-----------------|----------|
| 10% | 270.8 K | 288.3 K | 280.0 K | 286.8 K |
| 30% | 275.7 K | 288.3 K | 280.0 K | 286.8 K |
| 60% | 281.4 K | 288.3 K | 280.0 K | 286.8 K |
| 90% | 287.0 K | 288.3 K | 280.0 K | 286.8 K |

**Observations**:
- Idso is humidity-independent (captures temperature effect only)
- Brunt shows strong humidity dependence (physically correct)
- Swinbank and Simple don't account for humidity

### A.2 Convective Coefficient Comparisons

For U = 5 m/s:

| Correlation | h_conv [W/(m²·K)] |
|-------------|-------------------|
| McAdams | 24.7 |
| Jurges | 17.8 |
| Watmuff (L=1m) | 22.6 |

All three agree within ±15%, which is within typical measurement uncertainty.

---

**End of Algorithm Documentation**
