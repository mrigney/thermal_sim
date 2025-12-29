# Thermal Solver Algorithms

**Document Version**: 3.0
**Date**: December 22, 2025
**Module**: `src/solver.py`
**Project Version**: See [CHANGELOG.md](../CHANGELOG.md) for complete version history

**Document Changelog**:
- **v3.0 (Dec 22, 2025)**: Added optional lateral surface conduction (2D+1D solver) with operator splitting
- **v2.0 (Dec 21, 2025)**: Implemented von Neumann (flux) boundary condition at surface, removed separate surface temperature update, improved energy conservation
- **v1.0 (Dec 18, 2025)**: Initial version with Dirichlet surface BC

**Major Bug Fixes** (v2.0):
- Fixed energy conservation issue by switching from Dirichlet to von Neumann BC
- Eliminated runaway surface heating problem
- Reduced energy conservation error from ~1e-6 to <1e-8
- See [CHANGELOG.md](../CHANGELOG.md) for details

This document provides detailed mathematical formulations and numerical algorithms for the thermal solver module, which integrates all physics components to compute spatially-resolved temperature evolution on terrain.

---

## Table of Contents

1. [Overview](#overview)
2. [Subsurface Grid](#subsurface-grid)
3. [Temperature Field Data Structure](#temperature-field-data-structure)
4. [Surface Energy Balance](#surface-energy-balance)
5. [Subsurface Heat Conduction](#subsurface-heat-conduction)
6. [Semi-Implicit Time Stepping (IMEX)](#semi-implicit-time-stepping-imex)
7. [Thomas Algorithm for Tridiagonal Systems](#thomas-algorithm-for-tridiagonal-systems)
8. [Main Solver Integration](#main-solver-integration)
9. [Lateral Surface Conduction (Optional, v3.0+)](#lateral-surface-conduction-optional-v30)
10. [Numerical Stability and Accuracy](#numerical-stability-and-accuracy)
11. [Implementation Notes](#implementation-notes)
12. [References](#references)

---

## 1. Overview

The thermal solver integrates the subsurface heat equation with surface energy balance using a **von Neumann (flux) boundary condition**:

**Subsurface Heat Equation (1D at each surface point):**
```
ρ·c_p·∂T/∂t = ∂/∂z(k·∂T/∂z)
```

**Surface Boundary Condition (flux continuity):**
```
-k·∂T/∂z|_surface = Q_net = Q_solar + Q_atm + Q_emission + Q_conv
```

Where:
- `T(z)` = temperature at depth z [K] (z=0 is surface)
- `Q_net` = net energy flux into surface [W/m²]
- `ρ` = density [kg/m³]
- `c_p` = specific heat capacity [J/(kg·K)]
- `k` = thermal conductivity [W/(m·K)]

The numerical scheme employs:
- **Crank-Nicolson** for subsurface diffusion with **von Neumann BC at surface**
- **Thomas algorithm** for tridiagonal systems
- Surface temperature (layer 0) emerges from the heat equation

This approach provides:
- **Automatic energy conservation** (no artificial coupling)
- **Unconditional stability**
- **Second-order accuracy** in space, first-order in time
- Suitable for time steps dt = 60-120 seconds

---

## 2. Subsurface Grid

### 2.1 Stretched Grid Generation

The subsurface grid uses **geometric stretching** to achieve fine resolution near the surface (where temperature gradients are largest) and coarser resolution at depth (for computational efficiency).

**Grid Parameters:**
- `z_max` = maximum depth [m] (default: 0.5 m)
- `n_layers` = number of subsurface layers (default: 20)
- `stretch_factor` = geometric progression ratio (default: 1.2)

**Geometric Progression:**

Layer interfaces are generated using:
```
z_i = z_max · (r^i - 1) / (r^n - 1)    for i = 0, 1, ..., n
```

Where:
- `r` = stretch_factor
- `i` = interface index (0 = surface, n = bottom)
- `z_0 = 0` (surface)
- `z_n = z_max` (bottom boundary)

**Node-Centered Discretization:**

Cell centers (nodes) are computed as midpoints:
```
z_j = (z_{j-1} + z_j) / 2    for j = 1, 2, ..., n
```

**Cell Spacing:**
```
dz_j = z_j - z_{j-1}
```

**Typical Values (z_max=0.5m, n=20, r=1.2):**
- Surface layer: `dz_1 ≈ 0.008 m`
- Mid-depth: `dz_10 ≈ 0.020 m`
- Bottom layer: `dz_20 ≈ 0.045 m`

### 2.2 Fourier Number Constraint

The **Fourier number** characterizes the ratio of heat diffusion rate to thermal storage:

```
Fo = α·dt / dz²
```

Where:
- `α = k/(ρ·c_p)` = thermal diffusivity [m²/s]
- `dt` = time step [s]
- `dz` = minimum cell spacing [m]

**Stability:**
- Explicit schemes: require `Fo ≤ 0.5` (very restrictive)
- Crank-Nicolson: **unconditionally stable** for all `Fo`

**Accuracy:**
- Recommended: `Fo ~ 1-10` for good temporal resolution
- Too small (`Fo << 1`): many time steps required, inefficient
- Too large (`Fo >> 10`): temporal oscillations, poor accuracy

**Implementation:**

The `SubsurfaceGrid` class includes automatic Fourier number checking:

```python
def check_fourier_number(self, alpha, dt):
    """Check Fourier number for temporal accuracy."""
    dz_min = np.min(self.dz)
    Fo = alpha * dt / dz_min**2

    if Fo < 0.1:
        print(f"WARNING: Fo={Fo:.3f} is very small. Time step may be inefficient.")
    elif Fo > 50:
        print(f"WARNING: Fo={Fo:.3f} is very large. Temporal accuracy may be poor.")

    return Fo
```

### 2.3 Skin Depth Criterion

The domain depth must capture the **diurnal thermal penetration depth** (skin depth):

```
δ = √(2·α·T_period / π)
```

For typical materials:
- Sand: `α ≈ 3.5e-7 m²/s`, `T_period = 86400 s` → `δ ≈ 0.165 m`
- Rock: `α ≈ 1.0e-6 m²/s` → `δ ≈ 0.28 m`

**Recommended:**
```
z_max ≥ 3·δ
```

This ensures temperature variations decay to < 5% at the bottom boundary.

**Default Choice:**
- `z_max = 0.5 m` captures 3× skin depth for sand, 1.8× for rock
- Adequate for diurnal cycles in desert terrain

---

## 3. Temperature Field Data Structure

The `TemperatureField` class stores the complete thermal state:

```python
class TemperatureField:
    T_surface: np.ndarray      # Shape: (ny, nx)
    T_subsurface: np.ndarray   # Shape: (ny, nx, nz)
```

**Layout:**
- `T_surface[j, i]` = temperature at surface point (j, i) = `T_subsurface[j, i, 0]`
- `T_subsurface[j, i, k]` = temperature at subsurface layer k, point (j, i)
- `k = 0` → **surface layer**
- `k = 1` → first subsurface layer below surface
- `k = nz-1` → deepest layer (at z_max)

**Important Change (v2.0):**

The surface is now **layer 0 of the subsurface grid**, not a separate entity. This ensures:
- Single unified temperature field
- Automatic energy conservation through flux BC
- No artificial coupling coefficients needed

---

## 4. Surface Energy Balance

### 4.1 Energy Flux Components

The `compute_energy_balance()` function calculates all energy fluxes at the surface:

**1. Solar Radiation Absorption:**
```
Q_solar = α · (S_direct·cos(θ)·f_shadow + S_diffuse·SVF)
```

Where:
- `α` = solar absorptivity (material property)
- `S_direct` = direct beam irradiance [W/m²]
- `S_diffuse` = diffuse irradiance [W/m²]
- `θ` = solar incidence angle (from surface normal and sun vector)
- `f_shadow` = shadow factor (0 = shadowed, 1 = sunlit)
- `SVF` = sky view factor

**Incidence Angle:**
```
cos(θ) = max(0, n̂ · ŝ)
```
- `n̂` = surface unit normal
- `ŝ` = unit vector toward sun
- `max(0, ...)` ensures only illuminated surfaces absorb direct radiation

**2. Atmospheric Longwave Radiation:**
```
Q_atm = ε · σ · SVF · T_sky⁴
```

Where:
- `ε` = thermal emissivity (material property)
- `σ = 5.67e-8 W/(m²·K⁴)` = Stefan-Boltzmann constant
- `T_sky` = effective sky temperature [K]

**3. Thermal Emission:**
```
Q_emission = -ε · σ · T_surface⁴
```

Negative sign indicates energy loss from the surface.

**4. Convective Heat Transfer:**
```
Q_conv = h_conv · (T_air - T_surface)
```

Where:
- `h_conv` = convective heat transfer coefficient [W/(m²·K)]
- `T_air` = air temperature [K]

**Net Surface Flux:**
```
Q_net = Q_solar + Q_atm + Q_emission + Q_conv
```

This represents the energy available to:
1. Heat/cool the surface layer
2. Conduct into the subsurface

### 4.2 Implementation

```python
def compute_energy_balance(T_surface, terrain, materials,
                          solar_irradiance_direct, solar_irradiance_diffuse,
                          sun_vector_xyz, shadow_map, T_sky, T_air, h_conv):
    """
    Compute all energy balance components at the surface.

    Returns:
    --------
    Q_solar : np.ndarray (ny, nx)
        Solar radiation absorption [W/m²]
    Q_atm : np.ndarray (ny, nx)
        Atmospheric longwave radiation [W/m²]
    Q_emission : np.ndarray (ny, nx)
        Thermal emission [W/m²]
    Q_conv : np.ndarray (ny, nx)
        Convective heat transfer [W/m²]
    Q_net : np.ndarray (ny, nx)
        Net surface flux [W/m²]
    """
    # Solar incidence angle
    cos_theta = np.maximum(0, np.sum(terrain.normals * sun_vector_xyz, axis=2))

    # Solar absorption
    Q_solar = materials.absorptivity * (
        solar_irradiance_direct * cos_theta * shadow_map +
        solar_irradiance_diffuse * terrain.sky_view_factor
    )

    # Longwave radiation
    sigma = 5.67e-8  # W/(m²·K⁴)
    Q_atm = materials.emissivity * sigma * terrain.sky_view_factor * T_sky**4
    Q_emission = -materials.emissivity * sigma * T_surface**4

    # Convection
    Q_conv = h_conv * (T_air - T_surface)

    # Net flux
    Q_net = Q_solar + Q_atm + Q_emission + Q_conv

    return Q_solar, Q_atm, Q_emission, Q_conv, Q_net
```

---

## 5. Subsurface Heat Conduction

### 5.1 Governing Equation

The 1D vertical heat equation at each surface point (j, i):

```
ρ·c_p·∂T/∂t = ∂/∂z(k·∂T/∂z)
```

For constant material properties:
```
∂T/∂t = α·∂²T/∂z²
```

Where `α = k/(ρ·c_p)` is thermal diffusivity.

### 5.2 Crank-Nicolson Discretization

The **Crank-Nicolson** method uses an average of implicit and explicit schemes:

```
(T^{n+1} - T^n) / dt = α/2 · [∂²T^{n+1}/∂z² + ∂²T^n/∂z²]
```

This is:
- **Second-order accurate** in both time and space
- **Unconditionally stable** for all dt and dz
- **Implicit**, requiring a linear system solve

### 5.3 Finite Difference Form

At node k (k = 1, ..., nz, where k=1 is first subsurface layer):

**Spatial Derivative (centered difference):**
```
∂²T/∂z²|_k ≈ (T_{k-1} - 2·T_k + T_{k+1}) / dz_k²
```

**Crank-Nicolson Time Stepping:**
```
T_k^{n+1} - T_k^n     α   [ (T_{k-1}^{n+1} - 2·T_k^{n+1} + T_{k+1}^{n+1})
─────────────────  =  ─── · ─────────────────────────────────────────────
       dt            2   [              dz_k²

                          (T_{k-1}^n - 2·T_k^n + T_{k+1}^n) ]
                        + ─────────────────────────────────  ]
                                    dz_k²                    ]
```

**Rearrange to standard form:**
```
-r_{k-1}·T_{k-1}^{n+1} + (1 + 2·r_k)·T_k^{n+1} - r_{k+1}·T_{k+1}^{n+1}
    = r_{k-1}·T_{k-1}^n + (1 - 2·r_k)·T_k^n + r_{k+1}·T_{k+1}^n
```

Where:
```
r_k = α·dt / (2·dz_k²)
```

This is a **tridiagonal system**: `A·T^{n+1} = B·T^n + boundary terms`

### 5.4 Boundary Conditions

**Upper Boundary (surface, k=0) - von Neumann (v2.0):**

The surface energy balance provides a **flux boundary condition**:
```
-k·∂T/∂z|_surface = Q_net
```

Discretized energy balance at surface (layer 0):
```
ρ·c_p·dz[0]·(T[0]^{n+1} - T[0]^n)/dt = Q_net + k·(T[1]^{n+1} - T[0]^{n+1})/dz[0]
```

Using Crank-Nicolson (θ = 0.5) for the conduction term:
```
ρ·c_p·dz[0]·(T[0]^{n+1} - T[0]^n)/dt = Q_net
    + θ·k·(T[1]^{n+1} - T[0]^{n+1})/dz[0]
    + (1-θ)·k·(T[1]^n - T[0]^n)/dz[0]
```

This gives the first row of the tridiagonal system with Q_net as a source term.

**Lower Boundary (depth z_max, k=nz-1):**
- Zero-flux (insulated): `∂T/∂z|_{z=z_max} = 0`
- Implementation: `T_{nz+1} = T_{nz}` (ghost point)
- Same as v1.0

**Tridiagonal System Form:**

For nz subsurface layers, the system is:

```
┌                                    ┐   ┌       ┐     ┌       ┐
│ b₁  c₁   0   0  ···  0   0   0   0│   │ T₁^{n+1}│     │ d₁    │
│ a₂  b₂  c₂   0  ···  0   0   0   0│   │ T₂^{n+1}│     │ d₂    │
│  0  a₃  b₃  c₃  ···  0   0   0   0│   │ T₃^{n+1}│     │ d₃    │
│  ⋮   ⋮   ⋮   ⋮   ⋱   ⋮   ⋮   ⋮   ⋮│ · │   ⋮    │  =  │  ⋮    │
│  0   0   0   0  ··· aₙ bₙ  cₙ   0│   │ Tₙ^{n+1}│     │ dₙ    │
│  0   0   0   0  ···  0  aₙ₊₁ bₙ₊₁ │   │Tₙ₊₁^{n+1}│   │dₙ₊₁   │
└                                    ┘   └       ┘     └       ┘
```

Where:
- `aₖ = -r_{k-1}` (lower diagonal)
- `bₖ = 1 + 2·r_k` (main diagonal)
- `cₖ = -r_{k+1}` (upper diagonal)
- `d₁ = RHS₁ + a₁·T_surface^{n+1}` (includes surface BC)
- `dₙ₊₁ = RHS_{nz}` (zero-flux BC makes ghost point disappear)

### 5.5 Implementation (v2.0 - von Neumann BC)

```python
def solve_subsurface_tridiagonal(T_prev, Q_surface_net, thermal_diffusivity,
                                rho_cp, subsurface_grid, dt):
    """
    Solve 1D subsurface heat equation using Crank-Nicolson with flux BC.

    Parameters:
    -----------
    T_prev : np.ndarray (ny, nx, nz)
        Subsurface temperature at previous time
        Note: T_prev[:,:,0] is the surface temperature
    Q_surface_net : np.ndarray (ny, nx)
        Net energy flux into surface [W/m²] (boundary condition)
    thermal_diffusivity : np.ndarray (ny, nx)
        Thermal diffusivity α = k/(ρ·c_p) [m²/s]
    rho_cp : np.ndarray (ny, nx)
        Volumetric heat capacity ρ·c_p [J/(m³·K)]
    subsurface_grid : SubsurfaceGrid
        Grid specification
    dt : float
        Time step [s]

    Returns:
    --------
    T_new : np.ndarray (ny, nx, nz)
        Updated subsurface temperature
        T_new[:,:,0] is the updated surface temperature
    """
    ny, nx, nz = T_prev.shape
    T_new = np.zeros_like(T_prev)
    dz = subsurface_grid.dz

    # Loop over all surface points
    for j in range(ny):
        for i in range(nx):
            alpha = thermal_diffusivity[j, i]
            rho_cp_val = rho_cp[j, i]
            Q_net = Q_surface_net[j, i]

            # Build tridiagonal system with flux BC at k=0
            a, b, c, d = build_tridiagonal_with_flux_bc(
                T_prev[j, i, :], Q_net, alpha, rho_cp_val, dz, dt
            )

            # Solve using Thomas algorithm
            T_new[j, i, :] = solve_thomas(a, b, c, d)

    return T_new
```

**Key Differences from v1.0:**
- Takes `Q_surface_net` instead of `T_surface_new`
- Requires `rho_cp` to convert flux to temperature rate
- Returns surface temperature as `T_new[:,:,0]`

---

## 6. Energy Conservation and Flux BC (v2.0)

### 6.1 Why von Neumann BC?

The **von Neumann (flux) boundary condition** at the surface provides several advantages over the previous Dirichlet BC approach:

**1. Automatic Energy Conservation:**
- Net flux into surface = conductive flux into subsurface (by construction)
- No artificial coupling coefficients needed
- Energy balance is exact to discretization error

**2. Physical Consistency:**
- Matches true physics: energy crosses the surface, not temperature
- Surface temperature emerges from heat equation
- No separate surface layer needed

**3. Simpler Numerics:**
- Single solve (no iteration needed)
- No linearization approximations
- Fewer numerical parameters

### 6.2 Handling Nonlinear Emission

The emission term `Q_emission = -ε·σ·T⁴` is nonlinear, but we treat it **explicitly**:

```
Q_net^n = Q_solar^n + Q_atm^n - ε·σ·(T^n)⁴ + Q_conv^n
```

This is accurate because:
- Emission varies slowly (time scale ~100s)
- Time step dt = 60-120s is smaller than emission time scale
- Error is O(dt), acceptable for this term

**No linearization needed** - the flux is simply evaluated at the old temperature and used as the boundary condition.

### 6.3 Comparison with v1.0

**v1.0 (Dirichlet BC):**
```
1. Solve subsurface with T_surface^n as BC
2. Update T_surface^{n+1} using surface energy balance
3. Iterate 1-2 until convergence (or fixed 2 iterations)
```

**v2.0 (von Neumann BC):**
```
1. Compute Q_net^n from surface energy balance
2. Solve subsurface with Q_net^n as BC
   → Surface temperature T^{n+1}[0] emerges from solution
```

**Advantages:**
- No iteration
- Guaranteed energy conservation
- Simpler code (one function instead of two)

---

## 7. Thomas Algorithm for Tridiagonal Systems

### 7.1 Algorithm Description

The **Thomas algorithm** (tridiagonal matrix algorithm) is a specialized Gaussian elimination method for tridiagonal systems:

```
A·x = d
```

Where A is tridiagonal:
```
│ b₁  c₁   0   0  ···  0  │   │ x₁ │     │ d₁ │
│ a₂  b₂  c₂   0  ···  0  │   │ x₂ │     │ d₂ │
│  0  a₃  b₃  c₃  ···  0  │ · │ x₃ │  =  │ d₃ │
│  ⋮   ⋮   ⋮   ⋮   ⋱   ⋮  │   │  ⋮ │     │  ⋮ │
│  0   0   0   0  ··· bₙ  │   │ xₙ │     │ dₙ │
```

**Forward Elimination:**

```
c'₁ = c₁ / b₁
c'ᵢ = cᵢ / (bᵢ - aᵢ·c'ᵢ₋₁)    for i = 2, ..., n-1

d'₁ = d₁ / b₁
d'ᵢ = (dᵢ - aᵢ·d'ᵢ₋₁) / (bᵢ - aᵢ·c'ᵢ₋₁)    for i = 2, ..., n
```

**Back Substitution:**

```
xₙ = d'ₙ
xᵢ = d'ᵢ - c'ᵢ·xᵢ₊₁    for i = n-1, ..., 1
```

**Complexity:**
- O(n) operations (vs O(n³) for general matrices)
- Very efficient for large systems

### 7.2 Implementation

```python
def solve_thomas(a, b, c, d):
    """
    Solve tridiagonal system using Thomas algorithm.

    Parameters:
    -----------
    a : np.ndarray (n,)
        Lower diagonal (a[0] is not used)
    b : np.ndarray (n,)
        Main diagonal
    c : np.ndarray (n,)
        Upper diagonal (c[n-1] is not used)
    d : np.ndarray (n,)
        Right-hand side

    Returns:
    --------
    x : np.ndarray (n,)
        Solution
    """
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n-1):
        denom = b[i] - a[i] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

    d_prime[n-1] = (d[n-1] - a[n-1] * d_prime[n-2]) / (b[n-1] - a[n-1] * c_prime[n-2])

    # Back substitution
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x
```

---

## 8. Main Solver Integration

### 8.1 ThermalSolver Class

The `ThermalSolver` class coordinates all components:

```python
class ThermalSolver:
    def __init__(self, terrain, materials, atmosphere, shadow_cache,
                 latitude, longitude, altitude=0.0,
                 subsurface_grid=None, dt=120.0):
        """
        Main thermal solver for terrain simulations.

        Parameters:
        -----------
        terrain : TerrainGrid
        materials : MaterialField
        atmosphere : AtmosphericConditions
        shadow_cache : ShadowCache
        latitude, longitude : float
            Location [degrees]
        altitude : float
            Elevation above sea level [m]
        subsurface_grid : SubsurfaceGrid (optional)
            Default: z_max=0.5m, n_layers=20
        dt : float
            Time step [s] (default: 120s)
        """
```

### 8.2 Time Stepping Algorithm (v2.0)

The main time stepping sequence in `step()`:

```
For each time step:
    1. Get atmospheric conditions (T_air, wind, T_sky, h_conv)
    2. Compute solar position and irradiance
    3. Get shadow map from cache
    4. Compute surface energy balance using T^n
       → Computes Q_net = Q_solar + Q_atm + Q_emission + Q_conv
    5. Solve subsurface heat equation with flux BC (Crank-Nicolson)
       → Surface BC: -k·∂T/∂z|_surface = Q_net
       → Updates entire T_subsurface^{n+1} including surface (layer 0)
    6. Extract surface temperature: T_surface^{n+1} = T_subsurface^{n+1}[:,:,0]
    7. Store new temperature field
    8. Advance time: t → t + dt
```

**Key Changes from v1.0:**
- Only **one solve** per time step (no surface update iteration)
- Surface temperature emerges from subsurface solver
- Q_net evaluated explicitly at old temperature
- Order is simpler and more natural

### 8.3 Multi-Day Simulation

The `run()` method provides a generator for long simulations:

```python
def run(self, start_time, end_time, output_interval=3600.0):
    """
    Run simulation from start_time to end_time.

    Yields:
    -------
    (time, temp_field) at each output_interval

    Example:
    --------
    >>> solver = ThermalSolver(...)
    >>> solver.initialize(T_initial=300.0)
    >>> for time, temps in solver.run(start_time, end_time, output_interval=3600):
    ...     # Process/save results every hour
    ...     visualize(temps.T_surface)
    ```

This allows:
- Memory-efficient processing of long simulations
- Periodic output/checkpointing
- User callbacks for analysis/visualization

---

## 9. Lateral Surface Conduction (Optional, v3.0+)

### 9.1 Overview

Starting in **version 3.0**, the thermal solver supports optional **2D+1D lateral heat conduction** at the surface, allowing heat to flow horizontally between adjacent cells in addition to vertical subsurface conduction.

**Key Features:**
- **Optional**: Disabled by default (fully backward compatible)
- **Operator splitting**: Explicit lateral + implicit vertical
- **Zero-flux boundaries**: Insulated lateral domain edges (Neumann BC)
- **Energy conserving**: < 1e-4 relative error with lateral conduction enabled
- **Configurable**: Adjustable lateral conductivity factor for anisotropic materials

**Use Cases:**
- Shadows creating sharp temperature gradients
- Localized heating/cooling features (rocks, vegetation)
- Validation against measured temperature distributions
- Realistic heat spreading in heterogeneous terrain

### 9.2 Governing Equations

**Full 3D Heat Equation:**
```
∂T/∂t = α·∇²T = α·(∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)
```

where α = k/(ρ·cp) is thermal diffusivity.

**Implemented as Operator Splitting:**

The solver uses dimensional splitting to decouple lateral and vertical heat flow:

1. **Lateral (horizontal) diffusion** (explicit forward Euler):
```
T_surface^* = T_surface^n + dt·α_lateral·(∂²T/∂x² + ∂²T/∂y²)
```

2. **Vertical (subsurface) conduction** (implicit Crank-Nicolson):
```
Solve vertical heat equation from T_surface^* → T^{n+1}
```

**Critical Implementation Detail:**

Lateral diffusion is applied **AFTER** the vertical solve (not before) to preserve energy conservation. The algorithm sequence is:

```
For each time step:
  1. Compute surface energy balance → Q_net (from T^n)
  2. Solve vertical subsurface conduction with flux BC (T^n → T^{n+1})
  3. [NEW] Apply lateral diffusion to NEW surface temperature (T^{n+1} → T_final^{n+1})
```

This ordering ensures that:
- Energy balance Q_net is computed from same T used in vertical solve
- Lateral redistribution happens after energy input
- Energy conservation is maintained (vertical input + lateral redistribution)

### 9.3 Spatial Discretization

**5-Point Stencil for Surface Laplacian:**

The lateral diffusion uses a standard centered finite difference approximation:

```
∇²T[j,i] ≈ (T[j,i+1] - 2·T[j,i] + T[j,i-1])/dx² +
           (T[j+1,i] - 2·T[j,i] + T[j-1,i])/dy²
```

**Boundary Conditions (Lateral Domain Edges):**

Zero-flux (Neumann) boundary conditions are enforced:
```
∂T/∂n|_boundary = 0
```

**Implementation:** Ghost cells created via `np.pad(T, pad_width=1, mode='edge')`:
- Ghost cell values = edge cell values
- Ensures ∂T/∂n = 0 at boundaries
- Physically represents insulated lateral boundaries

### 9.4 Temporal Discretization

**Explicit Forward Euler for Lateral Step:**

```python
T_new = T_old + dt * α_lateral * ∇²T
```

**Vectorized NumPy Implementation:**

```python
# Pad with ghost cells for zero-flux BC
T_pad = np.pad(T_surface, pad_width=1, mode='edge')

# Compute Laplacian using array slicing
laplacian = (
    (T_pad[1:-1, 2:] - 2*T_pad[1:-1, 1:-1] + T_pad[1:-1, :-2]) / dx**2 +
    (T_pad[2:, 1:-1] - 2*T_pad[1:-1, 1:-1] + T_pad[:-2, 1:-1]) / dy**2
)

# Explicit update
T_new = T_surface + dt * alpha_lateral * laplacian
```

**Performance:** Vectorization provides 10-100× speedup over nested Python loops.

### 9.5 Stability Constraints

**Explicit Lateral Diffusion Requires:**

```
Fo_lateral = α_lateral·dt/dx² < 0.5  (stability)
Fo_lateral < 0.01                    (accuracy)
```

where Fo is the Fourier number.

**Typical Parameter Check:**

For sand (α ~ 1e-6 m²/s), dx = 1.0 m, dt = 120 s:
```
Fo = (1e-6 m²/s)·(120 s)/(1.0 m)² = 1.2×10⁻⁴ << 0.5 ✓
```

Even for high-conductivity rock (α ~ 1e-5 m²/s):
```
Fo = (1e-5)·(120)/(1.0)² = 1.2×10⁻³ << 0.5 ✓
```

**Warnings Issued:**
- **UNSTABLE warning**: max(Fo) ≥ 0.5
- **INACCURATE warning**: max(Fo) ≥ 0.01 (explicit scheme O(dt) temporal error)

### 9.6 Energy Conservation

**Energy Conservation with Lateral Conduction:**

Total energy change in domain:
```
dE/dt = ∫∫ Q_net·dA + (lateral flux in) - (lateral flux out)
```

For domain with **zero-flux lateral boundaries**:
- Lateral flux in = Lateral flux out = 0 at boundaries
- Interior lateral fluxes cancel (conservation property of diffusion)
- **Total energy change = ∫∫ Q_net·dA** (same as 1D-only case)

**Test Results:**
- Energy conservation test: < 1e-4 relative error with lateral conduction enabled
- Same tolerance as 1D solver (validates correct implementation)

### 9.7 Configuration and Usage

**Enabling Lateral Conduction:**

```python
from src.solver import ThermalSolver

solver = ThermalSolver(
    terrain=terrain,
    materials=materials,
    atmosphere=atmosphere,
    shadow_cache=shadow_cache,
    latitude=35.0,
    longitude=-106.0,
    altitude=1500.0,
    subsurface_grid=subsurface_grid,
    dt=120.0,
    enable_lateral_conduction=True,     # Enable lateral diffusion
    lateral_conductivity_factor=1.0     # Use same k as vertical (isotropic)
)
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_lateral_conduction` | bool | False | Enable/disable lateral diffusion |
| `lateral_conductivity_factor` | float | 1.0 | Multiplier for lateral thermal conductivity |

**Lateral Conductivity Factor:**
- **1.0**: Isotropic material (same conductivity all directions)
- **< 1.0**: Anisotropic material with reduced lateral conduction
  - Common in layered sediments
  - Useful for sensitivity studies

**Example (Anisotropic Material):**
```python
solver = ThermalSolver(
    ...,
    enable_lateral_conduction=True,
    lateral_conductivity_factor=0.5  # Lateral k = 0.5 × vertical k
)
```

### 9.8 Performance Impact

**With Lateral Conduction Disabled:**
- Zero performance impact (feature entirely skipped)

**With Lateral Conduction Enabled:**
- Lateral diffusion: O(ny·nx) vectorized operations per time step
- Adds ~5-10% overhead for typical grids (< 100×100)
- Still much faster than vertical solve (which dominates)

**Scaling:**
| Grid Size | Overhead |
|-----------|----------|
| 10×10 | < 1% |
| 100×100 | ~5-10% |
| 1000×1000 | ~10-15% |

### 9.9 Testing and Validation

**Test Suite:** [tests/test_lateral_conduction.py](../tests/test_lateral_conduction.py) (15 tests)

**Test Categories:**
1. **Backward Compatibility**: 1D behavior unchanged when disabled
2. **Physics Validation**: Heat spreading, gradient-driven flow, symmetry
3. **Energy Conservation**: < 1e-4 relative error (CRITICAL)
4. **Stability**: Warnings for Fo > 0.5, Fo > 0.01
5. **Boundary Conditions**: Zero-flux at domain edges
6. **Integration**: Full solver runs with lateral conduction enabled

**Key Physics Tests:**
- Uniform temperature → no change (∇²T = 0)
- Hot spot → heat spreads to neighbors
- Temperature gradient → flow from hot to cold
- Zero diffusivity → no lateral heat flow

**Energy Conservation Test (Most Critical):**
```python
def test_energy_conservation_with_lateral_conduction():
    """Verify energy conservation < 1e-4 with lateral conduction enabled"""
    # Create solver with lateral conduction
    # Apply energy balance and step forward
    # Compute: dE_actual vs. dE_expected from Q_net
    # Assert: |dE_actual - dE_expected| / E_initial < 1e-4
```

### 9.10 Alternative Approaches Considered

**1. Implicit 2D Solver (ADI - Alternating Direction Implicit):**
- **Pros**: Unconditionally stable, allows larger time steps
- **Cons**: Complex implementation, requires sparse matrix solver
- **Decision**: Rejected - explicit scheme sufficient for typical parameters

**2. Fully 3D Solver:**
- **Pros**: Most accurate, handles all directions uniformly
- **Cons**: Extremely complex, major rewrite, computationally expensive
- **Decision**: Rejected - 2D+1D adequate for surface-dominated physics

**3. Lateral Conduction in All Subsurface Layers:**
- **Pros**: Full 3D physics in subsurface
- **Cons**: Much more expensive, limited physical benefit (subsurface lateral conduction weak)
- **Decision**: Deferred to future work if needed

### 9.11 Future Enhancements

**Potential Extensions:**
1. **Dirichlet BC option**: Fixed temperature at boundaries (for validation studies)
2. **Implicit 2D solver**: ADI or sparse matrix for very large time steps
3. **Anisotropic materials**: Different k in x, y, z directions (full tensor)
4. **3D subsurface lateral**: Extend to all layers (full 3D heat equation)
5. **GPU acceleration**: Parallelize lateral diffusion loop (CUDA/OpenCL)

---

## 10. Numerical Stability and Accuracy

### 10.1 Stability Analysis

**Crank-Nicolson (Subsurface):**
- **Unconditionally stable** for all dt, dz
- Von Neumann analysis: amplification factor |G| ≤ 1 for all wavenumbers
- No CFL-type restriction

**Semi-Implicit Surface (IMEX):**
- Conduction: implicit (stable)
- Emission: linearized implicit (stable for small dt)
- Explicit terms: stable if dt < characteristic time scales

**Characteristic Time Scales:**
- Radiative cooling: `τ_rad ~ ρ·c_p·d / (4·ε·σ·T³) ~ 100-1000s`
- Convective: `τ_conv ~ ρ·c_p·d / h_conv ~ 10-100s`
- Conductive: `τ_cond ~ d² / α ~ 1-10s`

For `dt = 60-120s`:
- Conduction: handled implicitly (stable)
- Emission linearization: accurate (τ_rad >> dt)
- Convection: explicit stable (τ_conv ~ dt, acceptable)

**Conclusion:**
The scheme is **unconditionally stable** with good accuracy for dt = 60-120s.

### 10.2 Accuracy Considerations

**Temporal Accuracy:**
- Crank-Nicolson: O(dt²) (second-order)
- Emission linearization: O(dt) (first-order in nonlinear term)
- Overall: **O(dt)** due to IMEX splitting

**Spatial Accuracy:**
- Finite differences: O(dz²) (second-order)
- Stretched grid: maintains accuracy with variable dz

**Error Sources:**
1. Emission linearization: small for dt << τ_rad
2. Explicit radiation/convection: requires dt not too large
3. Subsurface bottom BC: requires z_max ≥ 3·δ

**Recommended Parameters:**
- `dt = 60-120s` (720-1440 steps/day)
- `z_max = 0.5m` (3× skin depth for sand)
- `n_layers = 20` (stretched grid, Fo ~ 1-10)

### 10.3 Energy Conservation (v2.0 - Improved)

The **von Neumann BC** ensures energy conservation **by construction**:

```
Energy change = ∫∫∫ ρ·c_p·(T^{n+1} - T^n) dV
               = dt · ∫∫ Q_net dA + O(dt²)
```

**Why it's better:**
- Flux BC directly enforces: Energy in = Energy conducted
- No artificial coupling that can create/destroy energy
- Conservation is automatic, not something we need to ensure

**Comparison with v1.0:**

*v1.0 (Dirichlet BC):*
- Required careful tuning of coupling coefficient h_cond
- Iteration needed to achieve consistency
- Small energy conservation errors possible

*v2.0 (von Neumann BC):*
- Energy conservation guaranteed by physics
- No tuning parameters
- Error only from discretization: O(dt²)

**Diagnostic Check:**

```python
def energy_conservation_check(temp_field_old, temp_field_new,
                             Q_net, materials, subsurface_grid, dt, dx, dy):
    """Verify energy conservation."""
    # Total energy change (all subsurface layers including surface)
    dT = temp_field_new.T_subsurface - temp_field_old.T_subsurface
    rho_cp = materials.rho * materials.cp  # Shape: (ny, nx, nz)

    # Volume-weighted energy change
    dE_total = 0.0
    for k in range(subsurface_grid.n_layers):
        dE_k = np.sum(rho_cp[:,:,k] * dT[:,:,k] * subsurface_grid.dz[k] * dx * dy)
        dE_total += dE_k

    # Expected from fluxes
    dE_expected = dt * np.sum(Q_net * dx * dy)

    # Relative error
    error = abs(dE_total - dE_expected) / abs(dE_expected)

    return error
```

Typical error: **< 1e-8** for v2.0 (improved from ~1e-6 in v1.0)

---

## 11. Implementation Notes

### 11.1 Material Property Handling

Material properties vary spatially:
- `materials.absorptivity[j, i]`
- `materials.emissivity[j, i]`
- `materials.conductivity[j, i]`
- `materials.density[j, i]`
- `materials.heat_capacity[j, i]`

The solver handles this correctly:
- Energy balance: point-by-point with local properties
- Subsurface solve: loop over (j, i), use local α

### 11.2 Shadow Cache Integration

Shadows change slowly day-to-day:
- Pre-compute shadow maps for entire day (31 times)
- Reuse for 3-7 consecutive days
- `ShadowCache` handles interpolation

This amortizes expensive ray tracing over multiple simulation days.

### 11.3 Initialization Strategies

**1. Constant Temperature:**
```python
solver.initialize(T_initial=300.0)
```
Simple, but may have initial transient.

**2. Equilibrium Profile:**
Compute steady-state subsurface profile from surface temp:
```python
T_subsurface[k] = T_surface - Q_net * z[k] / k
```

**3. Spin-Up:**
Run 1-3 days to reach quasi-steady diurnal cycle, then start actual simulation.

### 11.4 Performance Optimization (Future)

Current implementation is CPU-based with NumPy loops. Future GPU acceleration:

**GPU Kernels (CuPy):**
- Vectorize over all surface points (j, i)
- Parallel tridiagonal solves (cyclic reduction or parallel Thomas)
- Fused energy balance kernels

**Estimated Speedup:**
- 10-100× for large domains (nx, ny > 1000)
- Critical for km-scale simulations

---

## 12. References

### Numerical Methods

1. **Crank, J., and Nicolson, P.** (1947). "A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type." *Proceedings of the Cambridge Philosophical Society*, 43(1), 50-67.

2. **Thomas, L. H.** (1949). "Elliptic problems in linear difference equations over a network." *Watson Sci. Comput. Lab. Rept.*, Columbia University, New York.

3. **Press, W. H., et al.** (2007). *Numerical Recipes: The Art of Scientific Computing*, 3rd ed. Cambridge University Press.

4. **LeVeque, R. J.** (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.

### Heat Transfer

5. **Incropera, F. P., et al.** (2011). *Fundamentals of Heat and Mass Transfer*, 7th ed. John Wiley & Sons.

6. **Carslaw, H. S., and Jaeger, J. C.** (1959). *Conduction of Heat in Solids*, 2nd ed. Oxford University Press.

### Computational Geophysics

7. **Eppelbaum, L., et al.** (2014). *Applied Geothermics*. Springer.

8. **Kalnay, E.** (2003). *Atmospheric Modeling, Data Assimilation and Predictability*. Cambridge University Press.

---

**End of Document**
