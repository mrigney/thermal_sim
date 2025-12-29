"""
Thermal Solver Module

This module implements the coupled surface energy balance and subsurface heat
conduction solver for terrain thermal simulation.

Key Components:
- SubsurfaceGrid: Vertical discretization specification
- TemperatureField: Temperature state management
- Energy balance computation (all flux terms)
- Tridiagonal subsurface solver (Crank-Nicolson)
- ThermalSolver: Main time-stepping coordinator

Numerical Scheme:
- IMEX (Implicit-Explicit) time stepping
- Implicit: Conduction terms (unconditionally stable)
- Explicit: Radiation and convection (linearized for stability)
- Crank-Nicolson for subsurface heat equation

Author: Thermal Terrain Simulator Team
Date: December 2025
"""

import numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta
import warnings

from .terrain import TerrainGrid
from .materials import MaterialField
from .atmosphere import AtmosphericConditions
from .solar import (solar_position, sun_vector, clear_sky_irradiance,
                   irradiance_on_surface, day_of_year, ShadowCache)


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
SECONDS_PER_DAY = 86400.0


# =============================================================================
# SUBSURFACE GRID
# =============================================================================

class SubsurfaceGrid:
    """
    Defines vertical discretization for subsurface heat conduction

    Creates a stretched grid with finer spacing near the surface where
    temperature gradients are largest.

    Parameters
    ----------
    z_max : float, optional
        Maximum depth [m], default 0.5m (sufficient for diurnal cycle)
    n_layers : int, optional
        Number of subsurface layers, default 20
    stretch_factor : float, optional
        Geometric stretching factor >1 for non-uniform grid, default 1.2

    Attributes
    ----------
    z_nodes : ndarray (n_layers,)
        Depth coordinates [m], z=0 at surface
    dz : ndarray (n_layers,)
        Layer thicknesses [m]
    z_interfaces : ndarray (n_layers+1,)
        Interface depths between layers
    """

    def __init__(self, z_max: float = 0.5, n_layers: int = 20,
                 stretch_factor: float = 1.2):

        if z_max <= 0:
            raise ValueError("z_max must be positive")
        if n_layers < 2:
            raise ValueError("Need at least 2 layers")
        if stretch_factor < 1.0:
            raise ValueError("stretch_factor must be >= 1.0")

        self.z_max = z_max
        self.n_layers = n_layers
        self.stretch_factor = stretch_factor

        # Create grid
        self.z_interfaces, self.z_nodes, self.dz = self._create_grid()

    def _create_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stretched grid with geometric progression

        Returns
        -------
        z_interfaces : ndarray (n_layers+1,)
            Interface depths
        z_nodes : ndarray (n_layers,)
            Cell center depths
        dz : ndarray (n_layers,)
            Cell thicknesses
        """
        if self.stretch_factor == 1.0:
            # Uniform grid
            z_interfaces = np.linspace(0, self.z_max, self.n_layers + 1)
        else:
            # Stretched grid (geometric progression)
            r = self.stretch_factor
            n = self.n_layers

            # Solve for first layer thickness
            # Sum of geometric series: z_max = dz0 * (r^n - 1) / (r - 1)
            dz0 = self.z_max * (r - 1) / (r**n - 1)

            # Create interfaces
            z_interfaces = np.zeros(n + 1)
            for i in range(1, n + 1):
                z_interfaces[i] = dz0 * (r**i - 1) / (r - 1)

        # Cell centers (midpoints)
        z_nodes = 0.5 * (z_interfaces[:-1] + z_interfaces[1:])

        # Cell thicknesses
        dz = np.diff(z_interfaces)

        return z_interfaces, z_nodes, dz

    def check_fourier_number(self, alpha: float, dt: float) -> float:
        """
        Compute Fourier number and check if in recommended range

        Parameters
        ----------
        alpha : float
            Thermal diffusivity [m²/s]
        dt : float
            Time step [s]

        Returns
        -------
        Fo : float
            Fourier number at finest grid spacing

        Notes
        -----
        For implicit schemes (Crank-Nicolson), stability is guaranteed.
        This check is for accuracy: recommended Fo ~ 1-10.
        """
        dz_min = self.dz.min()
        Fo = alpha * dt / dz_min**2

        if Fo < 0.1:
            warnings.warn(
                f"Low Fourier number (Fo={Fo:.3f}). "
                f"Consider larger dt for computational efficiency."
            )
        elif Fo > 100:
            warnings.warn(
                f"High Fourier number (Fo={Fo:.1f}). "
                f"Consider smaller dt for better time accuracy."
            )

        return Fo

    def get_info(self) -> str:
        """Return grid information as formatted string"""
        info = [
            "Subsurface Grid Information:",
            f"  Depth range: 0 to {self.z_max:.3f} m",
            f"  Number of layers: {self.n_layers}",
            f"  Stretch factor: {self.stretch_factor:.2f}",
            f"  Layer thickness range: {self.dz.min():.4f} to {self.dz.max():.4f} m",
            f"  Surface layer thickness: {self.dz[0]:.4f} m",
            f"  Bottom layer thickness: {self.dz[-1]:.4f} m",
        ]
        return "\n".join(info)


# =============================================================================
# TEMPERATURE FIELD
# =============================================================================

class TemperatureField:
    """
    Container for surface and subsurface temperature state

    Parameters
    ----------
    terrain : TerrainGrid
        Terrain geometry
    subsurface_grid : SubsurfaceGrid
        Vertical discretization

    Attributes
    ----------
    T_surface : ndarray (ny, nx)
        Surface temperatures [K]
    T_subsurface : ndarray (ny, nx, nz)
        Subsurface temperatures [K] at each depth
    time : datetime
        Current simulation time
    step_number : int
        Current time step number
    """

    def __init__(self, terrain: TerrainGrid, subsurface_grid: SubsurfaceGrid):
        self.ny, self.nx = terrain.elevation.shape
        self.nz = subsurface_grid.n_layers

        # Temperature arrays
        self.T_surface = np.zeros((self.ny, self.nx))
        self.T_subsurface = np.zeros((self.ny, self.nx, self.nz))

        # Metadata
        self.time = None
        self.step_number = 0

    def initialize(self, T_initial: float = 300.0):
        """
        Initialize temperature field to uniform value

        Parameters
        ----------
        T_initial : float, optional
            Initial temperature [K], default 300K (27°C)
        """
        self.T_surface[:] = T_initial
        self.T_subsurface[:] = T_initial
        self.step_number = 0

    def copy(self) -> 'TemperatureField':
        """Create deep copy of temperature field"""
        import copy
        return copy.deepcopy(self)


# =============================================================================
# ENERGY BALANCE CALCULATOR
# =============================================================================

def compute_energy_balance(
    T_surface: np.ndarray,
    terrain: TerrainGrid,
    materials: MaterialField,
    solar_irradiance_direct: float,
    solar_irradiance_diffuse: float,
    sun_vector_xyz: np.ndarray,
    shadow_map: np.ndarray,
    T_sky: float,
    T_air: float,
    h_conv: float
) -> Dict[str, np.ndarray]:
    """
    Compute all energy balance flux components

    Parameters
    ----------
    T_surface : ndarray (ny, nx)
        Surface temperature [K]
    terrain : TerrainGrid
        Terrain with normals and sky view factors
    materials : MaterialField
        Material properties at each point
    solar_irradiance_direct : float
        Direct beam irradiance [W/m²]
    solar_irradiance_diffuse : float
        Diffuse irradiance [W/m²]
    sun_vector_xyz : ndarray (3,)
        Unit vector toward sun
    shadow_map : ndarray (ny, nx) bool
        True where shadowed
    T_sky : float
        Effective sky temperature [K]
    T_air : float
        Air temperature [K]
    h_conv : float
        Convective heat transfer coefficient [W/(m²·K)]

    Returns
    -------
    fluxes : dict
        Dictionary containing:
        - 'Q_solar': Absorbed solar radiation [W/m²]
        - 'Q_atm': Atmospheric longwave [W/m²]
        - 'Q_emission': Surface emission [W/m²] (negative)
        - 'Q_conv': Convection [W/m²] (positive = heating)
        - 'Q_net': Net flux into surface [W/m²]
    """
    ny, nx = T_surface.shape

    # Solar radiation (absorbed)
    Q_solar = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            normal = terrain.normals[j, i]
            svf = terrain.sky_view_factor[j, i]
            shadowed = shadow_map[j, i]
            alpha = materials.alpha[j, i]

            # Total irradiance on surface
            I_total = irradiance_on_surface(
                solar_irradiance_direct,
                solar_irradiance_diffuse,
                sun_vector_xyz,
                normal,
                svf,
                shadowed
            )

            Q_solar[j, i] = alpha * I_total

    # Atmospheric longwave radiation
    epsilon = materials.epsilon
    svf = terrain.sky_view_factor
    Q_atm = epsilon * STEFAN_BOLTZMANN * svf * T_sky**4

    # Surface thermal emission (negative = cooling)
    Q_emission = -epsilon * STEFAN_BOLTZMANN * T_surface**4

    # Convection (positive when air warmer than surface)
    Q_conv = h_conv * (T_air - T_surface)

    # Net flux
    Q_net = Q_solar + Q_atm + Q_emission + Q_conv

    return {
        'Q_solar': Q_solar,
        'Q_atm': Q_atm,
        'Q_emission': Q_emission,
        'Q_conv': Q_conv,
        'Q_net': Q_net
    }


# =============================================================================
# SUBSURFACE SOLVER (TRIDIAGONAL)
# =============================================================================

def solve_subsurface_tridiagonal(
    T_prev: np.ndarray,
    Q_surface_net: np.ndarray,
    thermal_diffusivity: np.ndarray,
    rho_cp: np.ndarray,
    subsurface_grid: SubsurfaceGrid,
    dt: float
) -> np.ndarray:
    """
    Solve 1D subsurface heat equation using Crank-Nicolson method with flux BC

    Solves: ρc_p ∂T/∂t = ∂/∂z(k ∂T/∂z)

    Using Crank-Nicolson (implicit, 2nd order accurate, unconditionally stable):
    (T^(n+1) - T^n) / dt = 0.5 * [L(T^(n+1)) + L(T^n)]

    where L is the spatial operator: L(T) = α ∂²T/∂z²

    Upper boundary condition (von Neumann - flux continuity):
    -k * ∂T/∂z |_surface = Q_net (net flux into surface)

    Parameters
    ----------
    T_prev : ndarray (ny, nx, nz)
        Subsurface temperatures at previous time step [K]
        Note: T_prev[:,:,0] is the surface temperature
    Q_surface_net : ndarray (ny, nx)
        Net energy flux into surface [W/m²] (boundary condition)
    thermal_diffusivity : ndarray (ny, nx)
        α = k/(ρ·cp) [m²/s]
    rho_cp : ndarray (ny, nx)
        Volumetric heat capacity ρ·cp [J/(m³·K)]
    subsurface_grid : SubsurfaceGrid
        Vertical grid specification
    dt : float
        Time step [s]

    Returns
    -------
    T_new : ndarray (ny, nx, nz)
        Subsurface temperatures at new time step [K]
        T_new[:,:,0] is the updated surface temperature
    """
    ny, nx, nz = T_prev.shape
    T_new = np.zeros_like(T_prev)

    # Grid spacing
    dz = subsurface_grid.dz

    # Solve at each horizontal location
    for j in range(ny):
        for i in range(nx):
            alpha = thermal_diffusivity[j, i]
            rho_cp_val = rho_cp[j, i]
            T_old = T_prev[j, i, :]
            Q_net = Q_surface_net[j, i]

            # Build tridiagonal system: A·T_new = d
            # Using Crank-Nicolson with von Neumann BC at surface (layer 0)
            #
            # Upper BC (at surface, k=0): -k * dT/dz = Q_net
            # Energy equation at surface:
            #   ρ·cp·dz[0] * dT/dt = Q_net + k * (T[1] - T[0]) / dz[0]
            #
            # Lower BC (at depth, k=nz-1): zero flux (insulated)

            theta = 0.5  # Crank-Nicolson

            # Coefficients
            a = np.zeros(nz)  # sub-diagonal
            b = np.zeros(nz)  # diagonal
            c = np.zeros(nz)  # super-diagonal
            d = np.zeros(nz)  # right-hand side

            # First layer (k=0): Surface with flux BC
            # Energy balance: ρ·cp·dz[0] * dT/dt = Q_net + k * (T[1] - T[0]) / dz[0]
            #
            # Crank-Nicolson discretization:
            # ρ·cp·dz[0] * (T[0]^{n+1} - T[0]^n) / dt = Q_net +
            #     θ * k * (T[1]^{n+1} - T[0]^{n+1}) / dz[0] +
            #     (1-θ) * k * (T[1]^n - T[0]^n) / dz[0]

            if nz > 1:
                # Interface distance between nodes 0 and 1
                dz_interface = dz[0]
                r_01 = alpha / dz_interface**2

                # Collect terms for T^{n+1}:
                # ρ·cp·dz[0]/dt * T[0]^{n+1} - θ*k/dz[0] * T[0]^{n+1} + θ*k/dz[0] * T[1]^{n+1} = ...
                # Divide by (ρ·cp·dz[0]) to get dimensionless form:
                # (1/dt) * T[0]^{n+1} - θ*α/dz[0]^2 * T[0]^{n+1} + θ*α/dz[0]^2 * T[1]^{n+1} = RHS

                # Rearranging to standard form: b[0]*T[0] + c[0]*T[1] = d[0]
                a[0] = 0.0  # No layer above
                c[0] = -theta * dt * r_01 / dz[0]
                b[0] = 1.0 - c[0]

                # RHS: explicit conduction + old temperature + flux source
                explicit_flux = (1 - theta) * dt * r_01 / dz[0] * (T_old[1] - T_old[0])
                flux_source = dt * Q_net / (rho_cp_val * dz[0])

                d[0] = T_old[0] + explicit_flux + flux_source
            else:
                # Single layer case - direct flux application
                b[0] = 1.0
                flux_source = dt * Q_net / (rho_cp_val * dz[0])
                d[0] = T_old[0] + flux_source

            # Interior layers (k=1 to nz-2)
            for k in range(1, nz - 1):
                # Interface distances
                dz_km1 = dz[k-1]
                dz_kp1 = dz[k]
                r_minus = alpha / dz_km1**2
                r_plus = alpha / dz_kp1**2

                a[k] = -theta * dt * r_minus / dz[k]
                c[k] = -theta * dt * r_plus / dz[k]
                b[k] = 1.0 - a[k] - c[k]

                # RHS from old time level
                d[k] = T_old[k] + (1 - theta) * dt * (
                    r_plus / dz[k] * (T_old[k+1] - T_old[k]) -
                    r_minus / dz[k] * (T_old[k] - T_old[k-1])
                )

            # Last layer (k=nz-1): Neumann BC at bottom (zero flux)
            if nz > 1:
                k = nz - 1
                dz_km1 = dz[k-1]
                r_minus = alpha / dz_km1**2

                a[k] = -theta * dt * r_minus / dz[k]
                c[k] = 0.0  # No flux at bottom
                b[k] = 1.0 - a[k]

                d[k] = T_old[k] + (1 - theta) * dt * (
                    -r_minus / dz[k] * (T_old[k] - T_old[k-1])
                )

            # Solve tridiagonal system using Thomas algorithm
            T_new[j, i, :] = solve_thomas(a, b, c, d)

    return T_new


def solve_thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                 d: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm for tridiagonal matrix system

    Solves: A·x = d where A is tridiagonal with diagonals a, b, c

    Parameters
    ----------
    a : ndarray (n,)
        Sub-diagonal (a[0] is ignored)
    b : ndarray (n,)
        Main diagonal
    c : ndarray (n,)
        Super-diagonal (c[n-1] is ignored)
    d : ndarray (n,)
        Right-hand side

    Returns
    -------
    x : ndarray (n,)
        Solution
    """
    n = len(b)
    x = np.zeros(n)

    # Forward elimination
    c_star = np.zeros(n)
    d_star = np.zeros(n)

    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_star[i-1]
        if i < n - 1:
            c_star[i] = c[i] / denom
        d_star[i] = (d[i] - a[i] * d_star[i-1]) / denom

    # Back substitution
    x[n-1] = d_star[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i+1]

    return x


def apply_lateral_diffusion(
    T_surface: np.ndarray,
    thermal_diffusivity: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    conductivity_factor: float = 1.0
) -> np.ndarray:
    """
    Apply lateral heat diffusion at surface using explicit forward Euler.

    Uses 5-point stencil for Laplacian:
    ∇²T[j,i] ≈ (T[j,i+1] - 2*T[j,i] + T[j,i-1])/dx² +
               (T[j+1,i] - 2*T[j,i] + T[j-1,i])/dy²

    Boundary conditions: Zero-flux (Neumann) at domain edges

    Parameters
    ----------
    T_surface : np.ndarray (ny, nx)
        Current surface temperature field [K]
    thermal_diffusivity : np.ndarray (ny, nx)
        Thermal diffusivity α = k/(ρ·cp) [m²/s]
    dx, dy : float
        Grid spacing [m]
    dt : float
        Time step [s]
    conductivity_factor : float, optional
        Multiplier for lateral thermal diffusivity (default 1.0)

    Returns
    -------
    T_new : np.ndarray (ny, nx)
        Updated surface temperature after lateral diffusion [K]

    Notes
    -----
    Stability requires: α·dt/dx² < 0.5 (explicit forward Euler)
    For typical parameters (α~1e-6, dx~1m, dt~120s), Fo~1e-4 << 0.5 (stable)
    Accuracy: Recommend Fo < 0.01 for minimal temporal discretization error
    """
    ny, nx = T_surface.shape

    # Apply conductivity factor (for anisotropic materials or sensitivity studies)
    alpha_lateral = thermal_diffusivity * conductivity_factor

    # Compute Fourier number for stability and accuracy checks
    Fo_x = alpha_lateral * dt / dx**2
    Fo_y = alpha_lateral * dt / dy**2
    Fo_total = Fo_x + Fo_y
    max_Fo = np.max(Fo_total)

    # Stability warning
    if max_Fo >= 0.5:
        import warnings
        warnings.warn(
            f"Lateral diffusion may be UNSTABLE: max(Fo) = {max_Fo:.3f} >= 0.5. "
            f"Consider reducing dt or increasing dx/dy.",
            UserWarning
        )
    # Accuracy warning
    elif max_Fo >= 0.01:
        import warnings
        warnings.warn(
            f"Lateral diffusion may be INACCURATE: max(Fo) = {max_Fo:.3f} >= 0.01. "
            f"Temporal splitting error is O(dt). Consider smaller dt for better accuracy.",
            UserWarning
        )

    # Pad arrays with ghost cells for zero-flux boundary conditions
    # mode='edge' creates ghost cells with same value as edge: ∂T/∂n = 0
    T_pad = np.pad(T_surface, pad_width=1, mode='edge')

    # Compute Laplacian using vectorized array slicing (5-point stencil)
    # Interior points: T_pad[1:-1, 1:-1] corresponds to original T_surface
    laplacian = (
        (T_pad[1:-1, 2:] - 2*T_pad[1:-1, 1:-1] + T_pad[1:-1, :-2]) / dx**2 +
        (T_pad[2:, 1:-1] - 2*T_pad[1:-1, 1:-1] + T_pad[:-2, 1:-1]) / dy**2
    )

    # Explicit forward Euler update (vectorized)
    T_new = T_surface + dt * alpha_lateral * laplacian

    return T_new


# =============================================================================
# SURFACE UPDATE - REMOVED
# =============================================================================
#
# The update_surface_temperature() function has been removed.
# Surface temperature is now computed directly by the subsurface solver
# using a von Neumann (flux) boundary condition, which ensures proper
# energy conservation.


# =============================================================================
# MAIN THERMAL SOLVER
# =============================================================================

class ThermalSolver:
    """
    Main thermal solver coordinating all components

    Integrates surface energy balance and subsurface heat conduction
    over time using IMEX (Implicit-Explicit) time stepping.

    Parameters
    ----------
    terrain : TerrainGrid
        Terrain geometry
    materials : MaterialField
        Material properties
    atmosphere : AtmosphericConditions
        Atmospheric forcing
    shadow_cache : ShadowCache
        Pre-computed shadow maps
    latitude : float
        Site latitude [degrees N]
    longitude : float
        Site longitude [degrees W]
    altitude : float
        Site elevation [m above sea level]
    subsurface_grid : SubsurfaceGrid, optional
        Vertical grid (default: 0.5m depth, 20 layers)
    dt : float, optional
        Time step [s], default 120s (2 minutes)
    enable_lateral_conduction : bool, optional
        Enable lateral heat diffusion at surface (default: False)
    lateral_conductivity_factor : float, optional
        Multiplier for lateral thermal conductivity (default: 1.0)
        Values < 1.0 model anisotropic materials with reduced lateral conduction
    """

    def __init__(self,
                 terrain: TerrainGrid,
                 materials: MaterialField,
                 atmosphere: AtmosphericConditions,
                 shadow_cache: ShadowCache,
                 latitude: float,
                 longitude: float,
                 altitude: float = 0.0,
                 timezone_offset: float = 0.0,
                 subsurface_grid: Optional[SubsurfaceGrid] = None,
                 dt: float = 120.0,
                 enable_lateral_conduction: bool = False,
                 lateral_conductivity_factor: float = 1.0):

        self.terrain = terrain
        self.materials = materials
        self.atmosphere = atmosphere
        self.shadow_cache = shadow_cache
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone_offset = timezone_offset
        self.dt = dt
        self.enable_lateral_conduction = enable_lateral_conduction
        self.lateral_conductivity_factor = lateral_conductivity_factor

        # Create subsurface grid if not provided
        if subsurface_grid is None:
            self.subsurface_grid = SubsurfaceGrid()
        else:
            self.subsurface_grid = subsurface_grid

        # Temperature field
        self.temp_field = TemperatureField(terrain, self.subsurface_grid)

        # Latest energy fluxes (for diagnostics/visualization)
        self.latest_fluxes = None
        self.latest_shadow_map = None
        self.latest_solar_azimuth = 0.0
        self.latest_solar_elevation = 0.0

        # Check Fourier number
        alpha_typical = 5e-7  # Typical for sand
        Fo = self.subsurface_grid.check_fourier_number(alpha_typical, dt)
        print(f"Fourier number (typical material): {Fo:.2f}")

    def initialize(self, T_initial: float = 300.0):
        """
        Initialize temperature field

        Parameters
        ----------
        T_initial : float, optional
            Initial uniform temperature [K], default 300K
        """
        self.temp_field.initialize(T_initial)
        print(f"Initialized temperature field to {T_initial:.1f} K")

    def step(self, current_time: datetime) -> TemperatureField:
        """
        Advance solution by one time step

        Parameters
        ----------
        current_time : datetime
            Current simulation time

        Returns
        -------
        temp_field : TemperatureField
            Updated temperature field
        """
        # Get current state
        # Note: T_subsurface[:,:,0] is the surface temperature
        T_sub_old = self.temp_field.T_subsurface.copy()

        # Solar position and irradiance
        azimuth, elevation = solar_position(self.latitude, self.longitude, current_time, self.timezone_offset)

        if elevation > 0:
            doy = day_of_year(current_time)
            I_direct, I_diffuse = clear_sky_irradiance(
                elevation, doy, self.altitude, model='ineichen'
            )
            sun_vec = sun_vector(azimuth, elevation)
        else:
            I_direct = 0.0
            I_diffuse = 0.0
            sun_vec = np.array([0.0, 0.0, -1.0])

        # Shadow map (default to no shadows if not in cache)
        shadow_result = self.shadow_cache.get_shadow_map(current_time, interpolate=True)
        shadow_map = shadow_result[0]
        if shadow_map is None:
            # No cached shadow map - assume no shadows (sun might be below horizon)
            shadow_map = np.zeros((self.terrain.elevation.shape), dtype=bool)

        # Atmospheric state
        T_air = self.atmosphere.get_air_temperature(current_time)
        T_sky = self.atmosphere.get_sky_temperature(current_time, model='idso')
        h_conv = self.atmosphere.get_convection_coefficient(current_time)

        # Compute energy balance using OLD surface temperature
        T_surf_old = T_sub_old[:, :, 0]
        fluxes = compute_energy_balance(
            T_surf_old, self.terrain, self.materials,
            I_direct, I_diffuse, sun_vec, shadow_map,
            T_sky, T_air, h_conv
        )

        # Store latest fluxes, shadow map, and solar position for diagnostics
        self.latest_fluxes = fluxes
        self.latest_shadow_map = shadow_map
        self.latest_solar_azimuth = azimuth
        self.latest_solar_elevation = elevation

        # Get net flux into surface (this is our von Neumann BC)
        Q_net = fluxes['Q_net']

        # Get thermal properties (use surface layer values)
        k_thermal = self.materials.k[:, :, 0]
        rho = self.materials.rho[:, :, 0]
        cp = self.materials.cp[:, :, 0]
        rho_cp = rho * cp
        alpha = k_thermal / rho_cp

        # Solve subsurface with flux BC
        # This computes the new surface temperature (layer 0) and all subsurface temps
        T_sub_new = solve_subsurface_tridiagonal(
            T_sub_old, Q_net, alpha, rho_cp,
            self.subsurface_grid, self.dt
        )

        # Apply lateral conduction at surface (if enabled)
        # CRITICAL: Applied AFTER vertical solve to preserve energy conservation
        if self.enable_lateral_conduction:
            T_surf_after_lateral = apply_lateral_diffusion(
                T_surface=T_sub_new[:, :, 0],  # Use NEW surface temp after vertical solve
                thermal_diffusivity=alpha,
                dx=self.terrain.dx,
                dy=self.terrain.dy,
                dt=self.dt,
                conductivity_factor=self.lateral_conductivity_factor
            )
            # Update subsurface array with laterally-diffused surface
            T_sub_new[:, :, 0] = T_surf_after_lateral

        # Extract surface temperature (either with or without lateral diffusion)
        T_surf_new = T_sub_new[:, :, 0]

        # Update field
        self.temp_field.T_surface = T_surf_new
        self.temp_field.T_subsurface = T_sub_new
        self.temp_field.time = current_time
        self.temp_field.step_number += 1

        return self.temp_field

    def run(self, start_time: datetime, end_time: datetime,
            output_interval: float = 3600.0):
        """
        Run simulation from start to end time

        Parameters
        ----------
        start_time : datetime
            Start time
        end_time : datetime
            End time
        output_interval : float, optional
            Output interval [s], default 3600s (1 hour)

        Yields
        ------
        time : datetime
            Output time
        temp_field : TemperatureField
            Temperature field at output time
        """
        current_time = start_time
        next_output = start_time

        total_seconds = (end_time - start_time).total_seconds()
        n_steps = int(total_seconds / self.dt)

        print(f"Starting simulation:")
        print(f"  Start: {start_time}")
        print(f"  End: {end_time}")
        print(f"  Duration: {total_seconds/3600:.1f} hours")
        print(f"  Time step: {self.dt:.1f} s")
        print(f"  Total steps: {n_steps}")
        print()

        # Yield initial state
        self.temp_field.time = start_time
        yield start_time, self.temp_field.copy()

        step = 0
        while current_time < end_time:
            # Advance one step
            current_time += timedelta(seconds=self.dt)
            self.step(current_time)
            step += 1

            # Output at intervals
            if current_time >= next_output or current_time >= end_time:
                yield current_time, self.temp_field.copy()
                next_output += timedelta(seconds=output_interval)

                # Progress
                progress = 100.0 * (current_time - start_time).total_seconds() / total_seconds
                print(f"  Step {step}/{n_steps} ({progress:.1f}%): "
                      f"{current_time.strftime('%Y-%m-%d %H:%M')} - "
                      f"T_mean = {self.temp_field.T_surface.mean():.2f} K")
