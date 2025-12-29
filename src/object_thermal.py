"""
Thermal Solver for 3D Objects

Extends the 1D thermal solver to work on object faces. Each triangular face
receives an independent 1D thermal solution through the object's thickness.
"""

import numpy as np
from typing import List, Tuple
from src.solar import sun_vector, irradiance_on_surface


def compute_object_solar_flux(obj, sun_azimuth: float, sun_elevation: float,
                              I_direct: float, I_diffuse: float) -> np.ndarray:
    """
    Compute solar flux on each face of a thermal object.

    Parameters
    ----------
    obj : ThermalObject
        Thermal object with geometry and shadow data
    sun_azimuth : float
        Solar azimuth [degrees]
    sun_elevation : float
        Solar elevation [degrees]
    I_direct : float
        Direct beam irradiance [W/m²]
    I_diffuse : float
        Diffuse sky irradiance [W/m²]

    Returns
    -------
    solar_flux : ndarray, shape (n_faces,)
        Total solar flux per face [W/m²]
    """
    n_faces = obj.normals.shape[0]
    solar_flux = np.zeros(n_faces)

    # Sun vector
    sun_vec = sun_vector(sun_azimuth, sun_elevation)

    for face_idx in range(n_faces):
        normal = obj.normals[face_idx]
        shadow_frac = obj.shadow_fraction[face_idx]
        sky_view = obj.sky_view_factor[face_idx]

        # Compute total irradiance on this face
        # Shadow fraction: 0 = sunlit, 1 = shadowed
        shadowed = (shadow_frac > 0.5)

        flux = irradiance_on_surface(
            I_direct, I_diffuse,
            sun_vec, normal,
            sky_view_factor=sky_view,
            shadowed=shadowed
        )

        solar_flux[face_idx] = flux

    return solar_flux


def initialize_object_subsurface(obj, nz: int, z_max: float):
    """
    Initialize subsurface grid for object faces.

    Each face gets a 1D subsurface grid through the object thickness.

    Parameters
    ----------
    obj : ThermalObject
        Thermal object
    nz : int
        Number of subsurface layers
    z_max : float
        Maximum depth [m] - should be >= object thickness
    """
    n_faces = obj.normals.shape[0]

    # Use object thickness as max depth
    z_max = max(z_max, obj.thickness)

    # Create uniform grid through thickness
    obj.z_subsurface = np.linspace(0, z_max, nz)
    obj.dz = obj.z_subsurface[1] - obj.z_subsurface[0]

    # Initialize subsurface temperatures (same as surface initially)
    obj.T_subsurface = np.ones((n_faces, nz)) * obj.T_surface[:, np.newaxis]

    print(f"Initialized subsurface for {obj.name}:")
    print(f"  Faces: {n_faces}")
    print(f"  Layers: {nz}")
    print(f"  Thickness: {z_max:.3f} m")
    print(f"  Layer spacing: {obj.dz:.4f} m")


def solve_object_thermal_1d(obj, dt: float, T_air: float, T_sky: float,
                            wind_speed: float):
    """
    Solve 1D heat equation for all object faces.

    Uses implicit (backward Euler) solver for stability. Each face is
    treated independently with its own 1D thermal column.

    Parameters
    ----------
    obj : ThermalObject
        Thermal object with current state
    dt : float
        Time step [seconds]
    T_air : float
        Air temperature [K]
    T_sky : float
        Sky temperature for longwave exchange [K]
    wind_speed : float
        Wind speed [m/s]
    """
    if obj.T_subsurface is None:
        raise ValueError(f"Object {obj.name} subsurface not initialized")

    n_faces = obj.normals.shape[0]
    nz = obj.T_subsurface.shape[1]

    # Material properties
    k = obj.material.k  # Thermal conductivity [W/(m·K)]
    rho = obj.material.rho  # Density [kg/m³]
    cp = obj.material.cp  # Specific heat [J/(kg·K)]
    alpha_s = obj.material.alpha  # Solar absorptivity
    epsilon = obj.material.epsilon  # Thermal emissivity

    # Thermal diffusivity
    kappa = k / (rho * cp)

    # Stefan-Boltzmann constant
    sigma = 5.670374419e-8  # W/(m²·K⁴)

    # Stability parameter
    r = kappa * dt / obj.dz**2

    if r > 0.5:
        print(f"Warning: Large thermal diffusion number r={r:.3f} for {obj.name}")
        print(f"  Consider reducing time step or increasing nz")

    # Solve for each face independently
    for face_idx in range(n_faces):
        # Current temperature profile
        T = obj.T_subsurface[face_idx, :].copy()

        # Surface boundary condition (energy balance)
        T_surf = T[0]

        # Solar heating (absorbed)
        Q_solar = alpha_s * obj.solar_flux[face_idx]

        # Longwave out (emission)
        Q_lw_out = epsilon * sigma * T_surf**4

        # Longwave in (sky radiation)
        Q_lw_in = epsilon * sigma * T_sky**4

        # Convection (simple parameterization)
        # h = 5 + 2.8 * wind_speed  # Simplified convection coefficient
        h = 10.0 + 3.0 * wind_speed  # W/(m²·K)
        Q_conv = h * (T_air - T_surf)

        # Net surface flux [W/m²]
        Q_net = Q_solar - Q_lw_out + Q_lw_in + Q_conv

        # Conduction into subsurface
        Q_cond = -k * (T[1] - T[0]) / obj.dz

        # Surface energy balance to get new surface temperature
        # Thin surface layer approximation
        # dT_surf/dt = (Q_net + Q_cond) / (rho * cp * dz_surf)
        dz_surf = obj.dz / 2  # Half-layer at surface
        dT_surf = dt * (Q_net + Q_cond) / (rho * cp * dz_surf)
        T_new_surf = T_surf + dT_surf

        # Subsurface nodes: implicit solver
        # T_new[i] - T[i] = r * (T_new[i+1] - 2*T_new[i] + T_new[i-1])
        # Rearranged: -r*T_new[i-1] + (1+2r)*T_new[i] - r*T_new[i+1] = T[i]

        # Build tridiagonal system for interior nodes (1 to nz-2)
        # Bottom boundary: insulated (dT/dz = 0) or fixed temperature

        if nz > 2:
            # Tridiagonal matrix coefficients
            a = np.ones(nz - 2) * (-r)  # Lower diagonal
            b = np.ones(nz - 2) * (1 + 2*r)  # Main diagonal
            c = np.ones(nz - 2) * (-r)  # Upper diagonal
            d = T[1:-1].copy()  # RHS

            # Top boundary (connect to surface)
            d[0] += r * T_new_surf

            # Bottom boundary (insulated: T[nz-1] = T[nz-2])
            # This makes T_new[nz-1] = T_new[nz-2]
            # Last equation becomes: (1+r)*T_new[nz-2] = T[nz-2] + r*T_new[nz-3]
            b[-1] = 1 + r
            c[-1] = 0

            # Solve tridiagonal system
            T_new_interior = thomas_algorithm(a, b, c, d)

            # Update temperature profile
            T_new = np.zeros(nz)
            T_new[0] = T_new_surf
            T_new[1:-1] = T_new_interior
            T_new[-1] = T_new_interior[-1]  # Insulated bottom

        else:
            # Only 2 layers: surface and one subsurface
            T_new = np.array([T_new_surf, T[1]])

        # Store updated temperatures
        obj.T_subsurface[face_idx, :] = T_new
        obj.T_surface[face_idx] = T_new_surf


def thomas_algorithm(a, b, c, d):
    """
    Solve tridiagonal system using Thomas algorithm.

    Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]

    Parameters
    ----------
    a : ndarray
        Lower diagonal (length n-1, first element ignored)
    b : ndarray
        Main diagonal (length n)
    c : ndarray
        Upper diagonal (length n-1, last element ignored)
    d : ndarray
        Right-hand side (length n)

    Returns
    -------
    x : ndarray
        Solution (length n)
    """
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    # Forward sweep
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


def compute_object_sky_view_factors(obj):
    """
    Compute sky view factor for each face (simplified).

    For now, uses a simple heuristic based on face normal direction.
    A more sophisticated approach would use ray tracing.

    Parameters
    ----------
    obj : ThermalObject
        Thermal object

    Returns
    -------
    sky_view : ndarray, shape (n_faces,)
        Sky view factor per face (0 to 1)
    """
    n_faces = obj.normals.shape[0]
    sky_view = np.zeros(n_faces)

    for i in range(n_faces):
        normal = obj.normals[i]

        # Upward-facing surfaces see more sky
        # Horizontal surface (normal=[0,0,1]): sky_view = 0.5 (hemisphere)
        # Vertical surface: sky_view = 0.5
        # Downward-facing: sky_view = 0

        nz = normal[2]  # Z component of normal

        if nz > 0:
            # Upward-facing: sky view 0.5 to 1.0
            sky_view[i] = 0.5 + 0.5 * nz
        else:
            # Downward or sideways: reduced sky view
            sky_view[i] = 0.5 * max(0, 1 + nz)

    return sky_view
