"""
Thermal Solver Demonstration

This example demonstrates the complete thermal solver integration:
1. Setting up terrain, materials, atmosphere, and solar forcing
2. Initializing the thermal solver
3. Running a 24-hour diurnal simulation
4. Visualizing temperature evolution and energy balance components
5. Analyzing subsurface thermal behavior

Location: Albuquerque, NM (desert terrain)
Date: Summer solstice (maximum solar forcing)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

from src.terrain import create_synthetic_terrain
from src.materials import MaterialDatabase, MaterialField
from src.solar import ShadowCache, solar_position, clear_sky_irradiance
from src.atmosphere import AtmosphericConditions, create_diurnal_temperature, create_diurnal_wind
from src.solver import ThermalSolver, SubsurfaceGrid, compute_energy_balance

# Setup output directory
output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("THERMAL SOLVER DEMONSTRATION")
print("="*70)

# ============================================================================
# SECTION 1: Setup Terrain and Materials
# ============================================================================
print("\n[1] Setting up terrain and materials...")
t_start = time.time()

# Create synthetic terrain (50m x 50m at 0.5m resolution)
# Small domain for fast demonstration
nx, ny = 100, 100
dx = dy = 0.5  # meters
terrain = create_synthetic_terrain(
    nx=nx, ny=ny, dx=dx, dy=dy,
    terrain_type='rolling_hills'
)

# Compute geometric properties
terrain.compute_normals()
terrain.compute_sky_view_factor_simple()

# Load material database
material_db = MaterialDatabase()
material_db.load_from_json('data/materials/representative_materials.json')

# Create subsurface grid (needed for MaterialField)
subsurface_grid = SubsurfaceGrid(
    z_max=0.5,         # 0.5m depth
    n_layers=20,       # 20 subsurface layers
    stretch_factor=1.2 # Geometric stretching
)

# Assign materials based on slope
# Flat areas: sand (class_id=1), steep areas: granite (class_id=2)
slope_angle = np.degrees(np.arctan(terrain.slope))
material_class = np.where(slope_angle < 15, 1, 2)  # 1=sand, 2=granite

materials = MaterialField(ny, nx, subsurface_grid.n_layers, material_db)
materials.assign_from_classification(material_class)

print(f"Terrain: {nx}x{ny} grid, {dx}m spacing, {nx*dx:.1f}m x {ny*dy:.1f}m domain")
print(f"Materials: {np.sum(material_class==1)} sand points, {np.sum(material_class==2)} granite points")
print(f"[Timing: {time.time() - t_start:.3f}s]")

# ============================================================================
# SECTION 2: Setup Atmospheric Conditions
# ============================================================================
print("\n[2] Setting up atmospheric conditions...")
t_start = time.time()

# Location: Albuquerque, NM
latitude = 35.0844
longitude = -106.6504
altitude_m = 1500.0  # meters above sea level

# Summer solstice (maximum solar forcing)
date = datetime(2025, 6, 21, 0, 0, 0)

# Diurnal temperature variation (desert conditions)
# Mean: 30C, amplitude: 15C, min at sunrise (6 AM local = 13:00 UTC)
T_air_func = create_diurnal_temperature(
    T_mean=273.15 + 30.0,
    T_amplitude=15.0,
    sunrise_hour=13.0  # UTC time (6 AM local = UTC-7)
)

# Diurnal wind variation (light to moderate)
# Mean: 3 m/s, amplitude: 2 m/s
wind_func = create_diurnal_wind(
    wind_mean=3.0,
    wind_amplitude=2.0
)

# Create atmospheric conditions object
atmosphere = AtmosphericConditions(
    T_air=T_air_func,
    wind_speed=wind_func,
    relative_humidity=0.2,  # Dry desert air
    reference_height=2.0,   # Standard weather station height
    cloud_fraction=0.0      # Clear sky
)

# Test at noon local time (19:00 UTC)
test_time = date + timedelta(hours=19)
state = atmosphere.get_atmospheric_state(test_time)
print(f"Atmospheric test (noon local, 19:00 UTC):")
print(f"  T_air = {state['T_air']-273.15:.1f}C")
print(f"  T_sky = {state['T_sky_brunt']-273.15:.1f}C (Brunt model)")
print(f"  Wind = {state['wind_speed']:.1f} m/s")
print(f"  h_conv = {state['h_conv']:.1f} W/(m2 K)")
print(f"[Timing: {time.time() - t_start:.3f}s]")

# ============================================================================
# SECTION 3: Precompute Shadow Cache (or load from file)
# ============================================================================
print("\n[3] Setting up shadow cache...")
t_start = time.time()

# Try to load existing cache, otherwise compute new one
shadow_cache_file = os.path.join(output_dir, 'shadow_cache_100x100.npz')

if os.path.exists(shadow_cache_file):
    print(f"Loading shadow cache from {shadow_cache_file}...")
    shadow_cache = ShadowCache.load(shadow_cache_file)
else:
    print("Computing shadow cache for simulation day...")
    shadow_cache = ShadowCache()
    shadow_cache.compute_daily_shadows(
        terrain.elevation, dx, dy,
        latitude, longitude, date,
        time_step_minutes=30
    )
    # Save for future runs
    shadow_cache.save(shadow_cache_file)

print(f"Shadow cache: {len(shadow_cache.times)} shadow maps")
print(f"[Timing: {time.time() - t_start:.3f}s]")

# ============================================================================
# SECTION 4: Initialize Thermal Solver
# ============================================================================
print("\n[4] Initializing thermal solver...")
t_start = time.time()

print(f"Subsurface grid: {subsurface_grid.n_layers} layers to {subsurface_grid.z_max}m depth")
print(f"  Surface layer: dz = {subsurface_grid.dz[0]*1000:.2f} mm")
print(f"  Bottom layer:  dz = {subsurface_grid.dz[-1]*1000:.2f} mm")

# Default time step
dt_default = 120.0  # seconds

# Create thermal solver
solver = ThermalSolver(
    terrain=terrain,
    materials=materials,
    atmosphere=atmosphere,
    shadow_cache=shadow_cache,
    latitude=latitude,
    longitude=longitude,
    altitude=altitude_m,
    subsurface_grid=subsurface_grid,
    dt=dt_default
)

# Initialize with uniform temperature (25C)
T_initial = 273.15 + 25.0
solver.initialize(T_initial=T_initial)

print(f"Solver initialized: T_initial = {T_initial-273.15:.1f}C")
print(f"Time step: dt = {solver.dt}s ({86400/solver.dt:.0f} steps/day)")
print(f"[Timing: {time.time() - t_start:.3f}s]")

# ============================================================================
# SECTION 5: Run Thermal Simulation (24 hours)
# ============================================================================
print("\n[5] Running 24-hour thermal simulation...")
t_start = time.time()

# Simulation period: full day
start_time = date
end_time = date + timedelta(days=1)
output_interval = 3600.0  # Output every hour

# Storage for results
times = []
T_surface_history = []
T_subsurface_history = []

# Run simulation
step_count = 0
for sim_time, temp_field in solver.run(start_time, end_time, output_interval):
    times.append(sim_time)
    T_surface_history.append(temp_field.T_surface.copy())
    T_subsurface_history.append(temp_field.T_subsurface.copy())
    step_count += 1

    # Print progress
    hour = (sim_time - start_time).total_seconds() / 3600
    T_mean = np.mean(temp_field.T_surface) - 273.15
    T_min = np.min(temp_field.T_surface) - 273.15
    T_max = np.max(temp_field.T_surface) - 273.15
    print(f"  Hour {hour:5.1f}: T_surface = {T_min:6.2f} to {T_max:6.2f}C (mean {T_mean:6.2f}C)")

print(f"Simulation complete: {step_count} outputs, {int((end_time-start_time).total_seconds()/solver.dt)} total steps")
print(f"[Timing: {time.time() - t_start:.3f}s]")

# ============================================================================
# SECTION 6: Analyze and Visualize Results
# ============================================================================
print("\n[6] Generating visualizations...")
t_start = time.time()

# Convert times to hours since start
hours = np.array([(t - start_time).total_seconds() / 3600 for t in times])

# Extract temperature statistics
T_mean_history = np.array([np.mean(T) for T in T_surface_history]) - 273.15
T_min_history = np.array([np.min(T) for T in T_surface_history]) - 273.15
T_max_history = np.array([np.max(T) for T in T_surface_history]) - 273.15

# ============================================================================
# Figure 1: Temperature Evolution
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Surface temperature time series
ax = axes[0, 0]
ax.fill_between(hours, T_min_history, T_max_history, alpha=0.3, label='Min-Max Range')
ax.plot(hours, T_mean_history, 'k-', linewidth=2, label='Mean')
ax.axhline(T_initial - 273.15, color='gray', linestyle='--', linewidth=1, label='Initial')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Surface Temperature [C]')
ax.set_title('(a) Surface Temperature Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# (b) Surface temperature map at peak (hour 19-20, noon local)
ax = axes[0, 1]
peak_idx = np.argmax(T_mean_history)
T_peak = T_surface_history[peak_idx] - 273.15
im = ax.imshow(T_peak, origin='lower', extent=[0, nx*dx, 0, ny*dy], cmap='hot')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(f'(b) Surface Temperature at Peak (Hour {hours[peak_idx]:.1f} UTC)')
plt.colorbar(im, ax=ax, label='Temperature [C]')

# (c) Surface temperature map at minimum (early morning)
ax = axes[1, 0]
min_idx = np.argmin(T_mean_history)
T_min_map = T_surface_history[min_idx] - 273.15
im = ax.imshow(T_min_map, origin='lower', extent=[0, nx*dx, 0, ny*dy], cmap='cool')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(f'(c) Surface Temperature at Minimum (Hour {hours[min_idx]:.1f} UTC)')
plt.colorbar(im, ax=ax, label='Temperature [C]')

# (d) Temperature range (max - min over 24h)
ax = axes[1, 1]
T_range = np.max(np.array(T_surface_history), axis=0) - np.min(np.array(T_surface_history), axis=0)
im = ax.imshow(T_range, origin='lower', extent=[0, nx*dx, 0, ny*dy], cmap='YlOrRd')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('(d) Diurnal Temperature Range')
plt.colorbar(im, ax=ax, label='Range [K]')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'solver_temperature_evolution.png'), dpi=150, bbox_inches='tight')
print("[+] Saved: outputs/solver_temperature_evolution.png")

# ============================================================================
# Figure 2: Subsurface Temperature Profiles
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Select a point in the middle of the domain
j_mid, i_mid = ny // 2, nx // 2

# (a) Subsurface temperature vs depth at different times
ax = axes[0]
plot_hours = [1, 7, 13, 19]  # Early morning, sunrise, afternoon, sunset (UTC)
colors = ['blue', 'green', 'orange', 'red']

for ph, color in zip(plot_hours, colors):
    idx = np.argmin(np.abs(hours - ph))
    T_profile = T_subsurface_history[idx][j_mid, i_mid, :] - 273.15
    T_surf = T_surface_history[idx][j_mid, i_mid] - 273.15

    # Include surface point
    z_plot = np.concatenate([[0], subsurface_grid.z_nodes])
    T_plot = np.concatenate([[T_surf], T_profile])

    ax.plot(T_plot, z_plot * 100, marker='o', label=f'Hour {hours[idx]:.1f} UTC', color=color)

ax.set_xlabel('Temperature [C]')
ax.set_ylabel('Depth [cm]')
ax.set_title(f'(a) Subsurface Temperature Profile at ({i_mid*dx:.1f}m, {j_mid*dy:.1f}m)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_yaxis()  # Depth increases downward

# (b) Time-depth temperature evolution (center point)
ax = axes[1]
T_center_evolution = np.array([T_sub[j_mid, i_mid, :] for T_sub in T_subsurface_history]) - 273.15
im = ax.contourf(hours, subsurface_grid.z_nodes * 100, T_center_evolution.T,
                 levels=20, cmap='RdYlBu_r')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Depth [cm]')
ax.set_title(f'(b) Temperature Evolution vs Depth at ({i_mid*dx:.1f}m, {j_mid*dy:.1f}m)')
plt.colorbar(im, ax=ax, label='Temperature [C]')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'solver_subsurface_profiles.png'), dpi=150, bbox_inches='tight')
print("[+] Saved: outputs/solver_subsurface_profiles.png")

# ============================================================================
# Figure 3: Energy Balance Components
# ============================================================================
print("\n[7] Computing energy balance diagnostics...")

# Compute energy fluxes at selected times for visualization
from src.solar import sun_vector
diagnostic_times = []
Q_solar_history = []
Q_atm_history = []
Q_emission_history = []
Q_conv_history = []

plot_hours_diag = [1, 7, 13, 19]  # Early morning, sunrise, afternoon, sunset (UTC)

for ph in plot_hours_diag:
    idx = np.argmin(np.abs(hours - ph))
    sim_time = times[idx]
    T_surf = T_surface_history[idx]

    # Get atmospheric state
    atm_state = atmosphere.get_atmospheric_state(sim_time)
    T_air = atm_state['T_air']
    T_sky = atm_state['T_sky_brunt']
    h_conv = atm_state['h_conv']

    # Get solar conditions
    az, el = solar_position(latitude, longitude, sim_time)
    sun_vec = sun_vector(az, el)

    if el > 0:
        from src.solar import day_of_year
        doy = day_of_year(sim_time)
        S_direct, S_diffuse = clear_sky_irradiance(el, doy, altitude_m, model='ineichen')
    else:
        S_direct, S_diffuse = 0.0, 0.0

    shadow_map, _, _ = shadow_cache.get_shadow_map(sim_time, interpolate=True)
    if shadow_map is None:
        shadow_map = np.zeros_like(T_surf, dtype=bool)

    # Compute energy balance
    fluxes = compute_energy_balance(
        T_surf, terrain, materials,
        S_direct, S_diffuse, sun_vec, shadow_map,
        T_sky, T_air, h_conv
    )

    diagnostic_times.append(ph)
    Q_solar_history.append(np.mean(fluxes['Q_solar']))
    Q_atm_history.append(np.mean(fluxes['Q_atm']))
    Q_emission_history.append(np.mean(fluxes['Q_emission']))
    Q_conv_history.append(np.mean(fluxes['Q_conv']))

# Plot energy balance components
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(diagnostic_times))
width = 0.2

ax.bar(x - 1.5*width, Q_solar_history, width, label='Solar Absorption', color='orange')
ax.bar(x - 0.5*width, Q_atm_history, width, label='Atmospheric LW', color='blue')
ax.bar(x + 0.5*width, Q_emission_history, width, label='Thermal Emission', color='red')
ax.bar(x + 1.5*width, Q_conv_history, width, label='Convection', color='green')

ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Energy Flux [W/m2]')
ax.set_title('Mean Energy Balance Components at Selected Times')
ax.set_xticks(x)
ax.set_xticklabels([f'{h:.0f}:00' for h in diagnostic_times])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'solver_energy_balance.png'), dpi=150, bbox_inches='tight')
print("[+] Saved: outputs/solver_energy_balance.png")

print(f"[Timing: {time.time() - t_start:.3f}s]")

# ============================================================================
# SECTION 7: Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("SIMULATION SUMMARY")
print("="*70)

print(f"\nDomain:")
print(f"  Size: {nx*dx}m x {ny*dy}m ({nx}x{ny} points at {dx}m spacing)")
print(f"  Elevation range: {np.min(terrain.elevation):.2f} to {np.max(terrain.elevation):.2f} m")
print(f"  Materials: {np.sum(material_class==1)} sand, {np.sum(material_class==2)} granite points")

print(f"\nSubsurface grid:")
print(f"  Depth: {subsurface_grid.z_max}m, {subsurface_grid.n_layers} layers")
print(f"  Cell spacing: {subsurface_grid.dz[0]*1000:.2f} to {subsurface_grid.dz[-1]*1000:.2f} mm")

print(f"\nTemporal discretization:")
print(f"  Time step: {solver.dt}s")
print(f"  Duration: 24 hours ({int(86400/solver.dt)} steps)")
print(f"  Output interval: {output_interval/3600:.1f} hours ({len(times)} snapshots)")

print(f"\nTemperature statistics:")
print(f"  Initial: {T_initial-273.15:.1f}C (uniform)")
print(f"  Minimum: {np.min(T_min_history):.2f}C (hour {hours[np.argmin(T_min_history)]:.1f})")
print(f"  Maximum: {np.max(T_max_history):.2f}C (hour {hours[np.argmax(T_max_history)]:.1f})")
print(f"  Diurnal range: {np.max(T_max_history) - np.min(T_min_history):.2f} K")
print(f"  Spatial variability: {np.std(T_range):.2f} K (std of diurnal range)")

print(f"\nEnergy balance (mean values at selected times):")
print(f"  Max solar absorption: {np.max(Q_solar_history):.1f} W/m2")
print(f"  Atmospheric LW: {np.mean(Q_atm_history):.1f} W/m2")
print(f"  Max emission: {np.min(Q_emission_history):.1f} W/m2 (most negative)")
print(f"  Convection range: {np.min(Q_conv_history):.1f} to {np.max(Q_conv_history):.1f} W/m2")

print("\n" + "="*70)
print("DEMONSTRATION COMPLETE")
print("="*70)
print("\nGenerated outputs:")
print("  [+] outputs/solver_temperature_evolution.png")
print("  [+] outputs/solver_subsurface_profiles.png")
print("  [+] outputs/solver_energy_balance.png")
print("\nThe thermal solver successfully integrated:")
print("  - Terrain geometry and material properties")
print("  - Solar radiation with shadow computation")
print("  - Atmospheric longwave and convective exchange")
print("  - Subsurface 1D heat conduction (Crank-Nicolson)")
print("  - Semi-implicit surface energy balance (IMEX)")
print("\nResults show realistic diurnal temperature evolution with:")
print("  - Strong surface heating during day (solar absorption)")
print("  - Radiative cooling at night")
print("  - Subsurface thermal lag (phase delay with depth)")
print("  - Spatial variability from terrain and materials")
