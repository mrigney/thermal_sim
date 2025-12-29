"""
Example 03: Solar Radiation Demo

Demonstrates the solar radiation module capabilities:
- Solar position calculation over time
- Direct and diffuse irradiance models
- Shadow map computation
- Shadow caching for multi-day simulations
- Visualization of solar radiation on terrain

Usage:
    python 03_solar_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.terrain import create_synthetic_terrain
from src.solar import (solar_position, sun_vector, extraterrestrial_irradiance,
                       clear_sky_irradiance, irradiance_on_surface,
                       compute_shadow_map, ShadowCache, sunrise_sunset, day_of_year)
from src.visualization import TerrainVisualizer


def main():
    print("=" * 80)
    print("Thermal Terrain Simulator - Solar Radiation Demo")
    print("=" * 80)
    print()

    demo_start = time.time()

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # 1. SOLAR POSITION CALCULATIONS
    # =========================================================================
    print("1. Solar Position Calculations")
    print("-" * 80)
    t_start = time.time()

    # Location: Albuquerque, New Mexico (typical desert site)
    latitude = 35.0844  # degrees N
    longitude = -106.6504  # degrees W
    altitude = 1619.0  # meters above sea level

    # Date: Summer solstice
    date = datetime(2025, 6, 21, 12, 0, 0)  # Noon on summer solstice

    print(f"Location: {latitude}°N, {longitude}°W, {altitude}m elevation")
    print(f"Date/Time: {date}")
    print()

    # Calculate solar position
    azimuth, elevation = solar_position(latitude, longitude, date)
    print(f"Solar Position at Noon:")
    print(f"  Azimuth: {azimuth:.2f}° (clockwise from north)")
    print(f"  Elevation: {elevation:.2f}° (above horizon)")
    print()

    # Calculate sunrise/sunset
    sunrise, sunset = sunrise_sunset(latitude, longitude, date)
    print(f"Sunrise: {sunrise.strftime('%H:%M')}")
    print(f"Sunset: {sunset.strftime('%H:%M')}")
    daylight_hours = (sunset - sunrise).total_seconds() / 3600.0
    print(f"Daylight hours: {daylight_hours:.1f} hours")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 2. SOLAR POSITION THROUGHOUT DAY
    # =========================================================================
    print("2. Solar Path Throughout Day")
    print("-" * 80)
    t_start = time.time()

    times = []
    azimuths = []
    elevations = []

    current = date.replace(hour=0, minute=0)
    for hour in range(24):
        dt = current + timedelta(hours=hour)
        az, el = solar_position(latitude, longitude, dt)
        if el > 0:  # Only daylight hours
            times.append(hour)
            azimuths.append(az)
            elevations.append(el)

    print(f"Computed solar path for {len(times)} hours of daylight")

    # Plot solar path
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, elevations, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('Hour (UTC)')
    ax1.set_ylabel('Solar Elevation (degrees)')
    ax1.set_title('Solar Elevation Throughout Day')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)

    ax2.plot(times, azimuths, 'o-', linewidth=2, markersize=6, color='orange')
    ax2.set_xlabel('Hour (UTC)')
    ax2.set_ylabel('Solar Azimuth (degrees)')
    ax2.set_title('Solar Azimuth Throughout Day')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'solar_path.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: solar_path.png")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 3. IRRADIANCE CALCULATIONS
    # =========================================================================
    print("3. Clear Sky Irradiance Calculations")
    print("-" * 80)
    t_start = time.time()

    # Calculate for noon
    doy = day_of_year(date)
    I0 = extraterrestrial_irradiance(doy)
    print(f"Extraterrestrial irradiance: {I0:.1f} W/m²")

    # Clear sky conditions
    I_direct, I_diffuse = clear_sky_irradiance(elevation, doy, altitude,
                                               visibility=40.0, model='ineichen')

    print(f"\nClear sky irradiance at noon (Ineichen model):")
    print(f"  Direct beam (normal to sun): {I_direct:.1f} W/m²")
    print(f"  Diffuse (horizontal): {I_diffuse:.1f} W/m²")
    print(f"  Total on horizontal: {I_direct * np.sin(np.deg2rad(elevation)) + I_diffuse:.1f} W/m²")
    print()

    # Compare with simplified model
    I_direct_simple, I_diffuse_simple = clear_sky_irradiance(
        elevation, doy, altitude, visibility=40.0, model='simplified')

    print(f"Simplified model:")
    print(f"  Direct beam: {I_direct_simple:.1f} W/m²")
    print(f"  Diffuse: {I_diffuse_simple:.1f} W/m²")
    print()

    # Irradiance throughout day
    irrad_times = []
    irrad_direct = []
    irrad_diffuse = []
    irrad_total = []

    for hour in range(24):
        dt = current + timedelta(hours=hour)
        az, el = solar_position(latitude, longitude, dt)

        if el > 0:
            I_d, I_f = clear_sky_irradiance(el, doy, altitude, model='ineichen')
            I_total = I_d * np.sin(np.deg2rad(el)) + I_f

            irrad_times.append(hour)
            irrad_direct.append(I_d * np.sin(np.deg2rad(el)))
            irrad_diffuse.append(I_f)
            irrad_total.append(I_total)

    # Plot irradiance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(irrad_times, 0, irrad_direct, alpha=0.5, label='Direct', color='gold')
    ax.fill_between(irrad_times, irrad_direct,
                   [d+f for d,f in zip(irrad_direct, irrad_diffuse)],
                   alpha=0.5, label='Diffuse', color='lightblue')
    ax.plot(irrad_times, irrad_total, 'k-', linewidth=2, label='Total')

    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Irradiance on Horizontal Surface (W/m²)')
    ax.set_title('Solar Irradiance Throughout Day (Clear Sky)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'irradiance_daily.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: irradiance_daily.png")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 4. SHADOW COMPUTATION ON TERRAIN
    # =========================================================================
    print("4. Shadow Computation on Terrain")
    print("-" * 80)
    t_start = time.time()

    # Create synthetic terrain with interesting features
    nx, ny = 200, 200
    dx, dy = 1.0, 1.0  # 1 meter spacing, 200m x 200m domain
    print(f"Creating terrain: {nx}×{ny} grid, {dx}m spacing")

    terrain = create_synthetic_terrain(nx, ny, dx, dy, terrain_type='rolling_hills')
    terrain.compute_normals()
    terrain.compute_sky_view_factor_simple()

    print(f"Terrain elevation range: {terrain.elevation.min():.1f} to {terrain.elevation.max():.1f} m")
    print()

    # Compute shadows at different times of day
    shadow_times = [6, 9, 12, 15, 18]  # Hours
    print(f"Computing shadows at {len(shadow_times)} times: {shadow_times}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    vis = TerrainVisualizer()

    for idx, hour in enumerate(shadow_times):
        dt = date.replace(hour=hour)
        az, el = solar_position(latitude, longitude, dt)

        if el > 0:
            print(f"  Hour {hour:02d}: az={az:.1f}°, el={el:.1f}°", end='')

            # Compute shadow map
            shadow_map = compute_shadow_map(terrain.elevation, dx, dy, az, el,
                                           max_distance=500.0)

            shadowed_fraction = np.sum(shadow_map) / (nx * ny)
            print(f" -> {shadowed_fraction*100:.1f}% shadowed")

            # Visualize
            vis.plot_shadow_map(shadow_map, terrain, sun_az=az, sun_el=el, ax=axes[idx])
            axes[idx].set_title(f'{hour:02d}:00 - {shadowed_fraction*100:.0f}% shadowed')

    # Hide unused subplot
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shadows_daily.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: shadows_daily.png")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 5. SHADOW CACHING
    # =========================================================================
    print("5. Shadow Cache Pre-computation")
    print("-" * 80)
    t_start = time.time()

    cache = ShadowCache()

    # Compute shadows for entire day at 30-minute intervals
    print("Pre-computing shadow maps for one day...")
    cache.compute_daily_shadows(terrain.elevation, dx, dy,
                               latitude, longitude, date,
                               time_step_minutes=30,
                               max_distance=500.0)

    print()
    print(cache.get_info())
    print()

    # Test cache retrieval
    test_time = date.replace(hour=14, minute=37)  # Random time
    shadow_map, az, el = cache.get_shadow_map(test_time, interpolate=True)
    print(f"Retrieved shadow map for {test_time.strftime('%H:%M')}")
    print(f"  Nearest cached position: az={az:.1f}°, el={el:.1f}°")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 6. IRRADIANCE ON TERRAIN SURFACE
    # =========================================================================
    print("6. Total Irradiance on Terrain Surface")
    print("-" * 80)
    t_start = time.time()

    # Calculate irradiance at noon
    noon = date.replace(hour=12)
    az_noon, el_noon = solar_position(latitude, longitude, noon)
    I_d, I_f = clear_sky_irradiance(el_noon, doy, altitude, model='ineichen')
    s_vec = sun_vector(az_noon, el_noon)

    print(f"Computing surface irradiance at noon...")
    print(f"  Direct beam: {I_d:.1f} W/m²")
    print(f"  Diffuse: {I_f:.1f} W/m²")

    # Get shadow map for noon
    shadow_noon, _, _ = cache.get_shadow_map(noon, interpolate=True)

    # Calculate irradiance on each terrain point
    irradiance_map = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            normal = terrain.normals[j, i]
            svf = terrain.sky_view_factor[j, i]
            shadowed = shadow_noon[j, i]

            irradiance_map[j, i] = irradiance_on_surface(
                I_d, I_f, s_vec, normal, svf, shadowed
            )

    print(f"  Mean irradiance: {irradiance_map.mean():.1f} W/m²")
    print(f"  Range: {irradiance_map.min():.1f} to {irradiance_map.max():.1f} W/m²")
    print()

    # Visualize irradiance distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    extent = [0, nx*dx, 0, ny*dy]

    # Irradiance map
    im1 = ax1.imshow(irradiance_map, cmap='hot', extent=extent, origin='lower')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Total Surface Irradiance at Noon (W/m²)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Irradiance (W/m²)')

    # Histogram
    ax2.hist(irradiance_map.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Irradiance (W/m²)')
    ax2.set_ylabel('Number of Grid Points')
    ax2.set_title('Irradiance Distribution')
    ax2.axvline(irradiance_map.mean(), color='r', linestyle='--',
               linewidth=2, label=f'Mean: {irradiance_map.mean():.0f} W/m²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'surface_irradiance.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: surface_irradiance.png")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 7. SEASONAL VARIATION
    # =========================================================================
    print("7. Seasonal Solar Variation")
    print("-" * 80)
    t_start = time.time()

    # Four seasons
    dates = [
        datetime(2025, 3, 20, 12, 0),  # Spring equinox
        datetime(2025, 6, 21, 12, 0),  # Summer solstice
        datetime(2025, 9, 23, 12, 0),  # Fall equinox
        datetime(2025, 12, 21, 12, 0), # Winter solstice
    ]
    season_names = ['Spring Equinox', 'Summer Solstice', 'Fall Equinox', 'Winter Solstice']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (dt, name) in enumerate(zip(dates, season_names)):
        az_season, el_season = solar_position(latitude, longitude, dt)
        doy_season = day_of_year(dt)

        # Daily elevation curve
        hours = []
        elevations_season = []

        for h in range(24):
            dt_h = dt.replace(hour=h)
            _, el = solar_position(latitude, longitude, dt_h)
            if el > -5:  # Include twilight
                hours.append(h)
                elevations_season.append(el)

        axes[idx].plot(hours, elevations_season, 'o-', linewidth=2)
        axes[idx].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[idx].set_xlabel('Hour (UTC)')
        axes[idx].set_ylabel('Solar Elevation (degrees)')
        axes[idx].set_title(f'{name}\nMax elevation: {max(elevations_season):.1f}°')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-10, 90)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_variation.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: seasonal_variation.png")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    demo_time = time.time() - demo_start

    print("=" * 80)
    print("Solar Radiation Demo Complete!")
    print("=" * 80)
    print(f"Total execution time: {demo_time:.2f}s")
    print()
    print("Generated output files:")
    print("  - solar_path.png              (azimuth and elevation throughout day)")
    print("  - irradiance_daily.png        (direct/diffuse irradiance variation)")
    print("  - shadows_daily.png           (shadow maps at different times)")
    print("  - surface_irradiance.png      (irradiance distribution on terrain)")
    print("  - seasonal_variation.png      (solar elevation across seasons)")
    print()
    print("Solar module capabilities demonstrated:")
    print("  [+] Solar position calculations (azimuth, elevation)")
    print("  [+] Sunrise/sunset times")
    print("  [+] Extraterrestrial and clear-sky irradiance models")
    print("  [+] Shadow computation with ray marching")
    print("  [+] Shadow caching for performance")
    print("  [+] Surface irradiance on inclined terrain")
    print("  [+] Seasonal solar variation analysis")
    print()


if __name__ == '__main__':
    main()
