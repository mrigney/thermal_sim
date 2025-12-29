"""
Enhanced visualization module for thermal terrain simulations.

Provides multi-panel diagnostic plots including:
- Terrain elevation
- Temperature fields
- Shadow maps
- Energy flux breakdowns
- Subsurface temperature profiles

Author: Thermal Simulator Team
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from src.terrain import TerrainGrid
from src.solver import TemperatureField, SubsurfaceGrid


def plot_terrain_elevation(terrain: TerrainGrid, output_path: Path, dpi: int = 150):
    """
    Plot terrain elevation map.

    Parameters
    ----------
    terrain : TerrainGrid
        Terrain object
    output_path : Path
        Output file path
    dpi : int
        Plot resolution
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot elevation
    im = ax.imshow(terrain.elevation, cmap='terrain', origin='lower',
                  extent=[0, terrain.nx * terrain.dx, 0, terrain.ny * terrain.dy])

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Terrain Elevation', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Elevation [m]', rotation=270, labelpad=20)

    # Add statistics
    elev_min, elev_max = terrain.elevation.min(), terrain.elevation.max()
    elev_mean = terrain.elevation.mean()
    elev_std = terrain.elevation.std()

    stats_text = f'Min: {elev_min:.2f} m\nMax: {elev_max:.2f} m\n'
    stats_text += f'Mean: {elev_mean:.2f} m\nStd: {elev_std:.2f} m'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_hourly_summary(
    terrain: TerrainGrid,
    temp_field: TemperatureField,
    fluxes: Dict[str, np.ndarray],
    shadow_map: np.ndarray,
    subsurface_grid: SubsurfaceGrid,
    current_time: datetime,
    output_path: Path,
    solar_azimuth: float = 0.0,
    solar_elevation: float = 0.0,
    dpi: int = 150
):
    """
    Create 4-panel hourly summary plot.

    Panels:
    1. Temperature field (contour)
    2. Shadow map
    3. Energy fluxes (spatial maps)
    4. Subsurface temperature profile

    Parameters
    ----------
    terrain : TerrainGrid
        Terrain object
    temp_field : TemperatureField
        Temperature field
    fluxes : dict
        Energy flux dictionary (Q_solar, Q_atm, Q_emission, Q_convection, Q_net)
    shadow_map : ndarray
        Shadow map (True = shadowed)
    subsurface_grid : SubsurfaceGrid
        Subsurface grid for depth coordinates
    current_time : datetime
        Current simulation time
    output_path : Path
        Output file path
    dpi : int
        Plot resolution
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Extent for imshow
    extent = [0, terrain.nx * terrain.dx, 0, terrain.ny * terrain.dy]

    # --- Panel 1: Surface Temperature ---
    ax1 = fig.add_subplot(gs[0, 0])

    T_surf = temp_field.T_surface
    T_min, T_max = T_surf.min(), T_surf.max()
    T_mean = T_surf.mean()

    im1 = ax1.contourf(T_surf - 273.15, levels=15, cmap='RdYlBu_r', origin='lower',
                      extent=extent)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title(f'Surface Temperature\nMean: {T_mean-273.15:.1f}°C, Range: [{T_min-273.15:.1f}, {T_max-273.15:.1f}]°C',
                 fontsize=11, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Temperature [°C]', rotation=270, labelpad=20)

    # --- Panel 2: Shadow Map ---
    ax2 = fig.add_subplot(gs[0, 1])

    # Show shadow map with terrain elevation as background
    ax2.imshow(terrain.elevation, cmap='gray', alpha=0.3, origin='lower', extent=extent)
    shadow_display = np.ma.masked_where(~shadow_map, shadow_map)
    im2 = ax2.imshow(shadow_display, cmap='Blues', alpha=0.7, origin='lower',
                    extent=extent, vmin=0, vmax=1)

    pct_shadowed = 100.0 * np.sum(shadow_map) / shadow_map.size
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title(f'Shadow Map\n{pct_shadowed:.1f}% shadowed | Sun: Az={solar_azimuth:.1f}° El={solar_elevation:.1f}°',
                 fontsize=11, fontweight='bold')

    # --- Panel 3: Energy Fluxes ---
    ax3 = fig.add_subplot(gs[1, 0])

    # Compute flux statistics (in W/m²)
    flux_stats = {}
    for name in ['Q_solar', 'Q_atm', 'Q_emission', 'Q_conv', 'Q_net']:
        if name in fluxes:
            flux_stats[name] = float(np.mean(fluxes[name]))

    # Bar chart of mean fluxes
    flux_names = []
    flux_values = []
    flux_colors = []

    color_map = {
        'Q_solar': 'gold',
        'Q_atm': 'skyblue',
        'Q_emission': 'darkred',
        'Q_conv': 'lightcoral',
        'Q_net': 'black'
    }

    labels_map = {
        'Q_solar': 'Solar',
        'Q_atm': 'Atmospheric',
        'Q_emission': 'Emission',
        'Q_conv': 'Convection',
        'Q_net': 'Net'
    }

    for name in ['Q_solar', 'Q_atm', 'Q_emission', 'Q_conv', 'Q_net']:
        if name in flux_stats:
            flux_names.append(labels_map[name])
            flux_values.append(flux_stats[name])
            flux_colors.append(color_map[name])

    bars = ax3.barh(flux_names, flux_values, color=flux_colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, flux_values)):
        ax3.text(val, i, f' {val:.1f} W/m²', va='center',
                fontsize=9, fontweight='bold')

    ax3.set_xlabel('Mean Flux [W/m²]')
    ax3.set_title('Energy Flux Breakdown (Domain Average)', fontsize=11, fontweight='bold')
    ax3.axvline(0, color='black', linewidth=0.8)
    ax3.grid(axis='x', alpha=0.3)

    # --- Panel 4: Subsurface Temperature Profile ---
    ax4 = fig.add_subplot(gs[1, 1])

    # Extract subsurface profile at center point
    cy, cx = terrain.ny // 2, terrain.nx // 2
    T_profile = temp_field.T_subsurface[cy, cx, :] - 273.15  # Convert to Celsius
    depths = subsurface_grid.z_nodes

    ax4.plot(T_profile, -depths, 'o-', linewidth=2, markersize=6, color='darkred')
    ax4.set_xlabel('Temperature [°C]')
    ax4.set_ylabel('Depth [m]')
    ax4.set_title(f'Subsurface Profile (Center Point)\nSurface: {T_profile[0]:.1f}°C',
                 fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(T_profile[0], color='gray', linestyle='--', alpha=0.5, label='Surface T')

    # Add depth labels
    for i in [0, len(depths)//2, -1]:
        ax4.text(T_profile[i], -depths[i], f' {depths[i]:.2f}m',
                fontsize=8, va='center')

    # Overall title
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    fig.suptitle(f'Hourly Summary: {time_str}', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_flux_timeseries(diagnostics: list, output_path: Path, dpi: int = 150):
    """
    Plot energy flux time series from diagnostics.

    Parameters
    ----------
    diagnostics : list
        List of diagnostic dictionaries
    output_path : Path
        Output file path
    dpi : int
        Plot resolution
    """
    if not diagnostics:
        return

    # Extract data
    times = [datetime.fromisoformat(d['time']) for d in diagnostics]
    time_hours = [(t - times[0]).total_seconds() / 3600 for t in times]

    # Check if flux data is available
    has_fluxes = 'Q_solar_mean' in diagnostics[0]

    if has_fluxes:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        ax2 = None

    # Temperature
    T_mean = [d['T_surface_mean'] - 273.15 for d in diagnostics]
    T_min = [d['T_surface_min'] - 273.15 for d in diagnostics]
    T_max = [d['T_surface_max'] - 273.15 for d in diagnostics]

    ax1.plot(time_hours, T_mean, 'k-', linewidth=2, label='Mean')
    ax1.fill_between(time_hours, T_min, T_max, alpha=0.3, color='gray', label='Min/Max Range')
    ax1.set_ylabel('Surface Temperature [°C]')
    ax1.set_title('Surface Temperature Evolution', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Fluxes
    if has_fluxes and ax2 is not None:
        Q_solar = [d.get('Q_solar_mean', 0) for d in diagnostics]
        Q_atm = [d.get('Q_atm_mean', 0) for d in diagnostics]
        Q_emission = [d.get('Q_emission_mean', 0) for d in diagnostics]
        Q_conv = [d.get('Q_conv_mean', 0) for d in diagnostics]
        Q_net = [d.get('Q_net_mean', 0) for d in diagnostics]

        ax2.plot(time_hours, Q_solar, 'o-', label='Solar', color='gold', linewidth=1.5)
        ax2.plot(time_hours, Q_atm, 's-', label='Atmospheric', color='skyblue', linewidth=1.5)
        ax2.plot(time_hours, Q_emission, '^-', label='Emission', color='darkred', linewidth=1.5)
        ax2.plot(time_hours, Q_conv, 'v-', label='Convection', color='lightcoral', linewidth=1.5)
        ax2.plot(time_hours, Q_net, 'k-', label='Net', linewidth=2.5)

        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Energy Flux [W/m²]')
        ax2.set_title('Energy Flux Evolution (Domain Average)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', ncol=2)
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel('Time [hours]')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
