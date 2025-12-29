"""
Visualization Module for Thermal Terrain Simulator

Provides plotting and visualization tools for:
- Terrain geometry (elevation, slope, aspect)
- Sky view factors
- Material distributions
- Temperature fields (surface and subsurface)
- Shadow maps
- Energy budget terms

Author: Thermal Terrain Simulator Project
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource, Normalize
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable


class TerrainVisualizer:
    """
    Visualization tools for terrain data and simulation results
    """

    def __init__(self, figsize=(12, 10)):
        """
        Initialize visualizer

        Parameters:
        -----------
        figsize : tuple
            Default figure size for plots
        """
        self.figsize = figsize
        self.cmap_terrain = 'terrain'
        self.cmap_thermal = 'hot'
        self.cmap_sky = 'viridis'

    def plot_elevation(self, terrain, ax=None, show_colorbar=True, hillshade=True):
        """
        Plot terrain elevation with optional hillshading

        Parameters:
        -----------
        terrain : TerrainGrid
            Terrain object with elevation data
        ax : matplotlib axis, optional
            Axis to plot on. If None, creates new figure
        show_colorbar : bool
            Whether to show colorbar
        hillshade : bool
            Whether to apply hillshading for 3D effect

        Returns:
        --------
        ax : matplotlib axis
        im : image object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Get extent in meters
        extent = [0, terrain.nx * terrain.dx, 0, terrain.ny * terrain.dy]

        if hillshade and hasattr(terrain, 'normals'):
            # Apply hillshading for realistic 3D appearance
            ls = LightSource(azdeg=315, altdeg=45)

            # Create hillshade
            hillshade_data = ls.hillshade(terrain.elevation,
                                         vert_exag=2.0,
                                         dx=terrain.dx,
                                         dy=terrain.dy)

            # Overlay elevation data on hillshade
            im = ax.imshow(terrain.elevation,
                          cmap=self.cmap_terrain,
                          extent=extent,
                          origin='lower',
                          alpha=0.7)
            ax.imshow(hillshade_data,
                     cmap='gray',
                     extent=extent,
                     origin='lower',
                     alpha=0.3)
        else:
            im = ax.imshow(terrain.elevation,
                          cmap=self.cmap_terrain,
                          extent=extent,
                          origin='lower')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Terrain Elevation')
        ax.set_aspect('equal')

        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Elevation (m)')

        return ax, im

    def plot_slope_aspect(self, terrain, figsize=None):
        """
        Plot slope and aspect side-by-side

        Parameters:
        -----------
        terrain : TerrainGrid
            Terrain object with slope and aspect data
        figsize : tuple, optional
            Figure size override

        Returns:
        --------
        fig : matplotlib figure
        axes : array of axes
        """
        if figsize is None:
            figsize = (16, 6)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        extent = [0, terrain.nx * terrain.dx, 0, terrain.ny * terrain.dy]

        # Plot slope (in degrees)
        slope_deg = np.rad2deg(terrain.slope)
        im1 = axes[0].imshow(slope_deg,
                            cmap='YlOrRd',
                            extent=extent,
                            origin='lower')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('Slope')
        axes[0].set_aspect('equal')

        divider = make_axes_locatable(axes[0])
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.set_label('Slope (degrees)')

        # Plot aspect (in degrees, 0=North, 90=East)
        aspect_deg = np.rad2deg(terrain.aspect)
        im2 = axes[1].imshow(aspect_deg,
                            cmap='hsv',
                            extent=extent,
                            origin='lower',
                            vmin=0,
                            vmax=360)
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        axes[1].set_title('Aspect')
        axes[1].set_aspect('equal')

        divider = make_axes_locatable(axes[1])
        cax2 = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('Aspect (degrees from N)')

        plt.tight_layout()
        return fig, axes

    def plot_sky_view_factor(self, terrain, ax=None, show_colorbar=True):
        """
        Plot sky view factor

        Parameters:
        -----------
        terrain : TerrainGrid
            Terrain object with sky_view_factor data
        ax : matplotlib axis, optional
            Axis to plot on
        show_colorbar : bool
            Whether to show colorbar

        Returns:
        --------
        ax : matplotlib axis
        im : image object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        extent = [0, terrain.nx * terrain.dx, 0, terrain.ny * terrain.dy]

        im = ax.imshow(terrain.sky_view_factor,
                      cmap=self.cmap_sky,
                      extent=extent,
                      origin='lower',
                      vmin=0,
                      vmax=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Sky View Factor')
        ax.set_aspect('equal')

        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('SVF (0=fully obstructed, 1=open sky)')

        return ax, im

    def plot_material_distribution(self, terrain, material_db, ax=None):
        """
        Plot material classification map with legend

        Parameters:
        -----------
        terrain : TerrainGrid
            Terrain object with material_class data
        material_db : MaterialDatabase
            Material database for getting material names
        ax : matplotlib axis, optional
            Axis to plot on

        Returns:
        --------
        ax : matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        extent = [0, terrain.nx * terrain.dx, 0, terrain.ny * terrain.dy]

        # Get unique material classes
        unique_classes = np.unique(terrain.material_class)
        n_classes = len(unique_classes)

        # Create custom colormap
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        im = ax.imshow(terrain.material_class,
                      cmap=plt.cm.colors.ListedColormap(colors),
                      extent=extent,
                      origin='lower',
                      interpolation='nearest')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Material Distribution')
        ax.set_aspect('equal')

        # Create legend with material names
        patches = []
        for i, class_id in enumerate(unique_classes):
            mat = material_db.get_material(int(class_id))
            label = f"{mat.name}" if mat else f"Class {class_id}"
            patches.append(mpatches.Patch(color=colors[i], label=label))

        ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.05, 0.5))

        return ax

    def plot_temperature_field(self, T_surface, terrain, ax=None,
                               show_colorbar=True, temp_range=None,
                               units='K', title='Surface Temperature'):
        """
        Plot surface temperature field

        Parameters:
        -----------
        T_surface : ndarray
            Surface temperature array (ny, nx)
        terrain : TerrainGrid
            Terrain object for extent
        ax : matplotlib axis, optional
            Axis to plot on
        show_colorbar : bool
            Whether to show colorbar
        temp_range : tuple, optional
            (vmin, vmax) for temperature scale. If None, uses data range
        units : str
            'K' for Kelvin or 'C' for Celsius
        title : str
            Plot title

        Returns:
        --------
        ax : matplotlib axis
        im : image object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        extent = [0, terrain.nx * terrain.dx, 0, terrain.ny * terrain.dy]

        # Convert temperature if needed
        if units == 'C':
            T_plot = T_surface - 273.15
            label = 'Temperature (°C)'
        else:
            T_plot = T_surface
            label = 'Temperature (K)'

        # Set color range
        if temp_range is None:
            vmin, vmax = T_plot.min(), T_plot.max()
        else:
            vmin, vmax = temp_range

        im = ax.imshow(T_plot,
                      cmap=self.cmap_thermal,
                      extent=extent,
                      origin='lower',
                      vmin=vmin,
                      vmax=vmax)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.set_aspect('equal')

        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(label)

        return ax, im

    def plot_subsurface_profile(self, T_subsurface, z_nodes,
                                i=None, j=None, x=None, y=None,
                                terrain=None, ax=None):
        """
        Plot vertical temperature profile at a specific location

        Parameters:
        -----------
        T_subsurface : ndarray
            Subsurface temperature array (ny, nx, nz)
        z_nodes : ndarray
            Depth of each subsurface node (m)
        i, j : int, optional
            Grid indices of location
        x, y : float, optional
            Physical coordinates (m). Requires terrain object
        terrain : TerrainGrid, optional
            Terrain object for coordinate conversion
        ax : matplotlib axis, optional
            Axis to plot on

        Returns:
        --------
        ax : matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))

        # Get location indices
        if i is None or j is None:
            if x is not None and y is not None and terrain is not None:
                i = int(x / terrain.dx)
                j = int(y / terrain.dy)
            else:
                # Default to center
                j, i = T_subsurface.shape[0]//2, T_subsurface.shape[1]//2

        # Extract profile
        T_profile = T_subsurface[j, i, :]

        # Plot
        ax.plot(T_profile - 273.15, -z_nodes, 'o-', linewidth=2, markersize=6)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3, label='Surface')

        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Subsurface Temperature Profile at ({i}, {j})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax

    def plot_terrain_overview(self, terrain, material_db=None):
        """
        Create comprehensive overview of terrain with multiple subplots

        Parameters:
        -----------
        terrain : TerrainGrid
            Terrain object with all geometric data
        material_db : MaterialDatabase, optional
            For material distribution plot

        Returns:
        --------
        fig : matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Create grid of subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # Elevation with hillshading
        self.plot_elevation(terrain, ax=ax1, hillshade=True)

        # Slope
        extent = [0, terrain.nx * terrain.dx, 0, terrain.ny * terrain.dy]
        slope_deg = np.rad2deg(terrain.slope)
        im2 = ax2.imshow(slope_deg, cmap='YlOrRd', extent=extent, origin='lower')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Slope')
        ax2.set_aspect('equal')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im2, cax=cax2, label='Slope (degrees)')

        # Sky view factor
        self.plot_sky_view_factor(terrain, ax=ax3)

        # Material distribution (if available)
        if material_db is not None:
            self.plot_material_distribution(terrain, material_db, ax=ax4)
        else:
            # Plot aspect instead
            aspect_deg = np.rad2deg(terrain.aspect)
            im4 = ax4.imshow(aspect_deg, cmap='hsv', extent=extent,
                           origin='lower', vmin=0, vmax=360)
            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Y (m)')
            ax4.set_title('Aspect')
            ax4.set_aspect('equal')
            divider4 = make_axes_locatable(ax4)
            cax4 = divider4.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im4, cax=cax4, label='Aspect (degrees from N)')

        fig.suptitle('Terrain Overview', fontsize=16, fontweight='bold')

        return fig

    def plot_shadow_map(self, shadow_map, terrain, sun_az=None, sun_el=None,
                       ax=None, show_colorbar=False):
        """
        Plot shadow map

        Parameters:
        -----------
        shadow_map : ndarray
            Boolean array (True = shadowed, False = sunlit)
        terrain : TerrainGrid
            Terrain object for extent
        sun_az : float, optional
            Sun azimuth (degrees) for title
        sun_el : float, optional
            Sun elevation (degrees) for title
        ax : matplotlib axis, optional
            Axis to plot on
        show_colorbar : bool
            Whether to show colorbar

        Returns:
        --------
        ax : matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        extent = [0, terrain.nx * terrain.dx, 0, terrain.ny * terrain.dy]

        # Plot as binary image (sunlit=1, shadowed=0)
        shadow_float = (~shadow_map).astype(float)

        im = ax.imshow(shadow_float,
                      cmap='gray',
                      extent=extent,
                      origin='lower',
                      vmin=0,
                      vmax=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        if sun_az is not None and sun_el is not None:
            ax.set_title(f'Shadow Map (Sun: az={sun_az:.1f}°, el={sun_el:.1f}°)')
        else:
            ax.set_title('Shadow Map')

        ax.set_aspect('equal')

        # Add sun direction arrow if position provided
        if sun_az is not None:
            # Convert azimuth to arrow direction
            arrow_len = min(terrain.nx * terrain.dx, terrain.ny * terrain.dy) * 0.15
            dx = arrow_len * np.sin(np.deg2rad(sun_az))
            dy = arrow_len * np.cos(np.deg2rad(sun_az))

            # Place arrow in top-right corner
            x0 = terrain.nx * terrain.dx * 0.85
            y0 = terrain.ny * terrain.dy * 0.85

            ax.arrow(x0, y0, dx, dy, head_width=arrow_len*0.2,
                    head_length=arrow_len*0.3, fc='yellow', ec='orange',
                    linewidth=2, alpha=0.8, label='Sun direction')
            ax.legend(loc='upper left')

        return ax

    def plot_temperature_time_series(self, times, T_values, location_label='',
                                    ax=None, units='C'):
        """
        Plot temperature evolution over time at a specific location

        Parameters:
        -----------
        times : array-like
            Time points (datetime or hours)
        T_values : array-like
            Temperature values
        location_label : str
            Label for the location
        ax : matplotlib axis, optional
            Axis to plot on
        units : str
            'K' or 'C'

        Returns:
        --------
        ax : matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))

        if units == 'C':
            T_plot = np.array(T_values) - 273.15
            ylabel = 'Temperature (°C)'
        else:
            T_plot = T_values
            ylabel = 'Temperature (K)'

        ax.plot(times, T_plot, '-o', linewidth=2, markersize=4, label=location_label)

        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.set_title('Temperature Evolution')
        ax.grid(True, alpha=0.3)

        if location_label:
            ax.legend()

        return ax


def create_animation_frames(T_surface_sequence, terrain, output_dir='./animation_frames',
                           temp_range=None, units='C'):
    """
    Create a sequence of images for animation

    Parameters:
    -----------
    T_surface_sequence : list of ndarrays
        List of temperature fields at different times
    terrain : TerrainGrid
        Terrain object
    output_dir : str
        Directory to save frames
    temp_range : tuple, optional
        (vmin, vmax) for consistent color scale
    units : str
        'K' or 'C'

    Returns:
    --------
    None (saves images to disk)
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    visualizer = TerrainVisualizer()

    # Determine temperature range if not provided
    if temp_range is None and units == 'C':
        all_temps = np.concatenate([T - 273.15 for T in T_surface_sequence])
        temp_range = (all_temps.min(), all_temps.max())
    elif temp_range is None:
        all_temps = np.concatenate(T_surface_sequence)
        temp_range = (all_temps.min(), all_temps.max())

    for i, T_surface in enumerate(T_surface_sequence):
        fig, ax = plt.subplots(figsize=(10, 8))
        visualizer.plot_temperature_field(T_surface, terrain, ax=ax,
                                         temp_range=temp_range, units=units,
                                         title=f'Surface Temperature - Frame {i}')

        filename = os.path.join(output_dir, f'frame_{i:04d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved frame {i+1}/{len(T_surface_sequence)}")

    print(f"Animation frames saved to {output_dir}")


# Convenience function
def quick_terrain_plot(terrain, material_db=None):
    """
    Quick visualization of terrain - convenience wrapper

    Parameters:
    -----------
    terrain : TerrainGrid
        Terrain object
    material_db : MaterialDatabase, optional
        Material database

    Returns:
    --------
    fig : matplotlib figure
    """
    vis = TerrainVisualizer()
    return vis.plot_terrain_overview(terrain, material_db)


def quick_temp_plot(T_surface, terrain, units='C'):
    """
    Quick visualization of temperature field - convenience wrapper

    Parameters:
    -----------
    T_surface : ndarray
        Surface temperature array
    terrain : TerrainGrid
        Terrain object
    units : str
        'K' or 'C'

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    vis = TerrainVisualizer()
    fig, ax = plt.subplots(figsize=(10, 8))
    vis.plot_temperature_field(T_surface, terrain, ax=ax, units=units)
    return fig, ax
