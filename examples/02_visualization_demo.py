"""
Example 02: Visualization Demo

Demonstrates the visualization capabilities for the thermal terrain simulator.
Shows how to visualize:
- Terrain geometry (elevation, slope, aspect)
- Sky view factors
- Material distributions
- Synthetic temperature fields

This example can be run before any simulation to verify visualization setup.

Usage:
    python 02_visualization_demo.py
"""

import sys
import os
# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from src.terrain import TerrainGrid, create_synthetic_terrain
from src.materials import MaterialDatabase, MaterialField
from src.visualization import TerrainVisualizer, quick_terrain_plot, quick_temp_plot


def main():
    print("=" * 70)
    print("Thermal Terrain Simulator - Visualization Demo")
    print("=" * 70)
    print()

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Create synthetic terrain
    # =========================================================================
    print("Creating synthetic terrain...")

    # Create a 50m x 50m domain with 0.5m resolution (coarse for demo)
    nx, ny = 100, 100
    dx, dy = 0.5, 0.5  # meters

    # Generate interesting terrain - rolling hills with a ridge
    print("  - Generating rolling hills with ridge...")
    terrain = create_synthetic_terrain(nx, ny, dx, dy, terrain_type='rolling_hills')

    # Compute derived quantities
    print("  - Computing surface normals...")
    terrain.compute_normals()

    print("  - Computing slopes and aspects...")
    terrain.compute_slope_aspect()

    print("  - Computing sky view factors...")
    terrain.compute_sky_view_factor_simple()

    # =========================================================================
    # Assign materials based on terrain features
    # =========================================================================
    print("\nAssigning materials based on terrain features...")

    # Load material database
    material_db = MaterialDatabase()
    materials_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'materials', 'representative_materials.json')
    material_db.load_from_json(materials_path)

    print(f"  - Loaded {len(material_db.materials)} materials")

    # Create simple classification:
    # - High elevations and steep slopes: rock (granite, basalt)
    # - Medium slopes: sandstone, gravel
    # - Low slopes: sand, dry soil
    terrain.material_class = np.ones((ny, nx), dtype=np.int32)

    elevation_norm = (terrain.elevation - terrain.elevation.min()) / \
                     (terrain.elevation.max() - terrain.elevation.min() + 1e-10)
    slope_deg = np.rad2deg(terrain.slope)

    # Assign based on elevation and slope
    for j in range(ny):
        for i in range(nx):
            if elevation_norm[j, i] > 0.7 and slope_deg[j, i] > 15:
                terrain.material_class[j, i] = 3  # Basalt (dark rock)
            elif elevation_norm[j, i] > 0.5 or slope_deg[j, i] > 10:
                terrain.material_class[j, i] = 5  # Sandstone
            elif slope_deg[j, i] > 5:
                terrain.material_class[j, i] = 6  # Gravel
            elif elevation_norm[j, i] < 0.3:
                terrain.material_class[j, i] = 4  # Dry soil
            else:
                terrain.material_class[j, i] = 1  # Dry sand

    # Create material field (with subsurface layers for later)
    nz = 20  # 20 subsurface layers
    mat_field = MaterialField(ny, nx, nz, material_db)
    mat_field.assign_from_classification(terrain.material_class)

    print("  - Material distribution:")
    unique, counts = np.unique(terrain.material_class, return_counts=True)
    for class_id, count in zip(unique, counts):
        mat = material_db.get_material(class_id)
        percentage = 100 * count / (nx * ny)
        print(f"      {mat.name}: {percentage:.1f}%")

    # =========================================================================
    # Create synthetic temperature field for visualization demo
    # =========================================================================
    print("\nGenerating synthetic temperature field for demo...")

    # Simulate a daytime scenario with differential heating
    T_air = 25.0 + 273.15  # 25°C ambient

    # Base temperature
    T_surface = np.ones((ny, nx)) * T_air

    # Add effects:
    # 1. Elevation cooling (lapse rate effect)
    T_surface -= 0.0065 * terrain.elevation  # 6.5 K/km lapse rate

    # 2. Slope heating (south-facing slopes warmer)
    # Assume sun from south, high elevation
    sun_direction = np.array([0, 1, 1])  # From south, 45° elevation
    sun_direction = sun_direction / np.linalg.norm(sun_direction)

    # Dot product of normals with sun direction
    cos_angle = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            cos_angle[j, i] = max(0, np.dot(terrain.normals[j, i], sun_direction))

    # Add differential heating (0-15K based on sun angle)
    T_surface += 15.0 * cos_angle

    # 3. Material-dependent heating (darker materials warmer)
    for j in range(ny):
        for i in range(nx):
            absorptivity = mat_field.alpha[j, i]
            T_surface[j, i] += 10.0 * (absorptivity - 0.5)  # ±5K variation

    # 4. Add some random noise for realism
    T_surface += np.random.normal(0, 0.5, (ny, nx))

    print(f"  - Temperature range: {T_surface.min()-273.15:.1f}°C to {T_surface.max()-273.15:.1f}°C")

    # =========================================================================
    # Create synthetic subsurface profile
    # =========================================================================
    print("\nGenerating synthetic subsurface temperature profile...")

    # Simple subsurface grid
    nz = 20
    z_max = 2.0  # 2 meters depth
    z_nodes = np.linspace(0, z_max, nz)

    # Create subsurface temps (simple 1D profiles at each location)
    T_subsurface = np.zeros((ny, nx, nz))

    for j in range(ny):
        for i in range(nx):
            # Surface temperature
            T_surf = T_surface[j, i]
            # Deep temperature (less variation)
            T_deep = T_air
            # Linear interpolation with depth (simplified)
            for k in range(nz):
                depth_fraction = z_nodes[k] / z_max
                T_subsurface[j, i, k] = T_surf * (1 - depth_fraction) + T_deep * depth_fraction

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    visualizer = TerrainVisualizer()

    # -------------------------------------------------------------------------
    # 1. Comprehensive terrain overview
    # -------------------------------------------------------------------------
    print("\n1. Creating terrain overview...")
    fig1 = visualizer.plot_terrain_overview(terrain, material_db)
    plt.savefig(os.path.join(output_dir, 'output_terrain_overview.png'), dpi=150, bbox_inches='tight')
    print("   Saved: output_terrain_overview.png")

    # -------------------------------------------------------------------------
    # 2. Temperature field
    # -------------------------------------------------------------------------
    print("\n2. Creating temperature field plot...")
    fig2, ax2 = quick_temp_plot(T_surface, terrain, units='C')
    plt.savefig(os.path.join(output_dir, 'output_temperature_field.png'), dpi=150, bbox_inches='tight')
    print("   Saved: output_temperature_field.png")

    # -------------------------------------------------------------------------
    # 3. Subsurface profiles at multiple locations
    # -------------------------------------------------------------------------
    print("\n3. Creating subsurface temperature profiles...")
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

    # Plot profiles at three locations: hot spot, cold spot, average
    i_hot, j_hot = np.unravel_index(np.argmax(T_surface), T_surface.shape)
    i_cold, j_cold = np.unravel_index(np.argmin(T_surface), T_surface.shape)
    i_avg, j_avg = ny//2, nx//2

    visualizer.plot_subsurface_profile(T_subsurface, z_nodes, i=j_hot, j=i_hot, ax=axes3[0])
    axes3[0].set_title(f'Hottest Point\n({j_hot}, {i_hot})')

    visualizer.plot_subsurface_profile(T_subsurface, z_nodes, i=j_avg, j=i_avg, ax=axes3[1])
    axes3[1].set_title(f'Center Point\n({j_avg}, {i_avg})')

    visualizer.plot_subsurface_profile(T_subsurface, z_nodes, i=j_cold, j=i_cold, ax=axes3[2])
    axes3[2].set_title(f'Coldest Point\n({j_cold}, {i_cold})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'output_subsurface_profiles.png'), dpi=150, bbox_inches='tight')
    print("   Saved: output_subsurface_profiles.png")

    # -------------------------------------------------------------------------
    # 4. Individual terrain features
    # -------------------------------------------------------------------------
    print("\n4. Creating individual feature plots...")

    # Elevation with hillshade
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    visualizer.plot_elevation(terrain, ax=ax4, hillshade=True)
    plt.savefig(os.path.join(output_dir, 'output_elevation.png'), dpi=150, bbox_inches='tight')
    print("   Saved: output_elevation.png")

    # Slope and aspect
    fig5, axes5 = visualizer.plot_slope_aspect(terrain)
    plt.savefig(os.path.join(output_dir, 'output_slope_aspect.png'), dpi=150, bbox_inches='tight')
    print("   Saved: output_slope_aspect.png")

    # Sky view factor
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    visualizer.plot_sky_view_factor(terrain, ax=ax6)
    plt.savefig(os.path.join(output_dir, 'output_sky_view_factor.png'), dpi=150, bbox_inches='tight')
    print("   Saved: output_sky_view_factor.png")

    # -------------------------------------------------------------------------
    # 5. Demonstration of shadow visualization
    # -------------------------------------------------------------------------
    print("\n5. Creating synthetic shadow map visualization...")

    # Create synthetic shadow based on sun direction
    # Sun from southeast at 45° elevation
    sun_az = 135.0  # degrees from north (clockwise)
    sun_el = 45.0   # degrees above horizon

    # Simple shadow calculation (very rough - just for visualization demo)
    shadow_map = np.zeros((ny, nx), dtype=bool)
    sun_vec = np.array([
        np.sin(np.deg2rad(sun_az)) * np.cos(np.deg2rad(sun_el)),
        np.cos(np.deg2rad(sun_az)) * np.cos(np.deg2rad(sun_el)),
        np.sin(np.deg2rad(sun_el))
    ])

    # Mark as shadowed if surface faces away from sun
    for j in range(ny):
        for i in range(nx):
            cos_sun = np.dot(terrain.normals[j, i], sun_vec)
            if cos_sun < 0.1:  # Facing away from sun
                shadow_map[j, i] = True

    fig7, ax7 = plt.subplots(figsize=(10, 8))
    visualizer.plot_shadow_map(shadow_map, terrain, sun_az=sun_az, sun_el=sun_el, ax=ax7)
    plt.savefig(os.path.join(output_dir, 'output_shadow_map.png'), dpi=150, bbox_inches='tight')
    print("   Saved: output_shadow_map.png")

    # -------------------------------------------------------------------------
    # 6. Synthetic time series
    # -------------------------------------------------------------------------
    print("\n6. Creating temperature time series...")

    # Generate synthetic diurnal cycle at center point
    hours = np.linspace(0, 24, 49)  # 24 hours, every 30 min
    T_base = T_air
    T_amplitude = 15.0  # 15K diurnal variation

    # Simple sinusoidal variation (max at 2pm, min at 2am)
    T_series = T_base + T_amplitude * np.sin(2*np.pi*(hours - 8)/24)

    fig8, ax8 = plt.subplots(figsize=(12, 5))
    visualizer.plot_temperature_time_series(hours, T_series,
                                           location_label='Center point',
                                           ax=ax8, units='C')
    ax8.set_xlabel('Time (hours)')
    plt.savefig(os.path.join(output_dir, 'output_time_series.png'), dpi=150, bbox_inches='tight')
    print("   Saved: output_time_series.png")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Visualization Demo Complete!")
    print("=" * 70)
    print("\nGenerated output files:")
    print("  - output_terrain_overview.png      (comprehensive terrain view)")
    print("  - output_temperature_field.png     (surface temperatures)")
    print("  - output_subsurface_profiles.png   (vertical T profiles)")
    print("  - output_elevation.png             (hillshaded elevation)")
    print("  - output_slope_aspect.png          (slope and aspect maps)")
    print("  - output_sky_view_factor.png       (sky view factors)")
    print("  - output_shadow_map.png            (shadow demonstration)")
    print("  - output_time_series.png           (temperature evolution)")
    print()
    print("You can now use these visualization tools with real simulation data!")
    print()
    print("All plots saved successfully!")


if __name__ == '__main__':
    main()
