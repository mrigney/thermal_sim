"""
Simple example: Create synthetic terrain with materials.

This demonstrates the basic workflow of:
1. Creating or loading terrain
2. Loading material database
3. Assigning materials to terrain
"""

import sys
import os
# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.terrain import create_synthetic_terrain
from src.materials import create_representative_materials, MaterialField


def main():
    print("=" * 80)
    print("Thermal Terrain Simulator - Basic Example")
    print("=" * 80)
    
    # Create synthetic terrain
    print("\n1. Creating synthetic terrain...")
    nx, ny = 100, 100  # 100x100 grid
    dx, dy = 0.1, 0.1  # 0.1 m spacing
    
    terrain = create_synthetic_terrain(nx, ny, dx, dy, terrain_type='rolling_hills')
    print(terrain.get_info())
    
    # Compute sky view factors
    print("\n2. Computing sky view factors...")
    terrain.compute_sky_view_factor_simple()
    print(f"   Mean SVF: {terrain.sky_view_factor.mean():.3f}")
    print(f"   Min SVF: {terrain.sky_view_factor.min():.3f} (most occluded)")
    print(f"   Max SVF: {terrain.sky_view_factor.max():.3f} (most open)")
    
    # Load material database
    print("\n3. Loading material database...")
    material_db = create_representative_materials()
    material_db.print_summary()
    
    # Create material classification map
    print("\n4. Creating material classification map...")
    material_class = np.ones((ny, nx), dtype=np.int32)  # Start with all sand (class 1)
    
    # Add some granite patches (class 2) on high slopes
    high_slope_mask = terrain.slope > np.radians(20)  # slopes > 20 degrees
    material_class[high_slope_mask] = 2  # Granite on steep slopes
    
    # Add some soil (class 4) in valleys (low elevation)
    elevation_threshold = np.percentile(terrain.elevation, 30)
    low_elevation_mask = terrain.elevation < elevation_threshold
    material_class[low_elevation_mask] = 4  # Soil in low areas
    
    terrain.set_material_class(material_class)
    
    unique, counts = np.unique(material_class, return_counts=True)
    print(f"   Material distribution:")
    for mat_id, count in zip(unique, counts):
        mat = material_db.get_material(mat_id)
        percentage = 100 * count / (nx * ny)
        print(f"     Class {mat_id} ({mat.name}): {count} points ({percentage:.1f}%)")
    
    # Create material field
    print("\n5. Creating material property field...")
    nz = 20  # 20 subsurface layers
    mat_field = MaterialField(ny, nx, nz, material_db)
    mat_field.assign_from_classification(material_class)
    
    print(f"   Surface properties assigned:")
    print(f"     Alpha (absorptivity): [{mat_field.alpha.min():.3f}, {mat_field.alpha.max():.3f}]")
    print(f"     Epsilon (emissivity): [{mat_field.epsilon.min():.3f}, {mat_field.epsilon.max():.3f}]")
    print(f"     Roughness: [{mat_field.roughness.min():.4f}, {mat_field.roughness.max():.4f}] m")
    print(f"   Subsurface properties assigned:")
    print(f"     Conductivity k: [{mat_field.k.min():.2f}, {mat_field.k.max():.2f}] W/(m·K)")
    print(f"     Density ρ: [{mat_field.rho.min():.0f}, {mat_field.rho.max():.0f}] kg/m³")
    print(f"     Specific heat cp: [{mat_field.cp.min():.0f}, {mat_field.cp.max():.0f}] J/(kg·K)")
    
    # Compute some useful statistics
    print("\n6. Material thermal properties:")
    for mat_id in unique:
        mat = material_db.get_material(mat_id)
        mask = material_class == mat_id
        area = np.sum(mask) * dx * dy
        print(f"   {mat.name}:")
        print(f"     Area: {area:.2f} m²")
        print(f"     Thermal diffusivity: {mat.thermal_diffusivity()*1e6:.2f} × 10⁻⁶ m²/s")
        print(f"     Thermal inertia: {mat.thermal_inertia():.0f} J/(m²·K·s^(1/2))")
    
    print("\n" + "=" * 80)
    print("Setup complete! Ready for thermal simulation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
