"""
Materials module for thermal simulation.

Manages material properties including thermal conductivity, density, specific heat,
optical properties, and surface characteristics.
"""

import numpy as np
import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MaterialProperties:
    """
    Properties for a single material type.
    
    Attributes:
        class_id: Unique integer identifier
        name: Material name
        alpha: Solar absorptivity [dimensionless, 0-1]
        epsilon: Thermal emissivity [dimensionless, 0-1]
        k: Thermal conductivity [W/(m·K)]
        rho: Density [kg/m³]
        cp: Specific heat capacity [J/(kg·K)]
        roughness: Surface roughness length [m] (for convection calculations)
    """
    class_id: int
    name: str
    alpha: float  # Solar absorptivity
    epsilon: float  # Thermal emissivity
    k: float  # Thermal conductivity [W/(m·K)]
    rho: float  # Density [kg/m³]
    cp: float  # Specific heat [J/(kg·K)]
    roughness: float  # Surface roughness [m]
    
    def thermal_diffusivity(self) -> float:
        """Compute thermal diffusivity α = k/(ρ·cp) [m²/s]"""
        return self.k / (self.rho * self.cp)
    
    def thermal_inertia(self) -> float:
        """Compute thermal inertia sqrt(k·ρ·cp) [J/(m²·K·s^(1/2))]"""
        return np.sqrt(self.k * self.rho * self.cp)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'class_id': self.class_id,
            'name': self.name,
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'k': self.k,
            'rho': self.rho,
            'cp': self.cp,
            'roughness': self.roughness
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        """Create from dictionary."""
        return cls(**d)


class MaterialDatabase:
    """
    Database of material properties.
    
    Manages multiple material types and provides lookup functionality.
    """
    
    def __init__(self):
        self.materials: Dict[int, MaterialProperties] = {}
        
    def add_material(self, material: MaterialProperties):
        """Add a material to the database."""
        if material.class_id in self.materials:
            print(f"Warning: Overwriting material with class_id {material.class_id}")
        self.materials[material.class_id] = material
        
    def get_material(self, class_id: int) -> Optional[MaterialProperties]:
        """Get material by class ID."""
        return self.materials.get(class_id)

    def get_material_by_name(self, name: str) -> Optional[MaterialProperties]:
        """Get material by name (case-insensitive)."""
        name_lower = name.lower()
        for material in self.materials.values():
            if material.name.lower() == name_lower:
                return material
        return None

    def get_all_ids(self) -> List[int]:
        """Get list of all material class IDs."""
        return list(self.materials.keys())
    
    def save_to_json(self, filename: str):
        """Save database to JSON file."""
        data = {
            'materials': [mat.to_dict() for mat in self.materials.values()]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_json(self, filename: str):
        """Load database from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.materials = {}
        for mat_dict in data['materials']:
            material = MaterialProperties.from_dict(mat_dict)
            self.add_material(material)
    
    def print_summary(self):
        """Print summary of all materials in database."""
        print(f"\nMaterial Database: {len(self.materials)} materials")
        print("=" * 80)
        print(f"{'ID':<4} {'Name':<20} {'α':<6} {'ε':<6} {'k':<8} {'ρ':<8} {'cp':<8} {'rough':<8}")
        print(f"{'':4} {'':20} {'':6} {'':6} {'W/m·K':<8} {'kg/m³':<8} {'J/kg·K':<8} {'m':<8}")
        print("-" * 80)
        
        for class_id in sorted(self.materials.keys()):
            mat = self.materials[class_id]
            print(f"{mat.class_id:<4} {mat.name:<20} {mat.alpha:<6.3f} {mat.epsilon:<6.3f} "
                  f"{mat.k:<8.2f} {mat.rho:<8.1f} {mat.cp:<8.1f} {mat.roughness:<8.4f}")
        print("=" * 80)


class MaterialField:
    """
    Spatially-varying material properties on a grid.
    
    Maps material properties from database onto terrain grid based on
    classification map.
    """
    
    def __init__(self, ny: int, nx: int, nz: int, material_db: MaterialDatabase):
        """
        Initialize material field.
        
        Args:
            ny: Number of grid points in y
            nx: Number of grid points in x
            nz: Number of subsurface layers
            material_db: Material database
        """
        self.ny = ny
        self.nx = nx
        self.nz = nz
        self.material_db = material_db
        
        # Surface properties (2D)
        self.alpha = np.zeros((ny, nx), dtype=np.float32)
        self.epsilon = np.zeros((ny, nx), dtype=np.float32)
        self.roughness = np.zeros((ny, nx), dtype=np.float32)
        
        # Subsurface properties (3D) - could vary with depth
        self.k = np.zeros((ny, nx, nz), dtype=np.float32)
        self.rho = np.zeros((ny, nx, nz), dtype=np.float32)
        self.cp = np.zeros((ny, nx, nz), dtype=np.float32)
        
    def assign_from_classification(self, material_class: np.ndarray):
        """
        Assign material properties based on classification map.
        
        Args:
            material_class: Integer array of material class IDs, shape (ny, nx)
        """
        if material_class.shape != (self.ny, self.nx):
            raise ValueError(f"Classification map shape {material_class.shape} doesn't match grid size ({self.ny}, {self.nx})")
        
        # Get unique material classes in the map
        unique_classes = np.unique(material_class)
        
        # Check that all classes exist in database
        for class_id in unique_classes:
            if class_id not in self.material_db.materials:
                raise ValueError(f"Material class {class_id} not found in database")
        
        # Assign properties for each class
        for class_id in unique_classes:
            mask = (material_class == class_id)
            material = self.material_db.get_material(class_id)
            
            # Surface properties
            self.alpha[mask] = material.alpha
            self.epsilon[mask] = material.epsilon
            self.roughness[mask] = material.roughness
            
            # Subsurface properties (same at all depths for now)
            for iz in range(self.nz):
                self.k[mask, iz] = material.k
                self.rho[mask, iz] = material.rho
                self.cp[mask, iz] = material.cp
        
        print(f"Assigned material properties for {len(unique_classes)} material classes")


class MaterialFieldDepthVarying(MaterialField):
    """
    Spatially-varying material properties with depth-varying thermal characteristics.

    Extends MaterialField to support depth-varying thermal properties from
    SQLite database. Inherits surface property handling from base class.

    Key Differences from MaterialField:
    - Accepts MaterialDatabaseSQLite instead of MaterialDatabase
    - Interpolates depth-varying properties to subsurface grid depths
    - Fills 3D arrays (k, rho, cp) with depth-dependent values
    - Surface properties (alpha, epsilon, roughness) remain 2D

    Usage:
        db = MaterialDatabaseSQLite("materials.db")
        materials = MaterialFieldDepthVarying(ny, nx, nz, db)
        materials.assign_from_classification(material_class, subsurface_grid)
    """

    def __init__(self, ny: int, nx: int, nz: int, material_db):
        """
        Initialize material field with SQLite database support.

        Args:
            ny: Number of grid points in y
            nx: Number of grid points in x
            nz: Number of subsurface layers
            material_db: MaterialDatabaseSQLite instance
        """
        # Import here to avoid circular dependency
        from src.materials_db import MaterialDatabaseSQLite

        if not isinstance(material_db, MaterialDatabaseSQLite):
            raise TypeError(
                "MaterialFieldDepthVarying requires MaterialDatabaseSQLite, "
                f"got {type(material_db).__name__}"
            )

        # Call parent constructor
        super().__init__(ny, nx, nz, material_db)

    def assign_from_classification(self, material_class: np.ndarray,
                                   subsurface_grid=None):
        """
        Assign material properties based on classification map.

        If subsurface_grid is provided, interpolates depth-varying thermal
        properties to grid depths. Otherwise falls back to uniform depth
        (broadcasts surface properties).

        Args:
            material_class: Integer array of material class IDs or names, shape (ny, nx)
            subsurface_grid: SubsurfaceGrid instance for depth interpolation (optional)
        """
        if material_class.shape != (self.ny, self.nx):
            raise ValueError(
                f"Classification map shape {material_class.shape} doesn't match "
                f"grid size ({self.ny}, {self.nx})"
            )

        if subsurface_grid is not None:
            # Depth-varying assignment
            self._assign_depth_varying(material_class, subsurface_grid)
        else:
            # Fallback to uniform depth (broadcast surface properties)
            print("Warning: No subsurface_grid provided, using uniform depth properties")
            self._assign_uniform_depth(material_class)

    def _assign_depth_varying(self, material_class: np.ndarray, subsurface_grid):
        """
        Assign depth-varying properties from SQLite database.

        For each unique material in classification map:
        1. Query material from database
        2. Interpolate properties to subsurface grid depths
        3. Fill 3D arrays with interpolated values

        Args:
            material_class: Integer array of material class IDs, shape (ny, nx)
            subsurface_grid: SubsurfaceGrid with z_nodes array
        """
        # Get target depths from subsurface grid
        target_depths = subsurface_grid.z_nodes

        # Get unique material classes in the map
        unique_classes = np.unique(material_class)

        print(f"Assigning depth-varying properties for {len(unique_classes)} materials")
        print(f"  Interpolating to {len(target_depths)} depth points: "
              f"{target_depths[0]:.4f} to {target_depths[-1]:.4f} m")

        # Process each material
        for class_id in unique_classes:
            # Get material from database
            # Support both integer IDs and string names
            if isinstance(class_id, (int, np.integer)):
                # For integer IDs, we need to map to material names
                # This requires the classification map to use material_id values
                # For now, treat integers as material names for legacy compatibility
                material = self.material_db.get_material(str(class_id))
                if material is None:
                    # Try as material name lookup
                    material = self.material_db.get_material_by_name(str(class_id))
            else:
                # String name
                material = self.material_db.get_material_by_name(str(class_id))

            if material is None:
                # Try listing all materials to help debug
                available = self.material_db.list_materials()
                available_names = [name for _, name, _ in available]
                raise ValueError(
                    f"Material '{class_id}' not found in database. "
                    f"Available materials: {available_names}"
                )

            # Get mask for this material
            mask = (material_class == class_id)

            # Assign surface properties (2D, same as base class)
            self.alpha[mask] = material.alpha
            self.epsilon[mask] = material.epsilon
            self.roughness[mask] = material.roughness

            # Interpolate thermal properties to target depths
            interp_props = material.interpolate_to_depths(target_depths)

            # Assign depth-varying thermal properties (3D)
            for iz in range(self.nz):
                self.k[mask, iz] = interp_props['k'][iz]
                self.rho[mask, iz] = interp_props['rho'][iz]
                self.cp[mask, iz] = interp_props['cp'][iz]

            # Print summary for this material
            print(f"  - '{material.name}': {len(material.depths)} depth points -> "
                  f"{self.nz} grid layers")
            print(f"    k range: {interp_props['k'].min():.3f} to "
                  f"{interp_props['k'].max():.3f} W/(m*K)")

    def _assign_uniform_depth(self, material_class: np.ndarray):
        """
        Fallback: Assign uniform properties (broadcast to all depths).

        Uses surface values for all depths, similar to legacy MaterialField.

        Args:
            material_class: Integer array of material class IDs, shape (ny, nx)
        """
        unique_classes = np.unique(material_class)

        print(f"Warning: Assigning uniform (non-depth-varying) properties")

        for class_id in unique_classes:
            # Get material from database
            material = self.material_db.get_material_by_name(str(class_id))

            if material is None:
                raise ValueError(f"Material '{class_id}' not found in database")

            mask = (material_class == class_id)

            # Surface properties
            self.alpha[mask] = material.alpha
            self.epsilon[mask] = material.epsilon
            self.roughness[mask] = material.roughness

            # Subsurface properties (use surface values, broadcast to all depths)
            k_surface = material.k[0]  # First depth point
            rho_surface = material.rho[0]
            cp_surface = material.cp[0]

            for iz in range(self.nz):
                self.k[mask, iz] = k_surface
                self.rho[mask, iz] = rho_surface
                self.cp[mask, iz] = cp_surface


def create_representative_materials() -> MaterialDatabase:
    """
    Create a representative material database for testing.
    
    Includes common desert terrain materials with realistic properties.
    
    Returns:
        MaterialDatabase with representative materials
    """
    db = MaterialDatabase()
    
    # Material 1: Dry sand
    db.add_material(MaterialProperties(
        class_id=1,
        name="Dry Sand",
        alpha=0.60,  # Moderate solar absorption
        epsilon=0.90,  # High emissivity
        k=0.30,  # Low thermal conductivity
        rho=1600.0,  # kg/m³
        cp=800.0,  # J/(kg·K)
        roughness=0.001  # Smooth surface
    ))
    
    # Material 2: Granite/rock
    db.add_material(MaterialProperties(
        class_id=2,
        name="Granite",
        alpha=0.55,  # Moderate absorption (lighter colored)
        epsilon=0.95,  # Very high emissivity
        k=2.5,  # High thermal conductivity
        rho=2700.0,  # kg/m³
        cp=790.0,  # J/(kg·K)
        roughness=0.01  # Rough surface
    ))
    
    # Material 3: Basalt (darker rock)
    db.add_material(MaterialProperties(
        class_id=3,
        name="Basalt",
        alpha=0.75,  # High absorption (dark)
        epsilon=0.95,  # Very high emissivity
        k=2.0,  # High thermal conductivity
        rho=2900.0,  # kg/m³
        cp=840.0,  # J/(kg·K)
        roughness=0.015  # Rough surface
    ))
    
    # Material 4: Dry soil
    db.add_material(MaterialProperties(
        class_id=4,
        name="Dry Soil",
        alpha=0.65,  # Moderate-high absorption
        epsilon=0.92,  # High emissivity
        k=0.4,  # Low-moderate conductivity
        rho=1500.0,  # kg/m³
        cp=850.0,  # J/(kg·K)
        roughness=0.005  # Moderately smooth
    ))
    
    # Material 5: Sandstone
    db.add_material(MaterialProperties(
        class_id=5,
        name="Sandstone",
        alpha=0.50,  # Lower absorption (light colored)
        epsilon=0.93,  # High emissivity
        k=1.8,  # Moderate-high conductivity
        rho=2200.0,  # kg/m³
        cp=800.0,  # J/(kg·K)
        roughness=0.02  # Rough surface
    ))
    
    # Material 6: Gravel
    db.add_material(MaterialProperties(
        class_id=6,
        name="Gravel",
        alpha=0.62,  # Moderate absorption
        epsilon=0.91,  # High emissivity
        k=0.8,  # Moderate conductivity (air gaps between rocks)
        rho=1800.0,  # kg/m³ (bulk density with air gaps)
        cp=820.0,  # J/(kg·K)
        roughness=0.02  # Very rough surface
    ))
    
    return db


if __name__ == "__main__":
    # Test the materials module
    print("Testing materials module...")
    
    # Create representative database
    db = create_representative_materials()
    db.print_summary()
    
    # Save to file
    db.save_to_json("/home/claude/thermal_terrain_sim/data/materials/representative_materials.json")
    print("\nSaved representative materials to data/materials/representative_materials.json")
    
    # Test material field assignment
    print("\nTesting material field assignment...")
    ny, nx, nz = 50, 50, 10
    
    # Create a simple classification map (checkerboard pattern)
    material_class = np.zeros((ny, nx), dtype=np.int32)
    material_class[::2, ::2] = 1  # Dry sand
    material_class[1::2, 1::2] = 1  # Dry sand
    material_class[::2, 1::2] = 2  # Granite
    material_class[1::2, ::2] = 2  # Granite
    
    # Create material field
    mat_field = MaterialField(ny, nx, nz, db)
    mat_field.assign_from_classification(material_class)
    
    print(f"Alpha range: [{mat_field.alpha.min():.3f}, {mat_field.alpha.max():.3f}]")
    print(f"Epsilon range: [{mat_field.epsilon.min():.3f}, {mat_field.epsilon.max():.3f}]")
    print(f"k range: [{mat_field.k.min():.2f}, {mat_field.k.max():.2f}] W/(m·K)")
    
    print("\nMaterials module test complete!")
