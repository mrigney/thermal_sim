"""
Create materials database with both legacy and depth-varying materials.

This script populates a new SQLite materials database with:
1. Legacy materials from JSON (migrated as single-depth entries)
2. New depth-varying materials with proper source citations

Output: data/materials/materials.db

Usage:
    python scripts/create_materials_database.py

Author: Thermal Simulator Team
Date: January 2025
"""

import numpy as np
import uuid
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.materials_db import MaterialDatabaseSQLite, MaterialPropertiesDepth
from src.materials import MaterialDatabase, create_representative_materials


def create_database(db_path: str, overwrite: bool = False):
    """
    Create and populate materials database.

    Parameters
    ----------
    db_path : str
        Path to output database file
    overwrite : bool
        If True, delete existing database before creating new one
    """
    db_path = Path(db_path)

    # Check if database exists
    if db_path.exists():
        if overwrite:
            print(f"Deleting existing database: {db_path}")
            db_path.unlink()
        else:
            response = input(f"Database {db_path} exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            db_path.unlink()

    # Create parent directory if needed
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("Creating Materials Database")
    print("="*70)
    print(f"Output: {db_path}\n")

    # Create database
    db = MaterialDatabaseSQLite(str(db_path), create_if_missing=True)

    # Add materials
    n_materials = 0

    # =========================================================================
    # PART 1: Migrate legacy materials from JSON
    # =========================================================================
    print("PART 1: Migrating legacy materials from JSON")
    print("-" * 70)

    # Load legacy materials
    legacy_db = create_representative_materials()

    for class_id, mat in legacy_db.materials.items():
        # Convert to depth-varying format (single depth point at z=0)
        mat_depth = MaterialPropertiesDepth(
            material_id=str(uuid.uuid4()),
            name=mat.name,
            version=1,
            source_database="legacy_json",
            source_citation="Migrated from representative_materials.json",
            depths=np.array([0.0]),  # Single depth point
            k=np.array([mat.k]),
            rho=np.array([mat.rho]),
            cp=np.array([mat.cp]),
            alpha=mat.alpha,
            epsilon=mat.epsilon,
            roughness=mat.roughness
        )

        mat_id = db.add_material(mat_depth)
        n_materials += 1
        print(f"  [{n_materials}] Migrated: {mat.name}")
        print(f"      k={mat.k:.3f} W/(m*K), rho={mat.rho:.0f} kg/m^3, cp={mat.cp:.0f} J/(kg*K)")

    print(f"\n[OK] Migrated {n_materials} legacy materials\n")

    # =========================================================================
    # PART 2: Add depth-varying materials
    # =========================================================================
    print("PART 2: Adding depth-varying materials")
    print("-" * 70)

    # Material 1: Desert Sand (Depth-Varying)
    # Based on Presley & Christensen (1997) and NASA TPSX data
    # Surface: loose, low density, low conductivity
    # Depth: compacted, higher density, higher conductivity
    desert_sand = MaterialPropertiesDepth(
        material_id=str(uuid.uuid4()),
        name="Desert Sand (Depth-Varying)",
        version=1,
        source_database="NASA TPSX + literature",
        source_citation="Presley, M.A. & Christensen, P.R. (1997). Thermal conductivity measurements of particulate materials: 2. Results. JGR, 102(E3), 6551-6566.",
        notes="Depth profile represents compaction from loose surface to consolidated subsurface. Typical desert sand composition.",
        depths=np.array([0.0, 0.05, 0.10, 0.20, 0.50]),  # meters
        k=np.array([0.30, 0.35, 0.40, 0.45, 0.50]),  # W/(m*K)
        rho=np.array([1600, 1650, 1700, 1750, 1800]),  # kg/m^3
        cp=np.array([800, 810, 820, 830, 840]),  # J/(kg*K)
        alpha=0.60,  # Solar absorptivity
        epsilon=0.90,  # Thermal emissivity
        roughness=0.001  # Surface roughness [m]
    )
    mat_id = db.add_material(desert_sand)
    n_materials += 1
    print(f"  [{n_materials}] Added: {desert_sand.name}")
    print(f"      Depths: {desert_sand.depths}")
    print(f"      k range: {desert_sand.k.min():.2f} - {desert_sand.k.max():.2f} W/(m*K)")
    I_surface = desert_sand.thermal_inertia()[0] if len(desert_sand.depths) > 1 else desert_sand.thermal_inertia()
    print(f"      Thermal inertia (surface): {I_surface:.0f} J/(m^2*K*s^0.5)")
    print(f"      Source: {desert_sand.source_citation}")

    # Material 2: Basalt (Weathered to Fresh)
    # Based on Christensen (1986) and CINDAS TPMD
    # Surface: weathered, fractured, low conductivity
    # Depth: fresh bedrock, high conductivity
    basalt_weathered = MaterialPropertiesDepth(
        material_id=str(uuid.uuid4()),
        name="Basalt (Weathered to Fresh)",
        version=1,
        source_database="CINDAS TPMD + literature",
        source_citation="Christensen, P.R. (1986). The spatial distribution of rocks on Mars. Icarus, 68(2), 217-238.",
        notes="Weathering profile from fractured surface basalt to fresh bedrock. Based on terrestrial and Martian analog measurements.",
        depths=np.array([0.0, 0.02, 0.05, 0.10, 0.50]),  # meters
        k=np.array([1.5, 1.8, 2.0, 2.0, 2.0]),  # W/(m*K) - weathered to fresh
        rho=np.array([2700, 2800, 2900, 2900, 2900]),  # kg/m^3
        cp=np.array([840, 840, 840, 840, 840]),  # J/(kg*K) - roughly constant for basalt
        alpha=0.75,  # Dark rock, high solar absorption
        epsilon=0.95,  # High thermal emissivity
        roughness=0.01  # Rough surface [m]
    )
    mat_id = db.add_material(basalt_weathered)
    n_materials += 1
    print(f"\n  [{n_materials}] Added: {basalt_weathered.name}")
    print(f"      Depths: {basalt_weathered.depths}")
    print(f"      k range: {basalt_weathered.k.min():.2f} - {basalt_weathered.k.max():.2f} W/(m*K)")
    I_surface = basalt_weathered.thermal_inertia()[0] if len(basalt_weathered.depths) > 1 else basalt_weathered.thermal_inertia()
    print(f"      Thermal inertia (surface): {I_surface:.0f} J/(m^2*K*s^0.5)")
    print(f"      Source: {basalt_weathered.source_citation}")

    # Material 3: Lunar Regolith Analog
    # Based on Apollo mission data and laboratory measurements
    # Surface: fine dust, extremely low conductivity
    # Depth: consolidated regolith, increasing conductivity
    lunar_regolith = MaterialPropertiesDepth(
        material_id=str(uuid.uuid4()),
        name="Lunar Regolith Analog",
        version=1,
        source_database="Apollo mission data + NIST",
        source_citation="Cremers, C.J. (1975). Thermophysical properties of Apollo 14 fines. JGR, 80(32), 4466-4470.",
        notes="Depth profile based on Apollo core samples. Extreme low conductivity at surface due to vacuum conditions and fine particle size.",
        depths=np.array([0.0, 0.02, 0.05, 0.10, 0.30]),  # meters
        k=np.array([0.01, 0.015, 0.020, 0.025, 0.030]),  # W/(m*K) - very low!
        rho=np.array([1500, 1600, 1700, 1800, 1900]),  # kg/m^3
        cp=np.array([750, 760, 770, 780, 790]),  # J/(kg*K)
        alpha=0.85,  # High absorption (dark regolith)
        epsilon=0.92,  # High thermal emissivity
        roughness=0.005  # Fine powder surface [m]
    )
    mat_id = db.add_material(lunar_regolith)
    n_materials += 1
    print(f"\n  [{n_materials}] Added: {lunar_regolith.name}")
    print(f"      Depths: {lunar_regolith.depths}")
    print(f"      k range: {lunar_regolith.k.min():.3f} - {lunar_regolith.k.max():.3f} W/(m*K)")
    I_surface = lunar_regolith.thermal_inertia()[0] if len(lunar_regolith.depths) > 1 else lunar_regolith.thermal_inertia()
    print(f"      Thermal inertia (surface): {I_surface:.0f} J/(m^2*K*s^0.5)")
    print(f"      Source: {lunar_regolith.source_citation}")

    print(f"\n[OK] Added {n_materials - len(legacy_db.materials)} depth-varying materials\n")

    # =========================================================================
    # Summary
    # =========================================================================
    print("="*70)
    print("DATABASE CREATION COMPLETE")
    print("="*70)
    print(f"\nTotal materials: {n_materials}")
    print(f"  - Legacy (uniform depth): {len(legacy_db.materials)}")
    print(f"  - Depth-varying: {n_materials - len(legacy_db.materials)}")
    print(f"\nDatabase file: {db_path.absolute()}")
    print(f"Database size: {db_path.stat().st_size / 1024:.1f} KB")

    # Print all materials
    print("\n" + "-"*70)
    print("Material Summary:")
    print("-"*70)
    db.print_summary()

    db.close()
    print("\n[OK] Database closed successfully\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create SQLite materials database with legacy and depth-varying materials"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/materials/materials.db',
        help='Output database path (default: data/materials/materials.db)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing database without prompting'
    )

    args = parser.parse_args()

    create_database(args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
