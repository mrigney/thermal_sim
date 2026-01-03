"""
SQLite-based materials database with depth-varying thermal properties.

This module provides a robust materials database supporting:
- Depth-varying thermal properties (k, rho, cp)
- Surface radiative properties (alpha, epsilon)
- Spectral emissivity data for sensor modeling
- Material versioning and provenance tracking
- Property uncertainty bounds (min/max ranges)

Design Philosophy (from ChatGPT materials discussion):
- Thermal inertia as organizing principle
- Immutable material records with UUIDs
- Explicit lineage tracking (supersedes)
- Source database citations for all properties
- Separation of thermal (depth-varying) vs radiative (surface) properties
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class MaterialPropertiesDepth:
    """
    Material properties with depth-varying thermal characteristics.

    This class represents a complete material with:
    - Depth-varying thermal properties (arrays)
    - Surface-only radiative properties (scalars)
    - Metadata and provenance tracking

    Attributes
    ----------
    material_id : str
        Unique identifier (UUID)
    name : str
        Material name (e.g., "Desert Sand (Depth-Varying)")
    version : int
        Version number for lineage tracking
    supersedes : str, optional
        material_id of previous version (None if original)
    created_at : str
        ISO 8601 timestamp of creation
    source_database : str
        Source citation (e.g., "NASA TPSX", "CINDAS TPMD")
    source_citation : str, optional
        Detailed citation or reference
    notes : str, optional
        Additional information

    Depth-varying thermal properties (arrays of equal length):
    depths : ndarray
        Depths [m] where properties are defined
    k : ndarray
        Thermal conductivity [W/(m·K)] at each depth
    rho : ndarray
        Density [kg/m³] at each depth
    cp : ndarray
        Specific heat [J/(kg·K)] at each depth

    Surface radiative properties (scalars):
    alpha : float
        Broadband solar absorptivity [0-1]
    epsilon : float
        Broadband thermal emissivity [0-1]
    roughness : float
        Surface roughness [m]

    Uncertainty bounds (optional):
    k_min, k_max : ndarray, optional
        Thermal conductivity uncertainty bounds
    rho_min, rho_max : ndarray, optional
        Density uncertainty bounds
    cp_min, cp_max : ndarray, optional
        Specific heat uncertainty bounds
    alpha_min, alpha_max : float, optional
        Absorptivity uncertainty bounds
    epsilon_min, epsilon_max : float, optional
        Emissivity uncertainty bounds
    """

    # Core metadata
    material_id: str
    name: str
    version: int = 1
    supersedes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source_database: str = "custom"
    source_citation: Optional[str] = None
    notes: Optional[str] = None

    # Depth-varying thermal properties (arrays)
    depths: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    k: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    rho: np.ndarray = field(default_factory=lambda: np.array([2000.0]))
    cp: np.ndarray = field(default_factory=lambda: np.array([800.0]))

    # Surface radiative properties (scalars)
    alpha: float = 0.5
    epsilon: float = 0.9
    roughness: float = 0.01

    # Uncertainty bounds (optional, arrays)
    k_min: Optional[np.ndarray] = None
    k_max: Optional[np.ndarray] = None
    rho_min: Optional[np.ndarray] = None
    rho_max: Optional[np.ndarray] = None
    cp_min: Optional[np.ndarray] = None
    cp_max: Optional[np.ndarray] = None

    # Radiative uncertainty bounds (scalars)
    alpha_min: Optional[float] = None
    alpha_max: Optional[float] = None
    epsilon_min: Optional[float] = None
    epsilon_max: Optional[float] = None
    roughness_min: Optional[float] = None
    roughness_max: Optional[float] = None

    def __post_init__(self):
        """Validate array shapes and sort by depth."""
        # Convert to numpy arrays if needed
        if not isinstance(self.depths, np.ndarray):
            self.depths = np.array(self.depths)
        if not isinstance(self.k, np.ndarray):
            self.k = np.array(self.k)
        if not isinstance(self.rho, np.ndarray):
            self.rho = np.array(self.rho)
        if not isinstance(self.cp, np.ndarray):
            self.cp = np.array(self.cp)

        # Check array lengths match
        n = len(self.depths)
        if len(self.k) != n or len(self.rho) != n or len(self.cp) != n:
            raise ValueError(
                f"Thermal property array lengths must match depths array length ({n})"
            )

        # Sort by depth (ascending)
        sort_idx = np.argsort(self.depths)
        self.depths = self.depths[sort_idx]
        self.k = self.k[sort_idx]
        self.rho = self.rho[sort_idx]
        self.cp = self.cp[sort_idx]

        if self.k_min is not None:
            self.k_min = np.array(self.k_min)[sort_idx]
        if self.k_max is not None:
            self.k_max = np.array(self.k_max)[sort_idx]
        if self.rho_min is not None:
            self.rho_min = np.array(self.rho_min)[sort_idx]
        if self.rho_max is not None:
            self.rho_max = np.array(self.rho_max)[sort_idx]
        if self.cp_min is not None:
            self.cp_min = np.array(self.cp_min)[sort_idx]
        if self.cp_max is not None:
            self.cp_max = np.array(self.cp_max)[sort_idx]

    def thermal_diffusivity(self) -> np.ndarray:
        """Compute thermal diffusivity α = k/(ρ·cp) [m²/s] at each depth."""
        return self.k / (self.rho * self.cp)

    def thermal_inertia(self) -> np.ndarray:
        """Compute thermal inertia sqrt(k·ρ·cp) [J/(m²·K·s^(1/2))] at each depth."""
        return np.sqrt(self.k * self.rho * self.cp)

    def interpolate_to_depths(self, target_depths: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Interpolate thermal properties to target depths.

        Uses linear interpolation with constant extrapolation:
        - Above shallowest data: use surface value
        - Below deepest data: use deepest value

        Parameters
        ----------
        target_depths : ndarray
            Depths [m] where properties are needed

        Returns
        -------
        properties : dict
            Dictionary with keys 'k', 'rho', 'cp' containing interpolated arrays
        """
        k_interp = np.interp(target_depths, self.depths, self.k)
        rho_interp = np.interp(target_depths, self.depths, self.rho)
        cp_interp = np.interp(target_depths, self.depths, self.cp)

        return {
            'k': k_interp,
            'rho': rho_interp,
            'cp': cp_interp
        }


# =============================================================================
# SQLITE DATABASE
# =============================================================================

class MaterialDatabaseSQLite:
    """
    SQLite-based materials database with depth-varying properties.

    This database stores materials with:
    - Immutable material records (UUID-based)
    - Depth-varying thermal properties
    - Surface radiative properties
    - Spectral emissivity data
    - Provenance tracking and versioning

    Database Schema
    ---------------
    - materials: Core metadata and versioning
    - thermal_properties: Depth-varying k, rho, cp
    - radiative_properties: Surface alpha, epsilon
    - surface_properties: Roughness
    - spectral_emissivity: Wavelength-dependent emissivity

    Usage
    -----
    >>> db = MaterialDatabaseSQLite("materials.db")
    >>> mat = MaterialPropertiesDepth(...)
    >>> db.add_material(mat)
    >>> retrieved = db.get_material_by_name("Desert Sand")
    >>> interp = retrieved.interpolate_to_depths(target_depths)
    """

    def __init__(self, db_path: str, create_if_missing: bool = True):
        """
        Initialize database connection.

        Parameters
        ----------
        db_path : str
            Path to SQLite database file
        create_if_missing : bool
            If True, create database with schema if it doesn't exist
        """
        self.db_path = db_path
        self.conn = None

        # Create database directory if needed
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        # Connect and create schema if needed
        self._connect()

        if create_if_missing and not self._schema_exists():
            self._create_schema()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")

    def _schema_exists(self) -> bool:
        """Check if database schema exists."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='materials'"
        )
        return cursor.fetchone() is not None

    def _create_schema(self):
        """Create database schema."""
        cursor = self.conn.cursor()

        # Materials table (core metadata)
        cursor.execute("""
            CREATE TABLE materials (
                material_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                supersedes TEXT,
                created_at TEXT NOT NULL,
                source_database TEXT,
                source_citation TEXT,
                notes TEXT,
                FOREIGN KEY (supersedes) REFERENCES materials(material_id),
                UNIQUE(name, version)
            )
        """)

        # Thermal properties (depth-varying)
        cursor.execute("""
            CREATE TABLE thermal_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_id TEXT NOT NULL,
                depth_m REAL NOT NULL,
                k_thermal REAL NOT NULL,
                rho REAL NOT NULL,
                cp REAL NOT NULL,
                k_min REAL,
                k_max REAL,
                rho_min REAL,
                rho_max REAL,
                cp_min REAL,
                cp_max REAL,
                FOREIGN KEY (material_id) REFERENCES materials(material_id),
                UNIQUE(material_id, depth_m)
            )
        """)
        cursor.execute("""
            CREATE INDEX idx_thermal_material ON thermal_properties(material_id)
        """)

        # Radiative properties (surface-only, broadband)
        cursor.execute("""
            CREATE TABLE radiative_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_id TEXT NOT NULL,
                alpha_solar REAL NOT NULL,
                epsilon_thermal REAL NOT NULL,
                alpha_min REAL,
                alpha_max REAL,
                epsilon_min REAL,
                epsilon_max REAL,
                FOREIGN KEY (material_id) REFERENCES materials(material_id),
                UNIQUE(material_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX idx_radiative_material ON radiative_properties(material_id)
        """)

        # Surface properties
        cursor.execute("""
            CREATE TABLE surface_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_id TEXT NOT NULL,
                roughness_m REAL NOT NULL,
                roughness_min REAL,
                roughness_max REAL,
                FOREIGN KEY (material_id) REFERENCES materials(material_id),
                UNIQUE(material_id)
            )
        """)

        # Spectral emissivity (wavelength-dependent)
        cursor.execute("""
            CREATE TABLE spectral_emissivity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_id TEXT NOT NULL,
                wavelength_um REAL NOT NULL,
                emissivity REAL NOT NULL,
                temperature_k REAL,
                FOREIGN KEY (material_id) REFERENCES materials(material_id),
                UNIQUE(material_id, wavelength_um, temperature_k)
            )
        """)
        cursor.execute("""
            CREATE INDEX idx_spectral_material ON spectral_emissivity(material_id)
        """)

        self.conn.commit()

    def add_material(self, mat: MaterialPropertiesDepth) -> str:
        """
        Add material to database.

        Parameters
        ----------
        mat : MaterialPropertiesDepth
            Material to add

        Returns
        -------
        material_id : str
            UUID of added material

        Raises
        ------
        ValueError
            If material with same name and version already exists
        """
        cursor = self.conn.cursor()

        # Check for duplicate name+version
        cursor.execute(
            "SELECT material_id FROM materials WHERE name=? AND version=?",
            (mat.name, mat.version)
        )
        if cursor.fetchone() is not None:
            raise ValueError(
                f"Material '{mat.name}' version {mat.version} already exists"
            )

        # Insert into materials table
        cursor.execute("""
            INSERT INTO materials (
                material_id, name, version, supersedes, created_at,
                source_database, source_citation, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mat.material_id, mat.name, mat.version, mat.supersedes,
            mat.created_at, mat.source_database, mat.source_citation, mat.notes
        ))

        # Insert thermal properties (one row per depth)
        for i in range(len(mat.depths)):
            cursor.execute("""
                INSERT INTO thermal_properties (
                    material_id, depth_m, k_thermal, rho, cp,
                    k_min, k_max, rho_min, rho_max, cp_min, cp_max
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mat.material_id, float(mat.depths[i]),
                float(mat.k[i]), float(mat.rho[i]), float(mat.cp[i]),
                float(mat.k_min[i]) if mat.k_min is not None else None,
                float(mat.k_max[i]) if mat.k_max is not None else None,
                float(mat.rho_min[i]) if mat.rho_min is not None else None,
                float(mat.rho_max[i]) if mat.rho_max is not None else None,
                float(mat.cp_min[i]) if mat.cp_min is not None else None,
                float(mat.cp_max[i]) if mat.cp_max is not None else None
            ))

        # Insert radiative properties
        cursor.execute("""
            INSERT INTO radiative_properties (
                material_id, alpha_solar, epsilon_thermal,
                alpha_min, alpha_max, epsilon_min, epsilon_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            mat.material_id, mat.alpha, mat.epsilon,
            mat.alpha_min, mat.alpha_max, mat.epsilon_min, mat.epsilon_max
        ))

        # Insert surface properties
        cursor.execute("""
            INSERT INTO surface_properties (
                material_id, roughness_m, roughness_min, roughness_max
            ) VALUES (?, ?, ?, ?)
        """, (
            mat.material_id, mat.roughness, mat.roughness_min, mat.roughness_max
        ))

        self.conn.commit()
        return mat.material_id

    def get_material(self, material_id: str) -> Optional[MaterialPropertiesDepth]:
        """
        Retrieve material by ID.

        Parameters
        ----------
        material_id : str
            Material UUID

        Returns
        -------
        material : MaterialPropertiesDepth or None
            Material if found, None otherwise
        """
        cursor = self.conn.cursor()

        # Get core metadata
        cursor.execute("""
            SELECT material_id, name, version, supersedes, created_at,
                   source_database, source_citation, notes
            FROM materials
            WHERE material_id = ?
        """, (material_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        # Get thermal properties (multiple rows, ordered by depth)
        cursor.execute("""
            SELECT depth_m, k_thermal, rho, cp,
                   k_min, k_max, rho_min, rho_max, cp_min, cp_max
            FROM thermal_properties
            WHERE material_id = ?
            ORDER BY depth_m
        """, (material_id,))

        thermal_rows = cursor.fetchall()
        if not thermal_rows:
            raise ValueError(f"Material {material_id} has no thermal properties")

        depths = np.array([r['depth_m'] for r in thermal_rows])
        k = np.array([r['k_thermal'] for r in thermal_rows])
        rho = np.array([r['rho'] for r in thermal_rows])
        cp = np.array([r['cp'] for r in thermal_rows])

        # Check for uncertainty bounds
        k_min = np.array([r['k_min'] for r in thermal_rows])
        k_max = np.array([r['k_max'] for r in thermal_rows])
        rho_min = np.array([r['rho_min'] for r in thermal_rows])
        rho_max = np.array([r['rho_max'] for r in thermal_rows])
        cp_min = np.array([r['cp_min'] for r in thermal_rows])
        cp_max = np.array([r['cp_max'] for r in thermal_rows])

        # Convert None arrays to None
        k_min = k_min if not np.all(k_min == None) else None
        k_max = k_max if not np.all(k_max == None) else None
        rho_min = rho_min if not np.all(rho_min == None) else None
        rho_max = rho_max if not np.all(rho_max == None) else None
        cp_min = cp_min if not np.all(cp_min == None) else None
        cp_max = cp_max if not np.all(cp_max == None) else None

        # Get radiative properties
        cursor.execute("""
            SELECT alpha_solar, epsilon_thermal,
                   alpha_min, alpha_max, epsilon_min, epsilon_max
            FROM radiative_properties
            WHERE material_id = ?
        """, (material_id,))

        rad_row = cursor.fetchone()
        if rad_row is None:
            raise ValueError(f"Material {material_id} has no radiative properties")

        # Get surface properties
        cursor.execute("""
            SELECT roughness_m, roughness_min, roughness_max
            FROM surface_properties
            WHERE material_id = ?
        """, (material_id,))

        surf_row = cursor.fetchone()
        if surf_row is None:
            raise ValueError(f"Material {material_id} has no surface properties")

        # Construct MaterialPropertiesDepth
        return MaterialPropertiesDepth(
            material_id=row['material_id'],
            name=row['name'],
            version=row['version'],
            supersedes=row['supersedes'],
            created_at=row['created_at'],
            source_database=row['source_database'],
            source_citation=row['source_citation'],
            notes=row['notes'],
            depths=depths,
            k=k,
            rho=rho,
            cp=cp,
            k_min=k_min,
            k_max=k_max,
            rho_min=rho_min,
            rho_max=rho_max,
            cp_min=cp_min,
            cp_max=cp_max,
            alpha=rad_row['alpha_solar'],
            epsilon=rad_row['epsilon_thermal'],
            alpha_min=rad_row['alpha_min'],
            alpha_max=rad_row['alpha_max'],
            epsilon_min=rad_row['epsilon_min'],
            epsilon_max=rad_row['epsilon_max'],
            roughness=surf_row['roughness_m'],
            roughness_min=surf_row['roughness_min'],
            roughness_max=surf_row['roughness_max']
        )

    def get_material_by_name(self, name: str, version: Optional[int] = None) -> Optional[MaterialPropertiesDepth]:
        """
        Retrieve material by name.

        Parameters
        ----------
        name : str
            Material name (case-insensitive)
        version : int, optional
            Specific version to retrieve. If None, get latest version.

        Returns
        -------
        material : MaterialPropertiesDepth or None
            Material if found, None otherwise
        """
        cursor = self.conn.cursor()

        if version is not None:
            # Get specific version
            cursor.execute("""
                SELECT material_id FROM materials
                WHERE LOWER(name) = LOWER(?) AND version = ?
            """, (name, version))
        else:
            # Get latest version
            cursor.execute("""
                SELECT material_id FROM materials
                WHERE LOWER(name) = LOWER(?)
                ORDER BY version DESC
                LIMIT 1
            """, (name,))

        row = cursor.fetchone()
        if row is None:
            return None

        return self.get_material(row['material_id'])

    def list_materials(self) -> List[Tuple[str, str, int]]:
        """
        List all materials in database.

        Returns
        -------
        materials : list of (material_id, name, version) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT material_id, name, version
            FROM materials
            ORDER BY name, version
        """)
        return [(row['material_id'], row['name'], row['version'])
                for row in cursor.fetchall()]

    def add_spectral_emissivity(self, material_id: str, wavelengths_um: np.ndarray,
                                emissivity: np.ndarray, temperature_k: Optional[float] = None):
        """
        Add spectral emissivity data for a material.

        Parameters
        ----------
        material_id : str
            Material UUID
        wavelengths_um : ndarray
            Wavelengths [μm]
        emissivity : ndarray
            Emissivity at each wavelength
        temperature_k : float, optional
            Temperature [K] for temperature-dependent emissivity
        """
        cursor = self.conn.cursor()

        for wl, em in zip(wavelengths_um, emissivity):
            cursor.execute("""
                INSERT INTO spectral_emissivity (
                    material_id, wavelength_um, emissivity, temperature_k
                ) VALUES (?, ?, ?, ?)
            """, (material_id, float(wl), float(em), temperature_k))

        self.conn.commit()

    def get_spectral_emissivity(self, material_id: str,
                                temperature_k: Optional[float] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get spectral emissivity data for a material.

        Parameters
        ----------
        material_id : str
            Material UUID
        temperature_k : float, optional
            Temperature [K]. If None, get data for temperature_k=NULL

        Returns
        -------
        wavelengths, emissivity : tuple of ndarrays or None
            Wavelengths [μm] and emissivity values, or None if no data
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT wavelength_um, emissivity
            FROM spectral_emissivity
            WHERE material_id = ? AND temperature_k IS ?
            ORDER BY wavelength_um
        """, (material_id, temperature_k))

        rows = cursor.fetchall()
        if not rows:
            return None

        wavelengths = np.array([r['wavelength_um'] for r in rows])
        emissivity = np.array([r['emissivity'] for r in rows])

        return wavelengths, emissivity

    def migrate_from_json(self, json_path: str, source_name: str = "legacy_json"):
        """
        Migrate materials from legacy JSON format to SQLite.

        Each JSON material becomes a single-depth entry (depth=0.0) with
        version=1 and source_database=source_name.

        Parameters
        ----------
        json_path : str
            Path to JSON materials file
        source_name : str
            Source database name for migrated materials
        """
        # Import here to avoid circular dependency
        from src.materials import MaterialDatabase

        # Load legacy JSON
        legacy_db = MaterialDatabase()
        legacy_db.load_from_json(json_path)

        # Convert each material
        for mat in legacy_db.materials.values():
            # Create new material with single depth point
            new_mat = MaterialPropertiesDepth(
                material_id=str(uuid.uuid4()),
                name=mat.name,
                version=1,
                supersedes=None,
                source_database=source_name,
                notes=f"Migrated from {json_path}",
                depths=np.array([0.0]),
                k=np.array([mat.k]),
                rho=np.array([mat.rho]),
                cp=np.array([mat.cp]),
                alpha=mat.alpha,
                epsilon=mat.epsilon,
                roughness=mat.roughness
            )

            self.add_material(new_mat)

    def print_summary(self):
        """Print database summary."""
        materials = self.list_materials()

        print(f"\nMaterials Database: {self.db_path}")
        print(f"Total materials: {len(materials)}\n")

        if materials:
            print(f"{'Name':<40} {'Version':<10} {'ID':<40}")
            print("-" * 90)
            for mat_id, name, version in materials:
                print(f"{name:<40} {version:<10} {mat_id:<40}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
