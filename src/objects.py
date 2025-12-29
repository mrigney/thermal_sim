"""
3D Object Thermal Modeling Module

This module provides functionality for loading, placing, and thermally modeling
3D mesh objects (buildings, vehicles, equipment) on terrain. Objects are loaded
from OBJ Wavefront files and receive thermal solutions using a 1D solver per face.

Classes:
    ThermalObject: Represents a 3D thermal object with geometry and thermal state

Functions:
    load_obj_file: Parse OBJ Wavefront files
    compute_face_normal: Calculate face normal from vertices
    apply_rotation: Apply Euler angle rotation to vertices
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass


def load_obj_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load a 3D mesh from an OBJ Wavefront file.

    Parses vertices (v), faces (f), and optionally vertex normals (vn) from
    a standard OBJ file format. Handles both triangular and quad faces
    (quads are automatically triangulated).

    Parameters
    ----------
    filepath : str
        Path to the OBJ file

    Returns
    -------
    vertices : np.ndarray
        Vertex positions, shape (N, 3) where N is the number of vertices.
        Each row is [x, y, z] in object-local coordinates.
    faces : np.ndarray
        Face definitions, shape (M, 3) where M is the number of triangular faces.
        Each row contains 3 vertex indices (0-indexed).
    normals : np.ndarray or None
        Vertex normals if provided in file, shape (N_normals, 3), or None.
        Will be computed per-face if not provided.

    Raises
    ------
    FileNotFoundError
        If the OBJ file doesn't exist
    ValueError
        If the OBJ file format is invalid or contains no geometry

    Notes
    -----
    Supported OBJ elements:
    - v x y z : Vertex positions
    - vn x y z : Vertex normals (optional)
    - f v1 v2 v3 : Triangular faces (1-indexed)
    - f v1//vn1 v2//vn2 v3//vn3 : Faces with normals
    - f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 : Faces with textures and normals

    Quad faces (4 vertices) are automatically split into 2 triangles.
    Comments (#) and unsupported elements are ignored.

    Examples
    --------
    >>> vertices, faces, normals = load_obj_file('data/objects/cube.obj')
    >>> print(f"Loaded {len(vertices)} vertices, {len(faces)} faces")
    Loaded 8 vertices, 12 faces
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"OBJ file not found: {filepath}")

    vertices = []
    normals = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            # Parse vertices
            if parts[0] == 'v':
                if len(parts) < 4:
                    raise ValueError(f"Invalid vertex line: {line}")
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])

            # Parse vertex normals
            elif parts[0] == 'vn':
                if len(parts) < 4:
                    raise ValueError(f"Invalid normal line: {line}")
                nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                normals.append([nx, ny, nz])

            # Parse faces
            elif parts[0] == 'f':
                if len(parts) < 4:
                    raise ValueError(f"Invalid face line (need at least 3 vertices): {line}")

                # Extract vertex indices from face definition
                # Handles formats: "v", "v/vt", "v//vn", "v/vt/vn"
                vertex_indices = []
                for vertex_str in parts[1:]:
                    # Split by '/' and take first element (vertex index)
                    idx_str = vertex_str.split('/')[0]
                    idx = int(idx_str) - 1  # Convert to 0-indexed
                    vertex_indices.append(idx)

                # Triangulate if quad face (4 vertices)
                if len(vertex_indices) == 3:
                    faces.append(vertex_indices)
                elif len(vertex_indices) == 4:
                    # Split quad into two triangles: (0,1,2) and (0,2,3)
                    faces.append([vertex_indices[0], vertex_indices[1], vertex_indices[2]])
                    faces.append([vertex_indices[0], vertex_indices[2], vertex_indices[3]])
                else:
                    # For polygons with >4 vertices, do simple fan triangulation
                    for i in range(1, len(vertex_indices) - 1):
                        faces.append([vertex_indices[0], vertex_indices[i], vertex_indices[i+1]])

    # Convert to numpy arrays
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)
    normals = np.array(normals, dtype=np.float64) if normals else None

    # Validation
    if len(vertices) == 0:
        raise ValueError(f"No vertices found in OBJ file: {filepath}")
    if len(faces) == 0:
        raise ValueError(f"No faces found in OBJ file: {filepath}")

    # Check for out-of-bounds face indices
    max_idx = np.max(faces)
    if max_idx >= len(vertices):
        raise ValueError(f"Face references vertex {max_idx} but only {len(vertices)} vertices exist")

    return vertices, faces, normals


def compute_face_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the normal vector for a triangular face.

    Uses the cross product of two edge vectors. The normal points outward
    according to the right-hand rule (counter-clockwise vertex ordering).

    Parameters
    ----------
    v0, v1, v2 : np.ndarray
        The three vertices of the triangle, each shape (3,) as [x, y, z]

    Returns
    -------
    normal : np.ndarray
        Unit normal vector, shape (3,). If the triangle is degenerate
        (zero area), returns [0, 0, 0].

    Notes
    -----
    Normal direction follows right-hand rule:
    - Fingers curl from v0→v1→v2
    - Thumb points in normal direction
    """
    # Two edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Cross product gives normal (perpendicular to face)
    normal = np.cross(edge1, edge2)

    # Normalize to unit length
    length = np.linalg.norm(normal)
    if length > 1e-10:
        normal = normal / length
    else:
        # Degenerate triangle (zero area)
        normal = np.array([0.0, 0.0, 0.0])

    return normal


def apply_rotation(vertices: np.ndarray, rotation_deg: np.ndarray) -> np.ndarray:
    """
    Apply Euler angle rotation to vertices.

    Rotations are applied in order: Z (yaw) → Y (pitch) → X (roll).
    This is the standard aerospace rotation sequence.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions, shape (N, 3)
    rotation_deg : np.ndarray
        Euler angles in degrees, shape (3,) as [rx, ry, rz]
        rx: Rotation about X axis (roll)
        ry: Rotation about Y axis (pitch)
        rz: Rotation about Z axis (yaw)

    Returns
    -------
    rotated_vertices : np.ndarray
        Rotated vertex positions, shape (N, 3)

    Notes
    -----
    Rotation matrices are applied right-to-left: R = Rx * Ry * Rz
    This means rotations are applied in order: Z, then Y, then X.
    """
    # Convert to radians
    rx, ry, rz = np.deg2rad(rotation_deg)

    # Rotation matrix about X axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    # Rotation matrix about Y axis (pitch)
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Rotation matrix about Z axis (yaw)
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combined rotation: Rx * Ry * Rz (applied right-to-left)
    R = Rx @ Ry @ Rz

    # Apply rotation to all vertices
    rotated = vertices @ R.T

    return rotated


class ThermalObject:
    """
    Represents a 3D thermal object with geometry and thermal state.

    A ThermalObject encapsulates the geometric mesh, material properties,
    thermal state (surface and subsurface temperatures), and environmental
    forcing (solar flux, shadows, convection) for a 3D object placed on terrain.

    Each triangular face of the object receives an independent 1D thermal
    solution through the object's thickness, similar to the terrain solver approach.

    Attributes
    ----------
    name : str
        Human-readable identifier
    vertices : np.ndarray
        Vertex positions in world coordinates, shape (N_vertices, 3)
    faces : np.ndarray
        Face definitions (vertex indices), shape (N_faces, 3)
    normals : np.ndarray
        Face normal vectors (outward), shape (N_faces, 3)
    centroids : np.ndarray
        Face centroid positions, shape (N_faces, 3)
    areas : np.ndarray
        Face areas in m², shape (N_faces,)
    location : np.ndarray
        Translation vector [x, y, z] in terrain coordinates
    rotation : np.ndarray
        Euler angles [rx, ry, rz] in degrees
    material : MaterialProperties
        Material thermal and optical properties
    thickness : float
        Object wall/surface thickness in meters
    T_surface : np.ndarray
        Surface temperature per face in Kelvin, shape (N_faces,)
    T_subsurface : np.ndarray
        Subsurface temperatures, shape (N_faces, N_layers)
    solar_flux : np.ndarray
        Total solar flux per face in W/m², shape (N_faces,)
    shadow_fraction : np.ndarray
        Shadow fraction per face (0=sun, 1=shadow), shape (N_faces,)
    sky_view_factor : np.ndarray
        Sky visibility fraction per face, shape (N_faces,)
    """

    def __init__(self, name: str, mesh_file: str, location: np.ndarray,
                 material, thickness: float, rotation: Optional[np.ndarray] = None):
        """
        Initialize a ThermalObject from an OBJ mesh file.

        Parameters
        ----------
        name : str
            Human-readable identifier for this object
        mesh_file : str
            Path to OBJ file (absolute or relative to data/objects/)
        location : np.ndarray
            Translation vector [x, y, z] in terrain coordinates (meters)
        material : MaterialProperties
            Material thermal and optical properties
        thickness : float
            Object wall thickness in meters (for 1D thermal solver)
        rotation : np.ndarray, optional
            Euler angles [rx, ry, rz] in degrees. Default is [0, 0, 0].
        """
        self.name = name
        self.material = material
        self.thickness = thickness
        self.location = np.array(location, dtype=np.float64)
        self.rotation = np.array(rotation, dtype=np.float64) if rotation is not None else np.zeros(3)

        # Load mesh from OBJ file
        vertices_local, self.faces, normals_from_file = load_obj_file(mesh_file)

        # Apply rotation (if any) before translation
        if np.any(self.rotation != 0):
            vertices_rotated = apply_rotation(vertices_local, self.rotation)
        else:
            vertices_rotated = vertices_local

        # Apply translation to get world coordinates
        self.vertices = vertices_rotated + self.location

        # Compute face geometry
        self._compute_face_geometry()

        # Initialize thermal state (will be set by solver)
        n_faces = len(self.faces)
        self.T_surface = np.zeros(n_faces, dtype=np.float64)
        self.T_subsurface = None  # Will be initialized when subsurface grid is set

        # Initialize environmental forcing (computed each timestep)
        self.solar_flux = np.zeros(n_faces, dtype=np.float64)
        self.shadow_fraction = np.zeros(n_faces, dtype=np.float64)
        self.sky_view_factor = np.ones(n_faces, dtype=np.float64) * 0.5  # Hemisphere default

    def _compute_face_geometry(self):
        """
        Compute face normals, centroids, and areas.

        Called internally during initialization. Computes geometric properties
        for each triangular face based on current vertex positions.
        """
        n_faces = len(self.faces)
        self.normals = np.zeros((n_faces, 3), dtype=np.float64)
        self.centroids = np.zeros((n_faces, 3), dtype=np.float64)
        self.areas = np.zeros(n_faces, dtype=np.float64)

        for i, face in enumerate(self.faces):
            # Get the three vertices of this face
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]

            # Compute normal
            self.normals[i] = compute_face_normal(v0, v1, v2)

            # Compute centroid (average of three vertices)
            self.centroids[i] = (v0 + v1 + v2) / 3.0

            # Compute area (half the magnitude of cross product)
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            self.areas[i] = 0.5 * np.linalg.norm(cross)

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the axis-aligned bounding box of the object.

        Returns
        -------
        bbox_min : np.ndarray
            Minimum coordinates [x_min, y_min, z_min]
        bbox_max : np.ndarray
            Maximum coordinates [x_max, y_max, z_max]
        """
        bbox_min = np.min(self.vertices, axis=0)
        bbox_max = np.max(self.vertices, axis=0)
        return bbox_min, bbox_max

    def __repr__(self) -> str:
        """String representation of the object."""
        bbox_min, bbox_max = self.get_bounding_box()
        return (f"ThermalObject(name='{self.name}', "
                f"vertices={len(self.vertices)}, faces={len(self.faces)}, "
                f"location={self.location}, "
                f"bbox=[{bbox_min[0]:.1f},{bbox_min[1]:.1f},{bbox_min[2]:.1f}] to "
                f"[{bbox_max[0]:.1f},{bbox_max[1]:.1f},{bbox_max[2]:.1f}])")


if __name__ == "__main__":
    # Quick test of OBJ loading
    print("Testing OBJ loader...")

    test_file = Path(__file__).parent.parent / "data" / "objects" / "examples" / "cube_1m.obj"
    if test_file.exists():
        vertices, faces, normals = load_obj_file(str(test_file))
        print(f"Loaded {test_file.name}:")
        print(f"  Vertices: {len(vertices)}")
        print(f"  Faces: {len(faces)}")
        print(f"  Normals from file: {len(normals) if normals is not None else 0}")
        print(f"  First vertex: {vertices[0]}")
        print(f"  First face: {faces[0]}")
    else:
        print(f"Test file not found: {test_file}")
