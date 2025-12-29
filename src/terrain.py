"""
Terrain module for thermal simulation.

Handles terrain geometry, derived properties (normals, slopes, aspects),
and sky view factor computation.
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class TerrainGrid:
    """
    Manages 2D terrain grid with geometry and derived properties.
    
    Attributes:
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        dx (float): Grid spacing in x direction [m]
        dy (float): Grid spacing in y direction [m]
        elevation (ndarray): Elevation at each grid point [m], shape (ny, nx)
        normals (ndarray): Unit normal vectors at each point, shape (ny, nx, 3)
        slope (ndarray): Slope angle at each point [radians], shape (ny, nx)
        aspect (ndarray): Aspect angle at each point [radians], shape (ny, nx)
        sky_view_factor (ndarray): Sky view factor at each point [0-1], shape (ny, nx)
        material_class (ndarray): Material class ID at each point, shape (ny, nx)
    """
    
    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        """
        Initialize terrain grid.
        
        Args:
            nx: Number of points in x direction
            ny: Number of points in y direction
            dx: Grid spacing in x [m]
            dy: Grid spacing in y [m]
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        
        # Terrain geometry
        self.elevation = np.zeros((ny, nx), dtype=np.float32)
        self.normals = np.zeros((ny, nx, 3), dtype=np.float32)
        self.slope = np.zeros((ny, nx), dtype=np.float32)
        self.aspect = np.zeros((ny, nx), dtype=np.float32)
        
        # View factors
        self.sky_view_factor = np.ones((ny, nx), dtype=np.float32)  # Default to 1 (flat terrain)
        
        # Material classification
        self.material_class = np.zeros((ny, nx), dtype=np.int32)
        
    def set_elevation(self, elevation: np.ndarray):
        """
        Set elevation data and compute derived quantities.
        
        Args:
            elevation: Elevation array, shape (ny, nx)
        """
        if elevation.shape != (self.ny, self.nx):
            raise ValueError(f"Elevation shape {elevation.shape} doesn't match grid size ({self.ny}, {self.nx})")
        
        self.elevation = elevation.astype(np.float32)
        self.compute_normals()
        self.compute_slope_aspect()
        
    def compute_normals(self):
        """
        Compute surface normal vectors using central differences.
        
        Uses finite differences to compute gradient, then constructs normal vectors.
        Boundaries use one-sided differences.
        """
        # Allocate gradient arrays
        dz_dx = np.zeros_like(self.elevation)
        dz_dy = np.zeros_like(self.elevation)
        
        # Central differences in interior
        dz_dx[:, 1:-1] = (self.elevation[:, 2:] - self.elevation[:, :-2]) / (2 * self.dx)
        dz_dy[1:-1, :] = (self.elevation[2:, :] - self.elevation[:-2, :]) / (2 * self.dy)
        
        # One-sided differences at boundaries
        dz_dx[:, 0] = (self.elevation[:, 1] - self.elevation[:, 0]) / self.dx
        dz_dx[:, -1] = (self.elevation[:, -1] - self.elevation[:, -2]) / self.dx
        dz_dy[0, :] = (self.elevation[1, :] - self.elevation[0, :]) / self.dy
        dz_dy[-1, :] = (self.elevation[-1, :] - self.elevation[-2, :]) / self.dy
        
        # Construct normal vectors: N = (-dz/dx, -dz/dy, 1)
        self.normals[:, :, 0] = -dz_dx
        self.normals[:, :, 1] = -dz_dy
        self.normals[:, :, 2] = 1.0
        
        # Normalize
        norm = np.sqrt(np.sum(self.normals**2, axis=2, keepdims=True))
        self.normals /= norm
        
    def compute_slope_aspect(self):
        """
        Compute slope and aspect from normal vectors.
        
        Slope: angle from horizontal [0, π/2] radians
        Aspect: azimuth angle of steepest descent [0, 2π] radians
                0 = North, π/2 = East, π = South, 3π/2 = West
        """
        # Slope: angle between normal and vertical (0,0,1)
        # cos(slope) = n_z, so slope = arccos(n_z)
        self.slope = np.arccos(np.clip(self.normals[:, :, 2], -1.0, 1.0))
        
        # Aspect: azimuth of steepest descent
        # Steepest descent direction is (-dz/dx, -dz/dy)
        # But our normal is (-dz/dx, -dz/dy, 1), so use first two components
        dz_dx = -self.normals[:, :, 0] / self.normals[:, :, 2]
        dz_dy = -self.normals[:, :, 1] / self.normals[:, :, 2]
        
        # Aspect angle: atan2(dy, dx) but measured clockwise from North
        # Standard atan2 gives counterclockwise from East
        # Convert: aspect = π/2 - atan2(dz_dy, dz_dx) = atan2(dz_dx, dz_dy)
        self.aspect = np.arctan2(dz_dx, dz_dy)
        
        # Ensure aspect is in [0, 2π]
        self.aspect = np.mod(self.aspect, 2 * np.pi)
        
        # For flat areas (slope near 0), aspect is undefined - set to 0
        flat_mask = self.slope < 1e-6
        self.aspect[flat_mask] = 0.0
        
    def compute_sky_view_factor_simple(self):
        """
        Compute sky view factor using simple horizon angle method.
        
        This is a simplified version that samples horizon in discrete azimuthal directions.
        More sophisticated methods (ray tracing, hemispherical integration) can be added later.
        
        Sky view factor (SVF) represents the fraction of hemisphere visible to the sky
        (not blocked by terrain). SVF = 1 for completely open locations, < 1 for
        locations with terrain blocking parts of the sky.
        """
        n_azimuth = 36  # Sample every 10 degrees
        azimuths = np.linspace(0, 2*np.pi, n_azimuth, endpoint=False)
        
        # Maximum search distance (in grid cells)
        max_distance = min(100, min(self.nx, self.ny) // 2)
        
        # Initialize horizon angles (elevation angle to horizon in each direction)
        horizon_angles = np.zeros((self.ny, self.nx, n_azimuth), dtype=np.float32)
        
        print("Computing sky view factors...")
        for az_idx, azimuth in enumerate(azimuths):
            # Direction vector in grid coordinates
            dx_step = np.sin(azimuth)
            dy_step = np.cos(azimuth)
            
            for i in range(self.ny):
                for j in range(self.nx):
                    z0 = self.elevation[i, j]
                    max_angle = 0.0
                    
                    # March along azimuth direction
                    for step in range(1, max_distance):
                        # Position in grid
                        x_sample = j + dx_step * step
                        y_sample = i + dy_step * step
                        
                        # Check bounds
                        if x_sample < 0 or x_sample >= self.nx - 1:
                            break
                        if y_sample < 0 or y_sample >= self.ny - 1:
                            break
                        
                        # Bilinear interpolation of elevation
                        ix = int(x_sample)
                        iy = int(y_sample)
                        fx = x_sample - ix
                        fy = y_sample - iy
                        
                        z_sample = (
                            (1-fx)*(1-fy)*self.elevation[iy, ix] +
                            fx*(1-fy)*self.elevation[iy, ix+1] +
                            (1-fx)*fy*self.elevation[iy+1, ix] +
                            fx*fy*self.elevation[iy+1, ix+1]
                        )
                        
                        # Horizontal distance
                        horizontal_dist = step * np.sqrt((self.dx*dx_step)**2 + (self.dy*dy_step)**2)
                        
                        # Elevation angle
                        angle = np.arctan2(z_sample - z0, horizontal_dist)
                        max_angle = max(max_angle, angle)
                    
                    horizon_angles[i, j, az_idx] = max_angle
        
        # Compute SVF by integrating over azimuth
        # SVF = (1/2π) ∫ cos²(horizon_angle) dφ
        # Discrete approximation
        cos_squared = np.cos(horizon_angles)**2
        self.sky_view_factor = np.mean(cos_squared, axis=2)
        
        print(f"Sky view factor range: [{self.sky_view_factor.min():.3f}, {self.sky_view_factor.max():.3f}]")
        
    def set_material_class(self, material_class: np.ndarray):
        """
        Set material classification map.
        
        Args:
            material_class: Integer array of material class IDs, shape (ny, nx)
        """
        if material_class.shape != (self.ny, self.nx):
            raise ValueError(f"Material class shape {material_class.shape} doesn't match grid size ({self.ny}, {self.nx})")
        
        self.material_class = material_class.astype(np.int32)
        
    def get_info(self) -> str:
        """
        Get summary information about the terrain.
        
        Returns:
            String with terrain statistics
        """
        info = []
        info.append(f"Terrain Grid Information:")
        info.append(f"  Size: {self.nx} x {self.ny} points")
        info.append(f"  Spacing: {self.dx} m x {self.dy} m")
        info.append(f"  Physical extent: {self.nx*self.dx:.1f} m x {self.ny*self.dy:.1f} m")
        info.append(f"  Elevation range: [{self.elevation.min():.2f}, {self.elevation.max():.2f}] m")
        info.append(f"  Slope range: [{np.degrees(self.slope.min()):.1f}, {np.degrees(self.slope.max()):.1f}] degrees")
        info.append(f"  Sky view factor range: [{self.sky_view_factor.min():.3f}, {self.sky_view_factor.max():.3f}]")
        info.append(f"  Material classes: {np.unique(self.material_class)}")
        
        return "\n".join(info)


def create_synthetic_terrain(nx: int, ny: int, dx: float, dy: float, 
                             terrain_type: str = 'rolling_hills') -> TerrainGrid:
    """
    Create synthetic terrain for testing.
    
    Args:
        nx: Number of points in x
        ny: Number of points in y
        dx: Grid spacing in x [m]
        dy: Grid spacing in y [m]
        terrain_type: Type of terrain to generate
            - 'flat': Flat terrain
            - 'rolling_hills': Sinusoidal hills
            - 'ridge': Simple ridge
            - 'valley': Simple valley
    
    Returns:
        TerrainGrid object with synthetic elevation
    """
    terrain = TerrainGrid(nx, ny, dx, dy)
    
    # Create coordinate arrays
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    
    if terrain_type == 'flat':
        elevation = np.zeros((ny, nx))
        
    elif terrain_type == 'rolling_hills':
        # Sinusoidal hills with multiple wavelengths
        wavelength1 = 50.0  # meters
        wavelength2 = 30.0
        amplitude1 = 5.0  # meters
        amplitude2 = 2.0
        
        elevation = (amplitude1 * np.sin(2*np.pi*X/wavelength1) * np.cos(2*np.pi*Y/wavelength1) +
                    amplitude2 * np.sin(2*np.pi*X/wavelength2) * np.sin(2*np.pi*Y/wavelength2))
        
    elif terrain_type == 'ridge':
        # Ridge running along x direction
        ridge_width = 10.0  # meters
        ridge_height = 30.0  # meters
        y_center = ny * dy / 2
        
        elevation = ridge_height * np.exp(-((Y - y_center) / ridge_width)**2)
        
    elif terrain_type == 'valley':
        # Valley running along x direction
        valley_width = 30.0  # meters
        valley_depth = 8.0  # meters
        y_center = ny * dy / 2
        
        elevation = -valley_depth * np.exp(-((Y - y_center) / valley_width)**2)
        
    else:
        raise ValueError(f"Unknown terrain type: {terrain_type}")
    
    terrain.set_elevation(elevation)
    
    return terrain


if __name__ == "__main__":
    # Test the terrain module
    print("Testing terrain module...")
    
    # Create a simple test terrain
    terrain = create_synthetic_terrain(100, 100, 0.1, 0.1, terrain_type='rolling_hills')
    
    print("\n" + terrain.get_info())
    
    # Compute sky view factors (this will take a moment)
    terrain.compute_sky_view_factor_simple()
    
    print("\nTerrain module test complete!")
