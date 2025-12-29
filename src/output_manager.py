"""
Output Management Module for Thermal Terrain Simulator

Handles saving of simulation results for both terrain and objects with
separate file structures for modularity and external tool compatibility.
"""

import numpy as np
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict


class OutputManager:
    """
    Manages output files for terrain and object thermal simulations.

    Creates organized directory structure:
    output_dir/
    ├── terrain/
    │   ├── surface_temperature_NNNN.npy
    │   └── subsurface_temperature_NNNN.npy
    ├── objects/
    │   ├── object_name_1/
    │   │   ├── geometry.obj
    │   │   ├── face_temperature_NNNN.npy
    │   │   ├── subsurface_temperature_NNNN.npy
    │   │   └── face_solar_flux_NNNN.npy
    │   └── metadata.json
    ├── diagnostics/
    │   └── energy_balance.csv
    ├── config.yaml
    └── simulation_info.json
    """

    def __init__(self, output_dir: str, config: Any):
        """
        Initialize output manager.

        Parameters
        ----------
        output_dir : str
            Base output directory
        config : ThermalSimConfig
            Simulation configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config

        # Create directory structure
        self.terrain_dir = self.output_dir / "terrain"
        self.objects_dir = self.output_dir / "objects"
        self.diagnostics_dir = self.output_dir / "diagnostics"

        self.terrain_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        # Initialize counters
        self.save_count = 0
        self.object_dirs = {}  # Maps object name to directory path

        # Initialize diagnostics file
        self.energy_balance_file = self.diagnostics_dir / "energy_balance.csv"
        self._init_energy_balance_file()

        print(f"Output manager initialized:")
        print(f"  Output directory: {self.output_dir.absolute()}")
        print(f"  Terrain outputs: {self.terrain_dir.name}/")
        print(f"  Object outputs: {self.objects_dir.name}/")

    def _init_energy_balance_file(self):
        """Initialize energy balance CSV file with header."""
        with open(self.energy_balance_file, 'w') as f:
            f.write("time,elapsed_seconds,")
            f.write("terrain_solar_in,terrain_longwave_out,terrain_longwave_in,")
            f.write("terrain_sensible,terrain_latent,terrain_conduction,")
            f.write("terrain_net_energy,terrain_mean_temp,terrain_min_temp,terrain_max_temp")
            # Object energy terms will be added dynamically
            f.write("\n")

    def setup_objects(self, objects: List[Any]):
        """
        Set up output directories for objects and copy geometry files.

        Parameters
        ----------
        objects : List[ThermalObject]
            List of thermal objects
        """
        if not objects:
            return

        # Create metadata for all objects
        metadata = {
            "n_objects": len(objects),
            "objects": []
        }

        for obj in objects:
            # Create object-specific directory
            obj_dir = self.objects_dir / obj.name
            obj_dir.mkdir(parents=True, exist_ok=True)
            self.object_dirs[obj.name] = obj_dir

            # Copy mesh file to object directory
            # Find the original mesh file
            mesh_path = Path(obj.name + ".obj")  # This will be updated
            # For now, we'll note the mesh file path in metadata

            # Add to metadata
            obj_metadata = {
                "name": obj.name,
                "n_faces": obj.faces.shape[0],
                "n_vertices": obj.vertices.shape[0],
                "location": obj.location.tolist(),
                "rotation": obj.rotation.tolist() if obj.rotation is not None else None,
                "material": obj.material.name,
                "thickness": obj.thickness,
                "bounds": {
                    "x_min": float(obj.vertices[:, 0].min()),
                    "x_max": float(obj.vertices[:, 0].max()),
                    "y_min": float(obj.vertices[:, 1].min()),
                    "y_max": float(obj.vertices[:, 1].max()),
                    "z_min": float(obj.vertices[:, 2].min()),
                    "z_max": float(obj.vertices[:, 2].max())
                }
            }
            metadata["objects"].append(obj_metadata)

            # Save geometry to OBJ file
            self._save_obj_geometry(obj, obj_dir / "geometry.obj")

        # Save metadata
        with open(self.objects_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Set up output for {len(objects)} object(s)")

    def _save_obj_geometry(self, obj: Any, filepath: Path):
        """Save object geometry in OBJ format."""
        with open(filepath, 'w') as f:
            f.write(f"# Thermal object: {obj.name}\n")
            f.write(f"# Generated by Thermal Terrain Simulator\n")
            f.write(f"# Date: {datetime.now().isoformat()}\n\n")

            # Write vertices
            f.write(f"# {obj.vertices.shape[0]} vertices\n")
            for v in obj.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            # Write normals
            f.write(f"# {obj.normals.shape[0]} normals\n")
            for n in obj.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            f.write("\n")

            # Write faces (OBJ uses 1-indexed vertices)
            f.write(f"# {obj.faces.shape[0]} faces\n")
            for face in obj.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def save_timestep(self, time: datetime, elapsed: float,
                     terrain_T_surface: np.ndarray,
                     terrain_T_subsurface: Optional[np.ndarray] = None,
                     objects: Optional[List[Any]] = None,
                     diagnostics: Optional[Dict[str, float]] = None):
        """
        Save simulation state at a timestep.

        Parameters
        ----------
        time : datetime
            Current simulation time
        elapsed : float
            Elapsed time in seconds
        terrain_T_surface : ndarray, shape (ny, nx)
            Terrain surface temperature [K]
        terrain_T_subsurface : ndarray, shape (ny, nx, nz), optional
            Terrain subsurface temperatures [K]
        objects : List[ThermalObject], optional
            List of thermal objects with current state
        diagnostics : dict, optional
            Energy balance diagnostics
        """
        # Format output number with leading zeros
        num_str = f"{self.save_count:04d}"

        # Save terrain data
        np.save(self.terrain_dir / f"surface_temperature_{num_str}.npy",
                terrain_T_surface.astype(np.float32))

        if terrain_T_subsurface is not None:
            np.save(self.terrain_dir / f"subsurface_temperature_{num_str}.npy",
                    terrain_T_subsurface.astype(np.float32))

        # Save object data
        if objects:
            for obj in objects:
                obj_dir = self.object_dirs[obj.name]

                # Face surface temperatures
                np.save(obj_dir / f"face_temperature_{num_str}.npy",
                        obj.T_surface.astype(np.float32))

                # Subsurface temperatures (if computed)
                if obj.T_subsurface is not None:
                    np.save(obj_dir / f"subsurface_temperature_{num_str}.npy",
                            obj.T_subsurface.astype(np.float32))

                # Solar flux per face
                np.save(obj_dir / f"face_solar_flux_{num_str}.npy",
                        obj.solar_flux.astype(np.float32))

                # Shadow fractions
                np.save(obj_dir / f"face_shadow_fraction_{num_str}.npy",
                        obj.shadow_fraction.astype(np.float32))

        # Save diagnostics
        if diagnostics:
            self._append_diagnostics(time, elapsed, diagnostics)

        self.save_count += 1

        print(f"Saved timestep {self.save_count}: {time.isoformat()}, "
              f"Terrain T: [{terrain_T_surface.min():.2f}, {terrain_T_surface.max():.2f}] K")

    def _append_diagnostics(self, time: datetime, elapsed: float, diagnostics: Dict[str, float]):
        """Append energy balance diagnostics to CSV."""
        with open(self.energy_balance_file, 'a') as f:
            # Time info
            f.write(f"{time.isoformat()},{elapsed:.1f},")

            # Terrain energy terms
            f.write(f"{diagnostics.get('terrain_solar_in', 0.0):.3e},")
            f.write(f"{diagnostics.get('terrain_longwave_out', 0.0):.3e},")
            f.write(f"{diagnostics.get('terrain_longwave_in', 0.0):.3e},")
            f.write(f"{diagnostics.get('terrain_sensible', 0.0):.3e},")
            f.write(f"{diagnostics.get('terrain_latent', 0.0):.3e},")
            f.write(f"{diagnostics.get('terrain_conduction', 0.0):.3e},")
            f.write(f"{diagnostics.get('terrain_net_energy', 0.0):.3e},")
            f.write(f"{diagnostics.get('terrain_mean_temp', 0.0):.2f},")
            f.write(f"{diagnostics.get('terrain_min_temp', 0.0):.2f},")
            f.write(f"{diagnostics.get('terrain_max_temp', 0.0):.2f}")

            # Object energy terms (to be implemented)
            # TODO: Add per-object energy balance terms

            f.write("\n")

    def save_config(self):
        """Save copy of configuration to output directory."""
        import yaml

        # Convert config to dict (this is simplified - actual implementation
        # would use the config's to_dict method if available)
        config_dict = {
            "simulation": {
                "name": self.config.simulation.name,
                "start_time": self.config.simulation.start_time.isoformat(),
                "duration_seconds": self.config.simulation.duration_seconds,
                "time_step": self.config.simulation.dt
            },
            "output": {
                "directory": str(self.output_dir),
                "save_interval_seconds": self.config.output.save_interval_seconds
            }
            # Add more config sections as needed
        }

        with open(self.output_dir / "config.yaml", 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def save_simulation_info(self, info: Dict[str, Any]):
        """
        Save simulation metadata.

        Parameters
        ----------
        info : dict
            Simulation information (runtime, completion status, etc.)
        """
        with open(self.output_dir / "simulation_info.json", 'w') as f:
            json.dump(info, f, indent=2, default=str)

    def finalize(self, runtime: float, completed: bool = True):
        """
        Finalize outputs and save summary.

        Parameters
        ----------
        runtime : float
            Total simulation runtime in seconds
        completed : bool
            Whether simulation completed successfully
        """
        summary = {
            "completed": completed,
            "runtime_seconds": runtime,
            "n_timesteps_saved": self.save_count,
            "output_directory": str(self.output_dir.absolute()),
            "terrain_outputs": str(self.terrain_dir),
            "objects_outputs": str(self.objects_dir) if self.object_dirs else None,
            "n_objects": len(self.object_dirs),
            "end_time": datetime.now().isoformat()
        }

        self.save_simulation_info(summary)

        print(f"\nSimulation output summary:")
        print(f"  Status: {'Completed' if completed else 'Incomplete'}")
        print(f"  Runtime: {runtime:.1f} seconds")
        print(f"  Timesteps saved: {self.save_count}")
        print(f"  Output directory: {self.output_dir.absolute()}")
