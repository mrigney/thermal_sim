"""
Simulation runner module - orchestrates thermal simulation execution.

Handles:
- Setup of terrain, materials, atmosphere from configuration
- ThermalSolver initialization
- Main simulation loop with progress tracking
- Output saving and checkpointing

Author: Thermal Simulator Team
Date: December 2025
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import time
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.config import ThermalSimConfig
from src.terrain import TerrainGrid, create_synthetic_terrain
from src.materials import MaterialField, MaterialDatabase, MaterialProperties, create_representative_materials
from src.atmosphere import AtmosphericConditions, create_diurnal_temperature, create_diurnal_wind
from src.solar import ShadowCache
from src.solver import ThermalSolver, SubsurfaceGrid, TemperatureField
from src.visualization_enhanced import (
    plot_terrain_elevation,
    plot_hourly_summary,
    plot_flux_timeseries
)


class SimulationRunner:
    """
    Orchestrates thermal simulation execution from configuration.

    Handles initialization, running, and output saving.
    """

    def __init__(self, config: ThermalSimConfig, verbose: bool = False):
        """
        Initialize simulation from configuration.

        Parameters
        ----------
        config : ThermalSimConfig
            Complete validated configuration
        verbose : bool, optional
            Enable verbose progress logging (default False)
        """
        self.config = config
        self.verbose = verbose

        # Setup components
        self._setup_terrain()
        self._setup_materials()
        self._setup_objects()  # Load 3D objects (if any)
        self._setup_atmosphere()
        self._setup_shadow_cache()
        self._setup_subsurface_grid()
        self._setup_solver()
        self._setup_initial_conditions()
        self._setup_output()

        # Simulation state
        self.current_time = config.simulation.start_time
        self.step_count = 0
        self.wall_clock_start = None

    def _setup_terrain(self):
        """Initialize terrain grid from configuration"""
        if self.verbose:
            print(f"  - Setting up terrain ({self.config.terrain.type})...")

        if self.config.terrain.type == "flat":
            self.terrain = create_synthetic_terrain(
                nx=self.config.terrain.nx,
                ny=self.config.terrain.ny,
                dx=self.config.terrain.dx,
                dy=self.config.terrain.dy,
                terrain_type='flat'
            )
        elif self.config.terrain.type == "from_file":
            # Load elevation from file
            elevation_data = np.load(self.config.terrain.elevation_file)
            self.terrain = TerrainGrid(
                elevation=elevation_data,
                dx=self.config.terrain.dx,
                dy=self.config.terrain.dy
            )
        elif self.config.terrain.type == "synthetic":
            # Generate synthetic terrain (e.g., Gaussian hill)
            self.terrain = create_synthetic_terrain(
                nx=self.config.terrain.nx,
                ny=self.config.terrain.ny,
                dx=self.config.terrain.dx,
                dy=self.config.terrain.dy,
                terrain_type='rolling_hills'  # Use rolling hills by default
            )
        elif self.config.terrain.type == "ridge":
            # Generate synthetic terrain (e.g., Gaussian hill)
            self.terrain = create_synthetic_terrain(
                nx=self.config.terrain.nx,
                ny=self.config.terrain.ny,
                dx=self.config.terrain.dx,
                dy=self.config.terrain.dy,
                terrain_type='ridge'  # Use rolling hills by default
            )
        else:
            raise ValueError(f"Unknown terrain type: {self.config.terrain.type}")

    def _setup_materials(self):
        """Initialize material field from configuration"""
        if self.verbose:
            print(f"  - Setting up materials ({self.config.materials.type})...")

        # Load or create material database
        mat_db_path = Path('data/materials/representative_materials.json')
        if mat_db_path.exists():
            # Load from file if it exists
            mat_db = MaterialDatabase()
            mat_db.load_from_json(str(mat_db_path))
        else:
            # Create representative database
            mat_db = create_representative_materials()

        # Add custom materials if provided
        for i, custom_mat in enumerate(self.config.materials.custom_materials):
            # Assign unique class_id starting from 100 to avoid conflicts
            mat_db.add_material(MaterialProperties(
                class_id=100 + i,
                name=custom_mat['name'],
                k=custom_mat['conductivity'],
                rho=custom_mat['density'],
                cp=custom_mat['specific_heat'],
                alpha=custom_mat.get('absorptivity', 0.9),
                epsilon=custom_mat.get('emissivity', 0.9),
                roughness=custom_mat.get('roughness', 0.01)
            ))

        # Helper to find material class_id by name (case-insensitive)
        def find_material_by_name(name: str) -> int:
            """Find material class_id by name"""
            name_lower = name.lower()
            for class_id, mat in mat_db.materials.items():
                if mat.name.lower() == name_lower:
                    return class_id
            # If not found, raise error
            available = [m.name for m in mat_db.materials.values()]
            raise ValueError(f"Material '{name}' not found. Available: {available}")

        # Create material field
        ny, nx = self.terrain.ny, self.terrain.nx
        nz = self.config.subsurface.n_layers
        self.materials = MaterialField(ny, nx, nz, mat_db)

        if self.config.materials.type == "uniform":
            # Uniform material across entire domain
            material_name = self.config.materials.default_material
            material_class_id = find_material_by_name(material_name)

            # Create uniform classification map
            material_class = np.full((ny, nx), material_class_id, dtype=np.int32)
            self.materials.assign_from_classification(material_class)

        elif self.config.materials.type == "from_classification":
            # Load material classification from file
            classification = np.load(self.config.materials.classification_file)
            self.materials.assign_from_classification(classification)
        else:
            raise ValueError(f"Unknown materials type: {self.config.materials.type}")

    def _setup_objects(self):
        """Load 3D thermal objects from configuration"""
        self.objects = []

        if not hasattr(self.config, 'objects') or not self.config.objects:
            # No objects defined
            if self.verbose:
                print(f"  - No 3D objects defined")
            return

        if self.verbose:
            print(f"  - Loading {len(self.config.objects)} 3D object(s)...")

        from src.objects import ThermalObject

        # Load material database to get materials for objects
        mat_db_path = Path('data/materials/representative_materials.json')
        if mat_db_path.exists():
            mat_db = MaterialDatabase()
            mat_db.load_from_json(str(mat_db_path))
        else:
            mat_db = create_representative_materials()

        for obj_config in self.config.objects:
            if not obj_config.enabled:
                continue

            # Find material
            material = None
            for mat in mat_db.materials.values():
                if mat.name.lower() == obj_config.material.lower():
                    material = mat
                    break

            if material is None:
                available = [m.name for m in mat_db.materials.values()]
                raise ValueError(f"Material '{obj_config.material}' not found for object '{obj_config.name}'. Available: {available}")

            # Load object
            mesh_file = Path('data/objects') / obj_config.mesh_file
            if not mesh_file.exists():
                raise FileNotFoundError(f"Object mesh file not found: {mesh_file}")

            # Handle ground clamping
            location = obj_config.location.copy()  # Don't modify config
            if obj_config.ground_clamped:
                # Query terrain elevation at (x, y) location
                x, y, z_config = location

                # Convert x, y to grid indices
                ix = int(x / self.terrain.dx)
                iy = int(y / self.terrain.dy)

                # Clamp to grid bounds
                ix = max(0, min(ix, self.terrain.elevation.shape[1] - 1))
                iy = max(0, min(iy, self.terrain.elevation.shape[0] - 1))

                # Get terrain elevation at this location
                terrain_z = self.terrain.elevation[iy, ix]

                # Override z-coordinate with terrain elevation
                location[2] = terrain_z

                if self.verbose:
                    print(f"    Ground clamping '{obj_config.name}': z={z_config:.2f}m → {terrain_z:.2f}m at ({x:.1f}, {y:.1f})")

            # Create ThermalObject
            thermal_obj = ThermalObject(
                name=obj_config.name,
                mesh_file=str(mesh_file),
                location=location,
                material=material,
                thickness=obj_config.thickness,
                rotation=obj_config.rotation if obj_config.rotation else None
            )

            self.objects.append(thermal_obj)

            if self.verbose:
                print(f"    Loaded '{obj_config.name}': {thermal_obj.faces.shape[0]} faces, material={material.name}")

    def _setup_atmosphere(self):
        """Initialize atmospheric conditions from configuration"""
        if self.verbose:
            print(f"  - Setting up atmosphere...")

        # Create temperature function based on model
        if self.config.atmosphere.temperature_model == "diurnal":
            T_air_func = create_diurnal_temperature(
                T_mean=self.config.atmosphere.temperature_mean,
                T_amplitude=self.config.atmosphere.temperature_amplitude
            )
        elif self.config.atmosphere.temperature_model == "constant":
            T_air_func = self.config.atmosphere.temperature_mean
        else:
            raise ValueError(f"Unknown temperature model: {self.config.atmosphere.temperature_model}")

        # Create wind speed function based on model
        if self.config.atmosphere.wind_model == "diurnal":
            wind_func = create_diurnal_wind(
                wind_mean=self.config.atmosphere.wind_mean,
                wind_amplitude=self.config.atmosphere.wind_amplitude
            )
        elif self.config.atmosphere.wind_model == "constant":
            wind_func = self.config.atmosphere.wind_mean
        else:
            raise ValueError(f"Unknown wind model: {self.config.atmosphere.wind_model}")

        # Create AtmosphericConditions with the functions
        self.atmosphere = AtmosphericConditions(
            T_air=T_air_func,
            wind_speed=wind_func
        )

    def _setup_shadow_cache(self):
        """Initialize shadow cache"""
        if self.verbose:
            print(f"  - Setting up shadow cache...")

        # Always create fresh shadow cache for now (TODO: smart cache validation)
        # Old cache files may have incorrect timezone or date range
        cache_file = Path(self.config.output.directory) / "shadow_cache.npz"
        if cache_file.exists():
            if self.verbose:
                print(f"    Removing old shadow cache (will recompute)")
            cache_file.unlink()

        # Create new empty cache and populate it
        self.shadow_cache = ShadowCache()

        # Populate shadow cache for simulation period
        # Convert longitude: config uses West positive, solar_position expects East positive
        longitude_east_positive = -self.config.site.longitude

        if self.verbose:
            if self.objects:
                print(f"    Computing shadows for simulation period (terrain + {len(self.objects)} objects)...")
            else:
                print(f"    Computing shadows for simulation period...")

        # Compute shadows for each unique day in simulation
        start_date = self.config.simulation.start_time.date()
        end_date = self.config.simulation.end_time.date()

        from datetime import timedelta
        current_date = start_date
        while current_date <= end_date:
            dt_day = datetime.combine(current_date, datetime.min.time())

            # Compute terrain shadows (terrain self-shadowing only)
            self.shadow_cache.compute_daily_shadows(
                terrain_elevation=self.terrain.elevation,
                dx=self.terrain.dx,
                dy=self.terrain.dy,
                latitude=self.config.site.latitude,
                longitude=longitude_east_positive,
                date=dt_day,
                time_step_minutes=self.config.solver.shadow_timestep_minutes,
                timezone_offset=self.config.site.timezone_offset
            )
            current_date += timedelta(days=1)

        # If objects present, add object shadows to terrain shadow maps
        if self.objects:
            from src.solar import compute_object_shadows, solar_position
            import numpy as np

            if self.verbose:
                print(f"    Adding object shadows to terrain shadow maps...")

            # Create terrain coordinate grids
            nx, ny = self.terrain.elevation.shape[1], self.terrain.elevation.shape[0]
            x = np.arange(nx) * self.terrain.dx
            y = np.arange(ny) * self.terrain.dy
            terrain_x, terrain_y = np.meshgrid(x, y)

            # Update each cached shadow map with object shadows
            for idx in range(len(self.shadow_cache.times)):
                # Get sun position for this timestep
                current_time = self.shadow_cache.times[idx]
                azimuth, elevation = solar_position(
                    self.config.site.latitude,
                    longitude_east_positive,
                    current_time,
                    self.config.site.timezone_offset
                )

                # Compute object shadows for this sun position
                terrain_obj_shadows, object_shadows = compute_object_shadows(
                    self.terrain.elevation,
                    terrain_x,
                    terrain_y,
                    self.objects,
                    azimuth,
                    elevation
                )

                # Combine with existing terrain shadows (logical OR)
                existing_shadows = self.shadow_cache.shadow_maps[idx]
                combined_shadows = np.logical_or(existing_shadows, terrain_obj_shadows)
                self.shadow_cache.shadow_maps[idx] = combined_shadows

                # Store object shadow data for this timestep
                if not hasattr(self.shadow_cache, 'object_shadows'):
                    self.shadow_cache.object_shadows = []
                if len(self.shadow_cache.object_shadows) <= idx:
                    self.shadow_cache.object_shadows.append(object_shadows)
                else:
                    self.shadow_cache.object_shadows[idx] = object_shadows

        if self.verbose:
            print(f"    Shadow cache populated with {len(self.shadow_cache.times)} timesteps")

    def _setup_subsurface_grid(self):
        """Initialize subsurface vertical grid"""
        if self.verbose:
            print(f"  - Setting up subsurface grid...")

        self.subsurface_grid = SubsurfaceGrid(
            z_max=self.config.subsurface.z_max,
            n_layers=self.config.subsurface.n_layers,
            stretch_factor=self.config.subsurface.stretch_factor
        )

    def _setup_solver(self):
        """Initialize thermal solver"""
        if self.verbose:
            print(f"  - Setting up thermal solver...")

        # Convert longitude: config uses West positive, solar_position expects East positive
        longitude_east_positive = -self.config.site.longitude

        self.solver = ThermalSolver(
            terrain=self.terrain,
            materials=self.materials,
            atmosphere=self.atmosphere,
            shadow_cache=self.shadow_cache,
            latitude=self.config.site.latitude,
            longitude=longitude_east_positive,
            altitude=self.config.site.altitude,
            timezone_offset=self.config.site.timezone_offset,
            subsurface_grid=self.subsurface_grid,
            dt=self.config.simulation.dt,
            enable_lateral_conduction=self.config.solver.enable_lateral_conduction,
            lateral_conductivity_factor=self.config.solver.lateral_conductivity_factor
        )

    def _setup_initial_conditions(self):
        """Set initial temperature field"""
        if self.verbose:
            print(f"  - Setting up initial conditions ({self.config.initial_conditions.type})...")

        if self.config.initial_conditions.type == "uniform":
            # Uniform temperature
            T_init = self.config.initial_conditions.temperature_kelvin
            self.solver.initialize(T_init)
            self.solver.temp_field.time = self.config.simulation.start_time

        elif self.config.initial_conditions.type == "spinup":
            # Run spinup period
            if self.verbose:
                print(f"    Running {self.config.initial_conditions.spinup_days} day spinup...")

            # Start from uniform temperature
            T_init = self.config.initial_conditions.temperature_kelvin
            self.solver.initialize(T_init)

            # Set time to start of spinup
            spinup_start = self.config.simulation.start_time - timedelta(
                days=self.config.initial_conditions.spinup_days
            )
            self.solver.temp_field.time = spinup_start

            # Run spinup
            spinup_end = self.config.simulation.start_time
            spinup_time = spinup_start

            while spinup_time < spinup_end:
                print(f"  - Spintup time is currently ({spinup_time})")
                self.solver.step(spinup_time)
                spinup_time += timedelta(seconds=self.config.simulation.dt)
                self.solver.temp_field.time = spinup_time

            if self.verbose:
                print(f"    Spinup complete")

        elif self.config.initial_conditions.type == "from_file":
            # Load from file
            state = np.load(self.config.initial_conditions.initial_state_file)
            self.solver.temp_field.T_surface[:] = state['T_surface']
            self.solver.temp_field.T_subsurface[:] = state['T_subsurface']
            self.solver.temp_field.time = self.config.simulation.start_time
            self.solver.temp_field.step_number = 0

        else:
            raise ValueError(f"Unknown initial conditions type: {self.config.initial_conditions.type}")

    def _setup_output(self):
        """Setup output directory and metadata"""
        if self.verbose:
            print(f"  - Setting up output directory...")

        # Create output directory (already validated in config)
        self.output_dir = Path(self.config.output.directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for outputs
        (self.output_dir / "fields").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        # Save configuration to output directory
        config_file = self.output_dir / "config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            # Convert config to dict for YAML serialization
            # (could implement a to_dict() method on config classes)
            yaml.dump({
                'simulation': {
                    'name': self.config.simulation.name,
                    'start_time': self.config.simulation.start_time.isoformat(),
                    'end_time': self.config.simulation.end_time.isoformat(),
                    'time_step': self.config.simulation.dt
                },
                'site': {
                    'latitude': self.config.site.latitude,
                    'longitude': self.config.site.longitude,
                    'altitude': self.config.site.altitude
                },
                'terrain': {
                    'type': self.config.terrain.type,
                    'nx': self.config.terrain.nx,
                    'ny': self.config.terrain.ny,
                    'dx': self.config.terrain.dx,
                    'dy': self.config.terrain.dy
                },
                # ... etc
            }, f)

        # Create subdirectories
        (self.output_dir / "fields").mkdir(exist_ok=True)
        (self.output_dir / "diagnostics").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        if self.config.output.generate_plots:
            (self.output_dir / "plots").mkdir(exist_ok=True)

        # Initialize diagnostics log
        self.diagnostics = []

    def run(self):
        """
        Run the simulation from start to end time.

        Handles:
        - Main time stepping loop
        - Progress reporting
        - Periodic output saving
        - Checkpointing
        """
        # Calculate total steps
        total_duration = self.config.simulation.duration_seconds
        total_steps = int(total_duration / self.config.simulation.dt)
        save_interval_steps = int(self.config.output.save_interval_seconds / self.config.simulation.dt)

        if self.config.output.checkpoint_interval_seconds:
            checkpoint_interval_steps = int(
                self.config.output.checkpoint_interval_seconds / self.config.simulation.dt
            )
        else:
            checkpoint_interval_steps = None

        # Start wall clock timer
        self.wall_clock_start = time.time()

        # Main simulation loop
        self.current_time = self.config.simulation.start_time

        if self.verbose:
            print(f"Running {total_steps} time steps...")
            print()

        while self.current_time < self.config.simulation.end_time:
            # Advance one time step
            print(f"  - Running time ({self.current_time})...")
            temp_field = self.solver.step(self.current_time)

            # Update state
            self.current_time += timedelta(seconds=self.config.simulation.dt)
            self.step_count += 1

            # Progress reporting
            if self.verbose and self.step_count % 100 == 0:
                self._report_progress(total_steps)

            # Save outputs
            if self.step_count % save_interval_steps == 0:
                self._save_output(temp_field)

            # Checkpointing
            if checkpoint_interval_steps and self.step_count % checkpoint_interval_steps == 0:
                self._save_checkpoint(temp_field)

        # Final output
        self._save_output(temp_field)
        self._save_diagnostics()

        # Save shadow cache if it was updated
        cache_file = self.output_dir / "shadow_cache.npz"
        self.shadow_cache.save(str(cache_file))

        if self.verbose:
            print()
            self._print_final_summary()

    def _report_progress(self, total_steps: int):
        """Print progress report"""
        elapsed = time.time() - self.wall_clock_start
        percent = 100.0 * self.step_count / total_steps
        rate = self.step_count / elapsed if elapsed > 0 else 0
        eta_seconds = (total_steps - self.step_count) / rate if rate > 0 else 0

        T_mean = np.mean(self.solver.temp_field.T_surface)
        T_min = np.min(self.solver.temp_field.T_surface)
        T_max = np.max(self.solver.temp_field.T_surface)

        print(f"Step {self.step_count}/{total_steps} ({percent:.1f}%) | "
              f"Time: {self.current_time.strftime('%Y-%m-%d %H:%M')} | "
              f"T_surf: {T_mean:.1f}K [{T_min:.1f}, {T_max:.1f}] | "
              f"Rate: {rate:.1f} steps/s | ETA: {eta_seconds/60:.0f} min")

    def _save_output(self, temp_field: TemperatureField):
        """Save temperature fields and diagnostics"""
        timestamp = self.current_time.strftime("%Y%m%d_%H%M%S")

        if self.config.output.save_temperature_fields:
            output_file = self.output_dir / "fields" / f"temperature_{timestamp}.npz"
            np.savez_compressed(
                output_file,
                T_surface=temp_field.T_surface,
                T_subsurface=temp_field.T_subsurface,
                time=self.current_time.isoformat(),
                step=temp_field.step_number
            )

        if self.config.output.save_energy_diagnostics:
            # Compute energy diagnostics
            T_mean = np.mean(temp_field.T_surface)
            T_std = np.std(temp_field.T_surface)
            T_min = np.min(temp_field.T_surface)
            T_max = np.max(temp_field.T_surface)

            # Get flux diagnostics if available
            flux_data = {}
            if self.solver.latest_fluxes is not None:
                fluxes = self.solver.latest_fluxes
                flux_data = {
                    'Q_solar_mean': float(np.mean(fluxes['Q_solar'])),
                    'Q_atm_mean': float(np.mean(fluxes['Q_atm'])),
                    'Q_emission_mean': float(np.mean(fluxes['Q_emission'])),
                    'Q_conv_mean': float(np.mean(fluxes['Q_conv'])),
                    'Q_net_mean': float(np.mean(fluxes['Q_net'])),
                    'Q_solar_max': float(np.max(fluxes['Q_solar'])),
                    'Q_net_max': float(np.max(fluxes['Q_net'])),
                    'Q_net_min': float(np.min(fluxes['Q_net']))
                }

            self.diagnostics.append({
                'time': self.current_time.isoformat(),
                'step': self.step_count,
                'T_surface_mean': float(T_mean),
                'T_surface_std': float(T_std),
                'T_surface_min': float(T_min),
                'T_surface_max': float(T_max),
                **flux_data
            })

        # Generate plots if enabled
        if self.config.output.generate_plots:
            self._generate_plots(temp_field, timestamp)

    def _save_checkpoint(self, temp_field: TemperatureField):
        """Save simulation checkpoint for restart"""
        timestamp = self.current_time.strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.output_dir / "checkpoints" / f"checkpoint_{timestamp}.npz"

        np.savez_compressed(
            checkpoint_file,
            T_surface=temp_field.T_surface,
            T_subsurface=temp_field.T_subsurface,
            time=self.current_time.isoformat(),
            step=temp_field.step_number
        )

        if self.verbose:
            print(f"  [Checkpoint saved: {checkpoint_file.name}]")

    def _save_diagnostics(self):
        """Save diagnostics time series to JSON"""
        if not self.diagnostics:
            return

        diagnostics_file = self.output_dir / "diagnostics" / "timeseries.json"
        with open(diagnostics_file, 'w') as f:
            json.dump(self.diagnostics, f, indent=2)

    def _print_final_summary(self):
        """Print final simulation summary"""
        elapsed = time.time() - self.wall_clock_start
        rate = self.step_count / elapsed if elapsed > 0 else 0

        print()
        print("Simulation Summary:")
        print(f"  Total steps:       {self.step_count}")
        print(f"  Wall clock time:   {elapsed:.1f} s ({elapsed/60:.1f} min)")
        print(f"  Performance:       {rate:.1f} steps/s")
        print(f"  Simulated time:    {self.config.simulation.duration_seconds/3600:.1f} hours")
        print(f"  Final time:        {self.current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Output directory:  {self.output_dir}")

    def _generate_plots(self, temp_field: TemperatureField, timestamp: str):
        """
        Generate visualization plots based on configuration.

        Parameters
        ----------
        temp_field : TemperatureField
            Current temperature field
        timestamp : str
            Timestamp string for filenames
        """
        plot_dir = self.output_dir / "plots"
        ext = self.config.output.plot_format
        dpi = self.config.output.plot_dpi

        for plot_type in self.config.output.plot_types:
            try:
                if plot_type == "terrain_elevation":
                    # Plot terrain elevation (only once)
                    terrain_file = plot_dir / f"terrain_elevation.{ext}"
                    if not terrain_file.exists():
                        plot_terrain_elevation(self.terrain, terrain_file, dpi)

                elif plot_type == "surface_temperature":
                    self._plot_surface_temperature(temp_field, plot_dir, timestamp, ext, dpi)

                elif plot_type == "diagnostics_timeseries":
                    if len(self.diagnostics) > 1:  # Need at least 2 points
                        # Use enhanced flux timeseries plot
                        plot_flux_timeseries(
                            self.diagnostics,
                            plot_dir / f"diagnostics_timeseries.{ext}",
                            dpi
                        )

                elif plot_type == "hourly_summary":
                    # Multi-panel summary with T, shadow, fluxes, subsurface
                    if self.solver.latest_fluxes is not None and self.solver.latest_shadow_map is not None:
                        plot_hourly_summary(
                            self.terrain,
                            temp_field,
                            self.solver.latest_fluxes,
                            self.solver.latest_shadow_map,
                            self.subsurface_grid,
                            self.current_time,
                            plot_dir / f"hourly_summary_{timestamp}.{ext}",
                            solar_azimuth=self.solver.latest_solar_azimuth,
                            solar_elevation=self.solver.latest_solar_elevation,
                            dpi=dpi
                        )

                elif plot_type == "subsurface_profile":
                    self._plot_subsurface_profile(temp_field, plot_dir, timestamp, ext, dpi)

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to generate {plot_type} plot: {e}")

    def _plot_surface_temperature(self, temp_field, plot_dir, timestamp, ext, dpi):
        """Plot surface temperature field"""
        plt.figure(figsize=(10, 8))

        T_surface = temp_field.T_surface
        img = plt.imshow(T_surface, cmap='hot', origin='lower', aspect='auto')
        plt.colorbar(img, label='Temperature [K]')

        plt.title(f'Surface Temperature - {self.current_time.strftime("%Y-%m-%d %H:%M")}')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')

        # Add statistics text
        T_mean = np.mean(T_surface)
        T_min = np.min(T_surface)
        T_max = np.max(T_surface)
        stats_text = f'Mean: {T_mean:.1f} K\nMin: {T_min:.1f} K\nMax: {T_max:.1f} K'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(plot_dir / f"surface_temp_{timestamp}.{ext}", dpi=dpi)
        plt.close()

    def _plot_diagnostics_timeseries(self, plot_dir, ext, dpi):
        """Plot temperature diagnostics time series"""
        plt.figure(figsize=(12, 6))

        times = [d['step'] * self.config.simulation.dt / 3600 for d in self.diagnostics]  # Convert to hours
        T_mean = [d['T_surface_mean'] for d in self.diagnostics]
        T_min = [d['T_surface_min'] for d in self.diagnostics]
        T_max = [d['T_surface_max'] for d in self.diagnostics]

        plt.plot(times, T_mean, 'k-', linewidth=2, label='Mean')
        plt.fill_between(times, T_min, T_max, alpha=0.3, color='red', label='Min/Max Range')
        plt.plot(times, T_min, 'b--', linewidth=1, alpha=0.5)
        plt.plot(times, T_max, 'r--', linewidth=1, alpha=0.5)

        plt.xlabel('Simulation Time [hours]')
        plt.ylabel('Surface Temperature [K]')
        plt.title(f'Temperature Evolution - {self.config.simulation.name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Update timeseries plot (overwrite)
        plt.savefig(plot_dir / f"diagnostics_timeseries.{ext}", dpi=dpi)
        plt.close()

    def _plot_subsurface_profile(self, temp_field, plot_dir, timestamp, ext, dpi):
        """Plot subsurface temperature profile at center of domain"""
        plt.figure(figsize=(8, 10))

        # Get center point
        ny, nx = temp_field.T_surface.shape
        j_center = ny // 2
        i_center = nx // 2

        # Extract temperature profile
        T_profile = temp_field.T_subsurface[j_center, i_center, :]
        z_nodes = self.subsurface_grid.z_nodes

        plt.plot(T_profile, -z_nodes, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Temperature [K]')
        plt.ylabel('Depth [m]')
        plt.title(f'Subsurface Profile (Center) - {self.current_time.strftime("%Y-%m-%d %H:%M")}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(plot_dir / f"subsurface_profile_{timestamp}.{ext}", dpi=dpi)
        plt.close()
