"""
Configuration system for thermal terrain simulator.

Provides YAML-based configuration loading with three-level validation:
1. Schema validation (structure and types)
2. Physics validation (warnings for questionable settings)
3. Runtime validation (critical errors only)

Author: Thermal Simulator Team
Date: December 2025
"""

import yaml
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class SimulationConfig:
    """Main simulation configuration"""
    name: str
    start_time: datetime
    end_time: datetime
    dt: float  # seconds

    @property
    def duration_seconds(self) -> float:
        """Total simulation duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class SiteConfig:
    """Site location configuration"""
    latitude: float  # degrees N
    longitude: float  # degrees W (positive = West)
    altitude: float  # meters
    timezone_offset: float = 0.0  # hours from UTC (e.g., -7.0 for MST)


@dataclass
class TerrainConfig:
    """Terrain grid configuration"""
    type: str  # "flat", "from_file", "synthetic"
    nx: int
    ny: int
    dx: float  # meters
    dy: float  # meters
    flat_elevation: float = 0.0  # meters (for type="flat")
    elevation_file: Optional[str] = None  # for type="from_file"


@dataclass
class MaterialsConfig:
    """Materials configuration"""
    type: str  # "uniform" or "from_classification"
    default_material: str  # name from built-in database
    classification_file: Optional[str] = None  # for type="from_classification"
    custom_materials: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AtmosphereConfig:
    """Atmospheric conditions configuration"""
    temperature_model: str  # "diurnal"
    temperature_mean: float  # Kelvin
    temperature_amplitude: float  # Kelvin
    wind_model: str  # "diurnal"
    wind_mean: float  # m/s
    wind_amplitude: float  # m/s
    sky_temperature_model: str = "idso"  # or "swinbank"


@dataclass
class SubsurfaceConfig:
    """Subsurface grid configuration"""
    z_max: float = 0.5  # meters
    n_layers: int = 20
    stretch_factor: float = 1.2


@dataclass
class SolverConfig:
    """Solver configuration"""
    enable_lateral_conduction: bool = False
    lateral_conductivity_factor: float = 1.0
    shadow_timestep_minutes: float = 60.0  # Time step for shadow cache (minutes)


@dataclass
class InitialConditionsConfig:
    """Initial conditions configuration"""
    type: str  # "uniform", "spinup", or "from_file"
    temperature_kelvin: float = 298.0  # for type="uniform"
    spinup_days: int = 1  # for type="spinup"
    initial_state_file: Optional[str] = None  # for type="from_file"


@dataclass
class ObjectConfig:
    """3D thermal object configuration"""
    name: str  # Human-readable identifier
    mesh_file: str  # Path to OBJ file (relative to data/objects/ or absolute)
    location: List[float]  # [x, y, z] in terrain coordinates (meters)
    material: str  # Material name from database
    thickness: float  # Wall/surface thickness in meters (for 1D thermal solver)
    rotation: Optional[List[float]] = None  # [rx, ry, rz] Euler angles (degrees)
    enabled: bool = True  # Allow toggling objects on/off
    ground_clamped: bool = False  # If True, ignore z-coordinate and place on terrain surface


@dataclass
class OutputConfig:
    """Output configuration"""
    directory: str
    save_interval_seconds: float = 3600.0  # hourly
    save_temperature_fields: bool = True
    save_energy_diagnostics: bool = True
    checkpoint_interval_seconds: Optional[float] = None  # None = no checkpoints

    # Visualization options
    generate_plots: bool = False  # Auto-generate visualization plots
    plot_format: str = "png"  # png, pdf, svg
    plot_dpi: int = 150  # Resolution for raster formats
    plot_types: List[str] = field(default_factory=lambda: [
        "surface_temperature",
        "diagnostics_timeseries"
    ])  # Which plots to generate
    # Available plot types:
    #   - "terrain_elevation": Terrain elevation map
    #   - "surface_temperature": Temperature field snapshots
    #   - "diagnostics_timeseries": Time series of T/fluxes
    #   - "hourly_summary": Multi-panel (T, shadow, fluxes, subsurface)


@dataclass
class ThermalSimConfig:
    """Complete thermal simulation configuration"""
    simulation: SimulationConfig
    site: SiteConfig
    terrain: TerrainConfig
    materials: MaterialsConfig
    atmosphere: AtmosphereConfig
    subsurface: SubsurfaceConfig
    solver: SolverConfig
    initial_conditions: InitialConditionsConfig
    output: OutputConfig
    objects: List[ObjectConfig] = field(default_factory=list)  # 3D thermal objects (optional)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ThermalSimConfig':
        """
        Load configuration from YAML file with validation.

        Performs three-level validation:
        1. Schema validation (structure and types)
        2. Physics validation (warnings for questionable settings)
        3. Runtime validation (file existence, etc.)

        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file

        Returns
        -------
        ThermalSimConfig
            Validated configuration object
        """
        # Load YAML
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse and validate each section
        config = cls._parse_config(config_dict)

        # Physics validation (warnings)
        config._validate_physics()

        # Runtime validation (errors for critical issues)
        config._validate_runtime()

        return config

    @classmethod
    def _load_objects_from_file(cls, filepath: str) -> List[ObjectConfig]:
        """
        Load object definitions from a CSV/TSV file.

        File format (CSV with header):
        name,mesh_file,x,y,z,material,thickness,rx,ry,rz,enabled,ground_clamped

        Columns:
        - name: Object identifier (required)
        - mesh_file: Path to OBJ file (required)
        - x, y, z: Location coordinates in meters (required)
        - material: Material name (required)
        - thickness: Wall thickness in meters (required)
        - rx, ry, rz: Rotation angles in degrees (optional, default 0)
        - enabled: true/false or 1/0 (optional, default true)
        - ground_clamped: true/false or 1/0 (optional, default false)

        Lines starting with # are comments.
        Empty lines are ignored.

        Parameters
        ----------
        filepath : str
            Path to objects CSV file

        Returns
        -------
        List[ObjectConfig]
            List of object configurations
        """
        import csv

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Objects file not found: {filepath}")

        objects = []
        with open(filepath, 'r') as f:
            # Read lines and filter out comments
            lines = []
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    lines.append(line)

            if not lines:
                raise ValueError(f"No valid lines found in objects file: {filepath}")

            # Auto-detect delimiter from header line
            header = lines[0]
            delimiter = '\t' if '\t' in header else ','

            # Parse CSV from filtered lines
            import io
            csv_data = io.StringIO(''.join(lines))
            reader = csv.DictReader(csv_data, delimiter=delimiter)

            for i, row in enumerate(reader, start=2):  # Line 2 (after header)
                # Skip empty rows
                if not row or not row.get('name', '').strip():
                    continue

                try:
                    # Parse required fields
                    name = row['name'].strip()
                    mesh_file = row['mesh_file'].strip()
                    x = float(row['x'])
                    y = float(row['y'])
                    z = float(row['z'])
                    material = row['material'].strip()
                    thickness = float(row['thickness'])

                    # Parse optional rotation (default to [0, 0, 0])
                    rx = float(row.get('rx', 0))
                    ry = float(row.get('ry', 0))
                    rz = float(row.get('rz', 0))
                    rotation = [rx, ry, rz] if (rx != 0 or ry != 0 or rz != 0) else None

                    # Parse optional enabled flag (default to True)
                    enabled_str = row.get('enabled', 'true').strip().lower()
                    enabled = enabled_str in ('true', '1', 'yes', 'y')

                    # Parse optional ground_clamped flag (default to False)
                    ground_clamped_str = row.get('ground_clamped', 'false').strip().lower()
                    ground_clamped = ground_clamped_str in ('true', '1', 'yes', 'y')

                    obj = ObjectConfig(
                        name=name,
                        mesh_file=mesh_file,
                        location=[x, y, z],
                        material=material,
                        thickness=thickness,
                        rotation=rotation,
                        enabled=enabled,
                        ground_clamped=ground_clamped
                    )
                    objects.append(obj)

                except KeyError as e:
                    raise ValueError(
                        f"Missing required column in objects file (line {i}): {e}"
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Invalid value in objects file (line {i}): {e}"
                    )

        print(f"INFO: Loaded {len(objects)} object(s) from {filepath}")
        return objects

    @classmethod
    def _parse_config(cls, config_dict: Dict[str, Any]) -> 'ThermalSimConfig':
        """Parse configuration dictionary into dataclass objects"""

        # Parse simulation section
        sim_dict = config_dict.get('simulation', {})
        start_time = datetime.fromisoformat(sim_dict['start_time'])

        # Handle duration (either end_time or duration_hours)
        if 'end_time' in sim_dict:
            end_time = datetime.fromisoformat(sim_dict['end_time'])
        elif 'duration_hours' in sim_dict:
            end_time = start_time + timedelta(hours=sim_dict['duration_hours'])
        else:
            raise ValueError("Must specify either 'end_time' or 'duration_hours'")

        simulation = SimulationConfig(
            name=sim_dict.get('name', 'thermal_sim'),
            start_time=start_time,
            end_time=end_time,
            dt=sim_dict.get('time_step', 120.0)
        )

        # Parse site section
        site_dict = config_dict.get('site', {})
        site = SiteConfig(
            latitude=site_dict['latitude'],
            longitude=site_dict['longitude'],
            altitude=site_dict.get('altitude', 0.0),
            timezone_offset=site_dict.get('timezone_offset', 0.0)
        )

        # Parse terrain section
        terrain_dict = config_dict.get('terrain', {})
        terrain = TerrainConfig(
            type=terrain_dict.get('type', 'flat'),
            nx=terrain_dict['nx'],
            ny=terrain_dict['ny'],
            dx=terrain_dict['dx'],
            dy=terrain_dict['dy'],
            flat_elevation=terrain_dict.get('flat_elevation', 0.0),
            elevation_file=terrain_dict.get('elevation_file')
        )

        # Parse materials section
        materials_dict = config_dict.get('materials', {})
        materials = MaterialsConfig(
            type=materials_dict.get('type', 'uniform'),
            default_material=materials_dict.get('default_material', 'sand'),
            classification_file=materials_dict.get('classification_file'),
            custom_materials=materials_dict.get('custom_materials', [])
        )

        # Parse atmosphere section
        atmos_dict = config_dict.get('atmosphere', {})
        temp_dict = atmos_dict.get('temperature', {})
        wind_dict = atmos_dict.get('wind', {})

        atmosphere = AtmosphereConfig(
            temperature_model=temp_dict.get('model', 'diurnal'),
            temperature_mean=temp_dict.get('mean_kelvin', 298.15),
            temperature_amplitude=temp_dict.get('amplitude_kelvin', 10.0),
            wind_model=wind_dict.get('model', 'diurnal'),
            wind_mean=wind_dict.get('mean_speed', 3.0),
            wind_amplitude=wind_dict.get('amplitude', 1.0),
            sky_temperature_model=atmos_dict.get('sky_temperature_model', 'idso')
        )

        # Parse subsurface section
        subsurface_dict = config_dict.get('subsurface', {})
        subsurface = SubsurfaceConfig(
            z_max=subsurface_dict.get('z_max', 0.5),
            n_layers=subsurface_dict.get('n_layers', 20),
            stretch_factor=subsurface_dict.get('stretch_factor', 1.2)
        )

        # Parse solver section
        solver_dict = config_dict.get('solver', {})
        solver = SolverConfig(
            enable_lateral_conduction=solver_dict.get('enable_lateral_conduction', False),
            lateral_conductivity_factor=solver_dict.get('lateral_conductivity_factor', 1.0)
        )

        # Parse initial conditions section
        ic_dict = config_dict.get('initial_conditions', {})
        initial_conditions = InitialConditionsConfig(
            type=ic_dict.get('type', 'uniform'),
            temperature_kelvin=ic_dict.get('temperature_kelvin', 298.0),
            spinup_days=ic_dict.get('spinup_days', 1),
            initial_state_file=ic_dict.get('initial_state_file')
        )

        # Parse output section
        output_dict = config_dict.get('output', {})
        output = OutputConfig(
            directory=output_dict['directory'],
            save_interval_seconds=output_dict.get('save_interval_seconds', 3600.0),
            save_temperature_fields=output_dict.get('save_temperature_fields', True),
            save_energy_diagnostics=output_dict.get('save_energy_diagnostics', True),
            checkpoint_interval_seconds=output_dict.get('checkpoint_interval_seconds'),
            generate_plots=output_dict.get('generate_plots', False),
            plot_format=output_dict.get('plot_format', 'png'),
            plot_dpi=output_dict.get('plot_dpi', 150),
            plot_types=output_dict.get('plot_types', ['surface_temperature', 'diagnostics_timeseries'])
        )

        # Parse objects section (optional)
        # Objects can be specified inline (list) OR loaded from file (dict with 'from_file')
        objects_section = config_dict.get('objects', [])
        objects = []

        if isinstance(objects_section, dict) and 'from_file' in objects_section:
            # Load objects from external file
            objects_file = objects_section['from_file']
            objects = cls._load_objects_from_file(objects_file)
        elif isinstance(objects_section, list):
            # Parse inline object definitions
            for obj_dict in objects_section:
                obj = ObjectConfig(
                    name=obj_dict['name'],
                    mesh_file=obj_dict['mesh_file'],
                    location=obj_dict['location'],
                    material=obj_dict['material'],
                    thickness=obj_dict['thickness'],
                    rotation=obj_dict.get('rotation'),
                    enabled=obj_dict.get('enabled', True),
                    ground_clamped=obj_dict.get('ground_clamped', False)
                )
                objects.append(obj)
        else:
            raise ValueError(
                "objects section must be either a list of objects or a dict with 'from_file' key"
            )

        return cls(
            simulation=simulation,
            site=site,
            terrain=terrain,
            materials=materials,
            atmosphere=atmosphere,
            subsurface=subsurface,
            solver=solver,
            initial_conditions=initial_conditions,
            output=output,
            objects=objects
        )

    def _validate_physics(self):
        """
        Level 2 validation: Physics and stability warnings.

        Issues warnings for potentially problematic settings but doesn't fail.
        """
        # Check time step stability
        dx_min = min(self.terrain.dx, self.terrain.dy)

        # Rough stability estimate for lateral conduction (if enabled)
        if self.solver.enable_lateral_conduction:
            # Assume typical sand diffusivity
            alpha_typical = 1e-6  # m^2/s
            Fo_max = alpha_typical * self.simulation.dt / dx_min**2
            if Fo_max > 0.5:
                warnings.warn(
                    f"Time step dt={self.simulation.dt}s may be UNSTABLE for lateral conduction "
                    f"with dx={dx_min}m. Fourier number ~ {Fo_max:.3f} > 0.5. "
                    f"Recommend dt ≤ {0.5 * dx_min**2 / alpha_typical:.0f}s",
                    UserWarning
                )
            elif Fo_max > 0.01:
                warnings.warn(
                    f"Time step dt={self.simulation.dt}s may be INACCURATE for lateral conduction. "
                    f"Fourier number ~ {Fo_max:.3f} > 0.01. Consider smaller dt for better accuracy.",
                    UserWarning
                )

        # Check grid resolution
        if dx_min < 0.05:
            warnings.warn(
                f"Very fine grid spacing ({dx_min}m) - simulation may be slow. "
                f"Consider coarser grid if appropriate for your application.",
                UserWarning
            )

        if dx_min > 10.0:
            warnings.warn(
                f"Coarse grid spacing ({dx_min}m) may not resolve important features. "
                f"Consider finer resolution if needed.",
                UserWarning
            )

        # Check subsurface depth
        # Rule of thumb: skin depth ~ sqrt(alpha * period)
        # For diurnal (24hr), sand: ~0.15m, rock: ~0.5m
        if self.subsurface.z_max < 0.3:
            warnings.warn(
                f"Shallow subsurface depth ({self.subsurface.z_max}m) may not capture "
                f"full diurnal heat penetration. Recommend z_max ≥ 0.5m for daily cycles.",
                UserWarning
            )

        # Check output interval
        if self.output.save_interval_seconds > 7200:  # > 2 hours
            warnings.warn(
                f"Large output interval ({self.output.save_interval_seconds/3600:.1f} hours) "
                f"may miss important temporal features. Consider more frequent output.",
                UserWarning
            )

        # Check simulation duration vs time step
        n_steps = int(self.simulation.duration_seconds / self.simulation.dt)
        if n_steps > 100000:
            warnings.warn(
                f"Very long simulation ({n_steps} time steps). Consider larger dt if appropriate.",
                UserWarning
            )

        print(f"INFO: Simulation will run {n_steps} time steps over "
              f"{self.simulation.duration_seconds/3600:.1f} hours")

        # Check latitude range
        if not -90 <= self.site.latitude <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {self.site.latitude}")

        # Info about defaults used
        if self.subsurface.z_max == 0.5 and self.subsurface.n_layers == 20:
            print(f"INFO: Using default subsurface grid (z_max=0.5m, n_layers=20)")

    def _validate_runtime(self):
        """
        Level 3 validation: Runtime checks (file existence, etc.).

        Raises errors for critical issues that would prevent execution.
        """
        # Check terrain file if specified
        if self.terrain.type == "from_file":
            if self.terrain.elevation_file is None:
                raise ValueError("terrain.type='from_file' requires 'elevation_file'")
            if not Path(self.terrain.elevation_file).exists():
                raise FileNotFoundError(f"Terrain file not found: {self.terrain.elevation_file}")

        # Check materials file if specified
        if self.materials.type == "from_classification":
            if self.materials.classification_file is None:
                raise ValueError("materials.type='from_classification' requires 'classification_file'")
            if not Path(self.materials.classification_file).exists():
                raise FileNotFoundError(f"Materials file not found: {self.materials.classification_file}")

        # Check initial state file if specified
        if self.initial_conditions.type == "from_file":
            if self.initial_conditions.initial_state_file is None:
                raise ValueError("initial_conditions.type='from_file' requires 'initial_state_file'")
            if not Path(self.initial_conditions.initial_state_file).exists():
                raise FileNotFoundError(
                    f"Initial state file not found: {self.initial_conditions.initial_state_file}"
                )

        # Check object mesh files if specified
        for obj_config in self.objects:
            if not obj_config.enabled:
                continue  # Skip disabled objects

            # Resolve mesh file path (check relative to data/objects/ or absolute)
            mesh_path = Path(obj_config.mesh_file)
            if not mesh_path.is_absolute():
                # Try relative to data/objects/
                mesh_path = Path("data") / "objects" / obj_config.mesh_file

            if not mesh_path.exists():
                raise FileNotFoundError(
                    f"Object mesh file not found: {obj_config.mesh_file} "
                    f"(searched at {mesh_path.absolute()})"
                )

            # Validate location and rotation
            if len(obj_config.location) != 3:
                raise ValueError(
                    f"Object '{obj_config.name}' location must have 3 values [x,y,z], "
                    f"got {len(obj_config.location)}"
                )

            if obj_config.rotation is not None and len(obj_config.rotation) != 3:
                raise ValueError(
                    f"Object '{obj_config.name}' rotation must have 3 values [rx,ry,rz], "
                    f"got {len(obj_config.rotation)}"
                )

            # Check thickness is reasonable
            if obj_config.thickness <= 0:
                raise ValueError(
                    f"Object '{obj_config.name}' thickness must be positive, "
                    f"got {obj_config.thickness}"
                )
            if obj_config.thickness > 1.0:
                warnings.warn(
                    f"Object '{obj_config.name}' has very thick walls ({obj_config.thickness}m). "
                    f"This may affect thermal response.",
                    UserWarning
                )

        # Print object summary
        enabled_objects = [obj for obj in self.objects if obj.enabled]
        if enabled_objects:
            print(f"INFO: {len(enabled_objects)} thermal object(s) configured:")
            for obj in enabled_objects:
                print(f"  - {obj.name}: {obj.mesh_file} at {obj.location}")

        # Check output directory writable (create if doesn't exist)
        output_path = Path(self.output.directory)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Cannot create output directory {output_path}: {e}")

        # Check grid sizes reasonable
        total_cells = self.terrain.nx * self.terrain.ny * self.subsurface.n_layers
        if total_cells > 1e8:  # 100 million cells
            raise ValueError(
                f"Grid too large ({total_cells:.0e} cells). "
                f"Reduce nx, ny, or n_layers to avoid memory issues."
            )

        print(f"INFO: Output will be saved to: {output_path.absolute()}")
