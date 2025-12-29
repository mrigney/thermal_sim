"""
Thermal Terrain Simulator

A high-fidelity thermal simulation tool for computing spatially-resolved
temperature distributions on natural terrain for infrared scene generation.

Modules:
--------
terrain : Terrain geometry and properties
materials : Material properties and databases
visualization : Plotting and visualization tools
solar : Solar radiation and shadow computation
atmosphere : Atmospheric conditions and sky temperature
solver : Thermal equation solver and time integration
"""

__version__ = "0.1.0"

from .terrain import TerrainGrid, create_synthetic_terrain
from .materials import MaterialProperties, MaterialDatabase, MaterialField
from .visualization import TerrainVisualizer, quick_terrain_plot, quick_temp_plot
from .solar import (solar_position, sun_vector, clear_sky_irradiance,
                   compute_shadow_map, ShadowCache, irradiance_on_surface)
from .atmosphere import (AtmosphericConditions, create_diurnal_temperature,
                        create_diurnal_wind)
from .solver import (SubsurfaceGrid, TemperatureField, ThermalSolver,
                    compute_energy_balance)

__all__ = [
    'TerrainGrid',
    'create_synthetic_terrain',
    'MaterialProperties',
    'MaterialDatabase',
    'MaterialField',
    'TerrainVisualizer',
    'quick_terrain_plot',
    'quick_temp_plot',
    'solar_position',
    'sun_vector',
    'clear_sky_irradiance',
    'compute_shadow_map',
    'ShadowCache',
    'irradiance_on_surface',
    'AtmosphericConditions',
    'create_diurnal_temperature',
    'create_diurnal_wind',
    'SubsurfaceGrid',
    'TemperatureField',
    'ThermalSolver',
    'compute_energy_balance',
]
