#!/usr/bin/env python
"""
Thermal Terrain Simulator - Main CLI Entry Point

Run thermal terrain simulations from YAML configuration files.

Usage:
    python run_simulation.py --config configs/my_sim.yaml
    python run_simulation.py --config configs/base.yaml --output output/run_002
    python run_simulation.py --config configs/desert.yaml --verbose

Author: Thermal Simulator Team
Date: December 2025
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from src.config import ThermalSimConfig
from src.runner import SimulationRunner


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Thermal Terrain Simulator - Run physics-based thermal simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic run:
    python run_simulation.py --config configs/examples/desert_diurnal.yaml

  Override output directory:
    python run_simulation.py --config configs/my_sim.yaml --output output/run_042

  Verbose logging:
    python run_simulation.py --config configs/my_sim.yaml --verbose

Configuration file format:
  See configs/examples/ for example YAML configuration files.
  Documentation: docs/configuration.md
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Override output directory from config file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (progress updates, diagnostics)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate configuration and exit (do not run simulation)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Thermal Simulator v3.0.0'
    )

    return parser.parse_args()


def main():
    """Main entry point for simulation runner"""
    args = parse_arguments()

    print("=" * 70)
    print("Thermal Terrain Simulator v3.0.0")
    print("=" * 70)
    print()

    # Load and validate configuration
    print(f"Loading configuration from: {args.config}")
    try:
        config = ThermalSimConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: Configuration file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: Invalid configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    print(f"X Configuration loaded successfully: {config.simulation.name}")
    print()

    # Override output directory if specified
    if args.output:
        config.output.directory = args.output
        print(f"Output directory overridden: {args.output}")

    # Print simulation summary
    print("Simulation Configuration Summary:")
    print("-" * 70)
    print(f"  Name:          {config.simulation.name}")
    print(f"  Start time:    {config.simulation.start_time}")
    print(f"  End time:      {config.simulation.end_time}")
    print(f"  Duration:      {config.simulation.duration_seconds / 3600:.1f} hours")
    print(f"  Time step:     {config.simulation.dt} s")
    print(f"  Grid size:     {config.terrain.nx} × {config.terrain.ny} cells")
    print(f"  Grid spacing:  {config.terrain.dx} × {config.terrain.dy} m")
    print(f"  Subsurface:    {config.subsurface.n_layers} layers to {config.subsurface.z_max} m")
    print(f"  Site:          ({config.site.latitude}°N, {config.site.longitude}°W)")
    print(f"  Material:      {config.materials.default_material}")

    if config.solver.enable_lateral_conduction:
        print(f"  Lateral conduction: ENABLED (factor={config.solver.lateral_conductivity_factor})")
    else:
        print(f"  Lateral conduction: disabled")

    print(f"  Output dir:    {config.output.directory}")
    print("-" * 70)
    print()

    # Exit if validate-only mode
    if args.validate_only:
        print("✓ Configuration validation successful (--validate-only mode)")
        print("  No simulation run. Exit.")
        sys.exit(0)

    # Create and run simulation
    print("Initializing simulation...")
    try:
        runner = SimulationRunner(config, verbose=args.verbose)
    except Exception as e:
        print(f"ERROR: Failed to initialize simulation: {e}")
        sys.exit(1)

    print("X Simulation initialized")
    print()

    # Run simulation
    print("Starting simulation...")
    print("=" * 70)
    try:
        runner.run()
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Simulation interrupted by user (Ctrl+C)")
        print("Partial results may be saved in output directory")
        sys.exit(130)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"ERROR: Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 70)
    print("✓ Simulation completed successfully")
    print(f"  Output saved to: {config.output.directory}")
    print("=" * 70)


if __name__ == "__main__":
    main()
