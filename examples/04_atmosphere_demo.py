"""
Example 04: Atmospheric Conditions Demo

Demonstrates the atmosphere module capabilities:
- Sky temperature models (Swinbank, Brunt, Idso-Jackson)
- Convective heat transfer coefficients
- Diurnal variations in temperature and wind
- Humidity effects on sky temperature
- Complete atmospheric state over time

Usage:
    python 04_atmosphere_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.atmosphere import (
    AtmosphericConditions,
    sky_temperature_simple, sky_temperature_swinbank,
    sky_temperature_brunt, sky_temperature_idso_jackson,
    saturation_vapor_pressure,
    convection_coefficient_mcadams, convection_coefficient_jurges,
    convection_coefficient_watmuff,
    create_diurnal_temperature, create_diurnal_wind
)


def main():
    print("=" * 80)
    print("Thermal Terrain Simulator - Atmospheric Conditions Demo")
    print("=" * 80)
    print()

    demo_start = time.time()

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # 1. SKY TEMPERATURE MODEL COMPARISON
    # =========================================================================
    print("1. Sky Temperature Model Comparison")
    print("-" * 80)
    t_start = time.time()

    # Test conditions
    T_air_test = 300.0  # K (27°C)
    RH_values = np.linspace(0.1, 0.9, 9)

    print(f"Air temperature: {T_air_test:.1f} K ({T_air_test-273.15:.1f}°C)")
    print(f"Testing relative humidity range: {RH_values[0]:.0%} to {RH_values[-1]:.0%}")
    print()

    # Calculate sky temperatures for each model
    T_sky_simple = []
    T_sky_swinbank = []
    T_sky_brunt = []
    T_sky_idso = []

    for RH in RH_values:
        # Simple model (no humidity dependence)
        T_sky_simple.append(sky_temperature_simple(T_air_test))

        # Swinbank (no humidity dependence)
        T_sky_swinbank.append(sky_temperature_swinbank(T_air_test))

        # Brunt (humidity dependent)
        e_sat = saturation_vapor_pressure(T_air_test)
        e_a = RH * e_sat
        T_sky_brunt.append(sky_temperature_brunt(T_air_test, e_a, cloud_fraction=0.0))

        # Idso-Jackson (weak humidity dependence through temperature)
        T_sky_idso.append(sky_temperature_idso_jackson(T_air_test, e_a, cloud_fraction=0.0))

    # Print comparison table
    print("Sky Temperature Comparison:")
    print(f"{'RH':<8} {'Simple':<10} {'Swinbank':<12} {'Brunt':<10} {'Idso':<10} {'Brunt ΔT':<10}")
    print("-" * 70)
    for i, RH in enumerate(RH_values):
        dT_brunt = T_air_test - T_sky_brunt[i]
        print(f"{RH*100:5.0f}%   {T_sky_simple[i]:6.1f} K   {T_sky_swinbank[i]:6.1f} K    "
              f"{T_sky_brunt[i]:6.1f} K  {T_sky_idso[i]:6.1f} K  {dT_brunt:6.1f} K")
    print()

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(RH_values * 100, T_sky_simple, 'k--', label='Simple', linewidth=2)
    ax1.plot(RH_values * 100, T_sky_swinbank, 'b-', label='Swinbank', linewidth=2)
    ax1.plot(RH_values * 100, T_sky_brunt, 'r-', label='Brunt', linewidth=2)
    ax1.plot(RH_values * 100, T_sky_idso, 'g-', label='Idso-Jackson', linewidth=2)
    ax1.axhline(T_air_test, color='gray', linestyle=':', alpha=0.5, label='Air Temp')
    ax1.set_xlabel('Relative Humidity (%)')
    ax1.set_ylabel('Sky Temperature (K)')
    ax1.set_title(f'Sky Temperature Models (T_air = {T_air_test:.0f}K)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sky depression (T_air - T_sky)
    depression_simple = [T_air_test - T for T in T_sky_simple]
    depression_swinbank = [T_air_test - T for T in T_sky_swinbank]
    depression_brunt = [T_air_test - T for T in T_sky_brunt]
    depression_idso = [T_air_test - T for T in T_sky_idso]

    ax2.plot(RH_values * 100, depression_simple, 'k--', label='Simple', linewidth=2)
    ax2.plot(RH_values * 100, depression_swinbank, 'b-', label='Swinbank', linewidth=2)
    ax2.plot(RH_values * 100, depression_brunt, 'r-', label='Brunt', linewidth=2)
    ax2.plot(RH_values * 100, depression_idso, 'g-', label='Idso-Jackson', linewidth=2)
    ax2.set_xlabel('Relative Humidity (%)')
    ax2.set_ylabel('Sky Depression (K)')
    ax2.set_title('T_air - T_sky vs Humidity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sky_temperature_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: sky_temperature_comparison.png")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 2. CONVECTIVE COEFFICIENT COMPARISON
    # =========================================================================
    print("2. Convective Heat Transfer Coefficients")
    print("-" * 80)
    t_start = time.time()

    # Wind speed range
    wind_speeds = np.linspace(0, 15, 100)

    # Calculate h_conv for each correlation
    h_mcadams = [convection_coefficient_mcadams(U) for U in wind_speeds]
    h_jurges = [convection_coefficient_jurges(U) for U in wind_speeds]
    h_watmuff = [convection_coefficient_watmuff(U, characteristic_length=1.0) for U in wind_speeds]

    # Print typical values
    print("Convective Coefficient h_conv [W/(m²·K)]:")
    print(f"{'Wind Speed':<12} {'McAdams':<12} {'Jurges':<12} {'Watmuff':<12}")
    print("-" * 50)
    for U in [0, 1, 2, 5, 10, 15]:
        idx = np.argmin(np.abs(wind_speeds - U))
        print(f"{U:8.0f} m/s   {h_mcadams[idx]:8.1f}     {h_jurges[idx]:8.1f}     {h_watmuff[idx]:8.1f}")
    print()

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wind_speeds, h_mcadams, 'b-', label='McAdams (1954)', linewidth=2)
    ax.plot(wind_speeds, h_jurges, 'r-', label='Jurges (1924)', linewidth=2)
    ax.plot(wind_speeds, h_watmuff, 'g-', label='Watmuff et al. (1977)', linewidth=2)
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('h_conv [W/(m²·K)]')
    ax.set_title('Convective Heat Transfer Coefficient Correlations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 70)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convection_coefficients.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: convection_coefficients.png")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 3. DIURNAL ATMOSPHERIC VARIATIONS
    # =========================================================================
    print("3. Diurnal Atmospheric Variations")
    print("-" * 80)
    t_start = time.time()

    # Create diurnal temperature and wind functions
    T_mean = 300.0  # K (27°C)
    T_amplitude = 12.0  # K (24K peak-to-peak swing)
    wind_mean = 5.0  # m/s
    wind_amplitude = 2.0  # m/s

    T_diurnal = create_diurnal_temperature(T_mean, T_amplitude, sunrise_hour=6.0, sunset_hour=18.0)
    wind_diurnal = create_diurnal_wind(wind_mean, wind_amplitude)

    print(f"Mean air temperature: {T_mean:.1f} K ({T_mean-273.15:.1f}°C)")
    print(f"Temperature amplitude: {T_amplitude:.1f} K")
    print(f"Daily range: {T_mean-T_amplitude:.1f}K to {T_mean+T_amplitude:.1f}K")
    print(f"Mean wind speed: {wind_mean:.1f} m/s")
    print(f"Wind amplitude: {wind_amplitude:.1f} m/s")
    print()

    # Create atmospheric conditions object
    atmosphere = AtmosphericConditions(
        T_air=T_diurnal,
        wind_speed=wind_diurnal,
        relative_humidity=0.3,  # Constant 30% RH (typical desert)
        cloud_fraction=0.0  # Clear sky
    )

    # Simulate 24 hours
    start_date = datetime(2025, 6, 21, 0, 0, 0)
    times = []
    hours = []
    T_air_values = []
    wind_values = []
    T_sky_brunt_values = []
    T_sky_idso_values = []
    h_conv_values = []

    for hour in range(24):
        current_time = start_date + timedelta(hours=hour)
        times.append(current_time)
        hours.append(hour)

        state = atmosphere.get_atmospheric_state(current_time)
        T_air_values.append(state['T_air'])
        wind_values.append(state['wind_speed'])
        T_sky_brunt_values.append(state['T_sky_brunt'])
        T_sky_idso_values.append(state['T_sky_idso'])
        h_conv_values.append(state['h_conv'])

    # Print summary
    print(f"24-hour simulation:")
    print(f"  T_air range: {min(T_air_values):.1f} to {max(T_air_values):.1f} K")
    print(f"  Wind range: {min(wind_values):.1f} to {max(wind_values):.1f} m/s")
    print(f"  T_sky (Brunt) range: {min(T_sky_brunt_values):.1f} to {max(T_sky_brunt_values):.1f} K")
    print(f"  h_conv range: {min(h_conv_values):.1f} to {max(h_conv_values):.1f} W/(m²·K)")
    print()

    # Plot diurnal variations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Temperature
    ax1.plot(hours, T_air_values, 'r-', linewidth=2, label='Air Temperature')
    ax1.plot(hours, T_sky_brunt_values, 'b-', linewidth=2, label='Sky Temp (Brunt)')
    ax1.plot(hours, T_sky_idso_values, 'g--', linewidth=2, label='Sky Temp (Idso)')
    ax1.set_xlabel('Hour (UTC)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Air and Sky Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 23)

    # Sky depression
    depression_brunt = [T_air - T_sky for T_air, T_sky in zip(T_air_values, T_sky_brunt_values)]
    depression_idso = [T_air - T_sky for T_air, T_sky in zip(T_air_values, T_sky_idso_values)]
    ax2.plot(hours, depression_brunt, 'b-', linewidth=2, label='Brunt Model')
    ax2.plot(hours, depression_idso, 'g--', linewidth=2, label='Idso Model')
    ax2.set_xlabel('Hour (UTC)')
    ax2.set_ylabel('Sky Depression (K)')
    ax2.set_title('T_air - T_sky Throughout Day')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 23)

    # Wind speed
    ax3.plot(hours, wind_values, 'cyan', linewidth=2)
    ax3.set_xlabel('Hour (UTC)')
    ax3.set_ylabel('Wind Speed (m/s)')
    ax3.set_title('Wind Speed')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 23)

    # Convective coefficient
    ax4.plot(hours, h_conv_values, 'orange', linewidth=2)
    ax4.set_xlabel('Hour (UTC)')
    ax4.set_ylabel('h_conv [W/(m²·K)]')
    ax4.set_title('Convective Heat Transfer Coefficient')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 23)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'atmospheric_diurnal.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: atmospheric_diurnal.png")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 4. HUMIDITY EFFECTS ON LONGWAVE RADIATION
    # =========================================================================
    print("4. Humidity Effects on Longwave Radiation")
    print("-" * 80)
    t_start = time.time()

    # Calculate downward longwave radiation for different humidity levels
    RH_test = [0.1, 0.3, 0.5, 0.7, 0.9]
    epsilon_surface = 0.95  # Typical terrain emissivity
    SVF = 1.0  # Open sky
    SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant

    print(f"Downward longwave radiation from sky:")
    print(f"(Assuming ε_surface = {epsilon_surface:.2f}, SVF = {SVF:.1f})")
    print()
    print(f"{'RH':<8} {'T_sky (Brunt)':<15} {'L↓ (W/m²)':<12} {'% of Blackbody':<15}")
    print("-" * 55)

    for RH in RH_test:
        e_sat = saturation_vapor_pressure(T_air_test)
        e_a = RH * e_sat
        T_sky = sky_temperature_brunt(T_air_test, e_a, cloud_fraction=0.0)

        # Downward longwave radiation
        L_down = epsilon_surface * SIGMA * SVF * T_sky**4

        # Compare to blackbody at air temperature
        L_blackbody = epsilon_surface * SIGMA * SVF * T_air_test**4
        percent_blackbody = 100.0 * L_down / L_blackbody

        print(f"{RH*100:5.0f}%   {T_sky:8.1f} K      {L_down:8.1f}     {percent_blackbody:8.1f}%")

    print()
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # 5. CLOUD COVER EFFECTS
    # =========================================================================
    print("5. Cloud Cover Effects on Sky Temperature")
    print("-" * 80)
    t_start = time.time()

    cloud_fractions = np.linspace(0, 1, 11)
    T_air_cloud = 300.0  # K
    RH_cloud = 0.5  # 50% humidity
    e_sat_cloud = saturation_vapor_pressure(T_air_cloud)
    e_a_cloud = RH_cloud * e_sat_cloud

    T_sky_clouds = []
    for N in cloud_fractions:
        T_sky = sky_temperature_brunt(T_air_cloud, e_a_cloud, cloud_fraction=N)
        T_sky_clouds.append(T_sky)

    print(f"Cloud cover effects (T_air = {T_air_cloud:.1f}K, RH = {RH_cloud*100:.0f}%):")
    print(f"{'Cloud Cover':<15} {'T_sky':<10} {'ΔT (air-sky)':<12}")
    print("-" * 40)
    for N, T_sky in zip(cloud_fractions, T_sky_clouds):
        dT = T_air_cloud - T_sky
        print(f"{N*100:8.0f}%        {T_sky:6.1f} K   {dT:6.1f} K")
    print()

    # Plot cloud effects
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cloud_fractions * 100, T_sky_clouds, 'b-', linewidth=2, marker='o', markersize=6)
    ax.axhline(T_air_cloud, color='r', linestyle='--', linewidth=2, label='Air Temperature')
    ax.set_xlabel('Cloud Cover (%)')
    ax.set_ylabel('Sky Temperature (K)')
    ax.set_title(f'Cloud Cover Effect on Sky Temperature (T_air = {T_air_cloud:.0f}K, RH = {RH_cloud*100:.0f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cloud_cover_effects.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: cloud_cover_effects.png")
    print(f"[Timing: {time.time() - t_start:.3f}s]")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    demo_time = time.time() - demo_start

    print("=" * 80)
    print("Atmospheric Conditions Demo Complete!")
    print("=" * 80)
    print(f"Total execution time: {demo_time:.2f}s")
    print()
    print("Generated output files:")
    print("  - sky_temperature_comparison.png   (sky temp models vs humidity)")
    print("  - convection_coefficients.png      (h_conv correlations)")
    print("  - atmospheric_diurnal.png          (24-hour variations)")
    print("  - cloud_cover_effects.png          (cloud impact on sky temp)")
    print()
    print("Atmosphere module capabilities demonstrated:")
    print("  [+] Sky temperature models (Simple, Swinbank, Brunt, Idso-Jackson)")
    print("  [+] Convective heat transfer coefficients (McAdams, Jurges, Watmuff)")
    print("  [+] Humidity and vapor pressure calculations")
    print("  [+] Time-varying atmospheric conditions")
    print("  [+] Cloud cover effects")
    print("  [+] Complete atmospheric state management")
    print()


if __name__ == '__main__':
    main()
