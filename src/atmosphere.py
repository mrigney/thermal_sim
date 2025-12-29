"""
Atmospheric Conditions Module

This module provides atmospheric state management for thermal terrain simulations,
including air temperature, wind speed, humidity, sky temperature models, and
convective heat transfer coefficient calculations.

Key Components:
- AtmosphericConditions: Main class for managing atmospheric state
- Sky temperature models (Swinbank, Brunt, Idso-Jackson)
- Convective heat transfer coefficient correlations
- Time-varying atmospheric forcing support

Author: Thermal Terrain Simulator Team
Date: December 2025
"""

import numpy as np
from typing import Union, Callable, Optional, Tuple
from datetime import datetime, timedelta


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
GAS_CONSTANT_DRY_AIR = 287.05  # J/(kg·K)
SPECIFIC_HEAT_AIR = 1005.0  # J/(kg·K) at constant pressure
GRAVITY = 9.80665  # m/s²
KARMAN_CONSTANT = 0.41  # von Karman constant


# =============================================================================
# SKY TEMPERATURE MODELS
# =============================================================================

def sky_temperature_simple(T_air: float, offset: float = 20.0) -> float:
    """
    Simple sky temperature model: T_sky = T_air - offset

    Very rough approximation suitable for quick estimates only.

    Parameters
    ----------
    T_air : float
        Air temperature [K]
    offset : float, optional
        Temperature offset [K], default 20K

    Returns
    -------
    T_sky : float
        Sky temperature [K]

    Notes
    -----
    This is an oversimplification. Use physics-based models for accuracy.
    """
    return T_air - offset


def sky_temperature_swinbank(T_air: float) -> float:
    """
    Swinbank (1963) sky temperature model

    Empirical correlation based on clear-sky measurements.

    Parameters
    ----------
    T_air : float
        Air temperature [K]

    Returns
    -------
    T_sky : float
        Sky temperature [K]

    References
    ----------
    Swinbank, W. C. (1963). Long-wave radiation from clear skies.
    Quarterly Journal of the Royal Meteorological Society, 89(381), 339-348.

    Notes
    -----
    Formula: T_sky = 0.0552 * T_air^1.5
    Valid for clear sky conditions only.
    Typical sky depression: 20-40K below air temperature.
    """
    return 0.0552 * (T_air ** 1.5)


def sky_temperature_brunt(T_air: float, vapor_pressure: float,
                         cloud_fraction: float = 0.0) -> float:
    """
    Brunt (1932) sky temperature model with cloud correction

    Physics-based model using atmospheric emissivity from water vapor.

    Parameters
    ----------
    T_air : float
        Air temperature [K]
    vapor_pressure : float
        Water vapor pressure [Pa]
    cloud_fraction : float, optional
        Cloud cover fraction [0-1], default 0 (clear sky)

    Returns
    -------
    T_sky : float
        Sky temperature [K]

    References
    ----------
    Brunt, D. (1932). Notes on radiation in the atmosphere.
    Quarterly Journal of the Royal Meteorological Society, 58(247), 389-420.

    Notes
    -----
    Clear sky emissivity: ε_clear = 0.52 + 0.065 * sqrt(e_a [mbar])
    Cloud correction: ε_cloud = ε_clear * (1 + 0.22 * N²)
    where N is cloud fraction, e_a is vapor pressure in mbar.
    """
    # Convert vapor pressure to mbar
    e_a_mbar = vapor_pressure / 100.0

    # Clear sky emissivity (Brunt formula)
    epsilon_clear = 0.52 + 0.065 * np.sqrt(e_a_mbar)

    # Cloud correction (empirical)
    epsilon_cloud = epsilon_clear * (1.0 + 0.22 * cloud_fraction**2)

    # Effective sky temperature
    T_sky = epsilon_cloud**0.25 * T_air

    return T_sky


def sky_temperature_idso_jackson(T_air: float, vapor_pressure: float,
                                 cloud_fraction: float = 0.0) -> float:
    """
    Idso-Jackson (1969) sky temperature model

    More sophisticated clear-sky emissivity model based on atmospheric
    radiation measurements in Arizona.

    Parameters
    ----------
    T_air : float
        Air temperature [K]
    vapor_pressure : float
        Water vapor pressure [Pa]
    cloud_fraction : float, optional
        Cloud cover fraction [0-1], default 0 (clear sky)

    Returns
    -------
    T_sky : float
        Sky temperature [K]

    References
    ----------
    Idso, S. B., & Jackson, R. D. (1969). Thermal radiation from the
    atmosphere. Journal of Geophysical Research, 74(23), 5397-5403.

    Notes
    -----
    ε_clear = 1 - 0.261 * exp(-7.77e-4 * (273 - T_air)²)
    More accurate for desert conditions than Brunt model.
    """
    # Temperature in Celsius
    T_celsius = T_air - 273.15

    # Idso-Jackson clear sky emissivity
    epsilon_clear = 1.0 - 0.261 * np.exp(-7.77e-4 * (273.0 - T_air)**2)

    # Cloud correction (same as Brunt)
    epsilon_cloud = epsilon_clear * (1.0 + 0.22 * cloud_fraction**2)

    # Effective sky temperature
    T_sky = epsilon_cloud**0.25 * T_air

    return T_sky


def saturation_vapor_pressure(T_air: float) -> float:
    """
    Saturation vapor pressure using Tetens formula

    Parameters
    ----------
    T_air : float
        Air temperature [K]

    Returns
    -------
    e_sat : float
        Saturation vapor pressure [Pa]

    Notes
    -----
    Tetens formula (1930):
    e_sat = 611.2 * exp(17.67 * T_C / (T_C + 243.5))
    where T_C is temperature in Celsius

    Valid range: -40°C to +50°C
    Accuracy: ±0.1% in normal atmospheric conditions
    """
    T_celsius = T_air - 273.15
    e_sat = 611.2 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))
    return e_sat


# =============================================================================
# CONVECTIVE HEAT TRANSFER COEFFICIENTS
# =============================================================================

def convection_coefficient_mcadams(wind_speed: float,
                                  characteristic_length: float = 1.0) -> float:
    """
    McAdams correlation for forced convection over flat surfaces

    Simple and widely-used correlation for turbulent flow.

    Parameters
    ----------
    wind_speed : float
        Wind speed [m/s]
    characteristic_length : float, optional
        Characteristic length scale [m], default 1.0

    Returns
    -------
    h_conv : float
        Convective heat transfer coefficient [W/(m²·K)]

    References
    ----------
    McAdams, W. H. (1954). Heat Transmission (3rd ed.). McGraw-Hill.

    Notes
    -----
    h = 5.7 + 3.8 * U  [W/(m²·K)]
    where U is wind speed in m/s

    Valid for: U > 5 m/s (forced convection dominant)
    Typical range: 10-40 W/(m²·K)
    """
    return 5.7 + 3.8 * wind_speed


def convection_coefficient_jurges(wind_speed: float) -> float:
    """
    Jurges correlation for outdoor convection

    Based on measurements of heat transfer from building surfaces.

    Parameters
    ----------
    wind_speed : float
        Wind speed [m/s]

    Returns
    -------
    h_conv : float
        Convective heat transfer coefficient [W/(m²·K)]

    References
    ----------
    Jurges, W. (1924). Der Wärmeübergang an einer ebenen Wand.
    Gesundh. Ing. Beih. Reihe 1, Nr. 19.

    Notes
    -----
    h = 2.8 + 3.0 * U  [W/(m²·K)]

    Slightly lower than McAdams, often used for building energy simulations.
    """
    return 2.8 + 3.0 * wind_speed


def convection_coefficient_watmuff(wind_speed: float,
                                  characteristic_length: float = 1.0,
                                  solar_absorptivity: float = 0.9) -> float:
    """
    Watmuff et al. (1977) correlation for solar collectors

    Developed specifically for heated horizontal surfaces in outdoor conditions.

    Parameters
    ----------
    wind_speed : float
        Wind speed [m/s]
    characteristic_length : float, optional
        Characteristic length [m], default 1.0
    solar_absorptivity : float, optional
        Surface solar absorptivity [0-1], default 0.9

    Returns
    -------
    h_conv : float
        Convective heat transfer coefficient [W/(m²·K)]

    References
    ----------
    Watmuff, J. H., Charters, W. W. S., & Proctor, D. (1977).
    Solar and wind induced external coefficients for solar collectors.
    Revue Internationale d'Héliotechnique, 2, 56.

    Notes
    -----
    h = 2.8 + 3.0 * U  for natural convection dominated (U < 1 m/s)
    h = 8.6 * U^0.6 / L^0.4  for forced convection (U > 1 m/s)

    Particularly good for desert terrain applications.
    """
    if wind_speed < 1.0:
        # Natural + weak forced convection
        return 2.8 + 3.0 * wind_speed
    else:
        # Forced convection dominant
        return 8.6 * (wind_speed**0.6) / (characteristic_length**0.4)


def wind_speed_log_profile(wind_speed_ref: float,
                           height_ref: float,
                           height: float,
                           roughness_length: float) -> float:
    """
    Logarithmic wind profile for neutral atmospheric stability

    Parameters
    ----------
    wind_speed_ref : float
        Wind speed at reference height [m/s]
    height_ref : float
        Reference height [m]
    height : float
        Desired height [m]
    roughness_length : float
        Surface roughness length [m]

    Returns
    -------
    wind_speed : float
        Wind speed at desired height [m/s]

    Notes
    -----
    U(z) / U(z_ref) = ln(z / z0) / ln(z_ref / z0)

    Typical roughness lengths:
    - Water: 0.0001 m
    - Sand, snow: 0.001 m
    - Short grass: 0.01 m
    - Long grass, crops: 0.1 m
    - Forests, urban: 1.0 m

    Valid for z > 10 * z0 (surface layer)
    """
    if height <= roughness_length or height_ref <= roughness_length:
        raise ValueError("Height must be greater than roughness length")

    return wind_speed_ref * np.log(height / roughness_length) / np.log(height_ref / roughness_length)


# =============================================================================
# ATMOSPHERIC CONDITIONS CLASS
# =============================================================================

class AtmosphericConditions:
    """
    Manages atmospheric state for thermal terrain simulations

    Provides time-varying atmospheric conditions including air temperature,
    wind speed, humidity, and derived quantities like sky temperature and
    convective heat transfer coefficients.

    Supports both constant and time-varying conditions through flexible
    input specification (constants, callables, or future file-based input).

    Parameters
    ----------
    T_air : float or callable
        Air temperature [K]. Can be:
        - float: constant temperature
        - callable: function of time returning temperature
    wind_speed : float or callable
        Wind speed [m/s] at reference height
    relative_humidity : float or callable, optional
        Relative humidity [0-1], default 0.3
    reference_height : float, optional
        Reference height for wind measurements [m], default 2.0
    cloud_fraction : float or callable, optional
        Cloud cover fraction [0-1], default 0.0 (clear sky)

    Attributes
    ----------
    reference_height : float
        Height of wind speed measurements [m]

    Examples
    --------
    >>> # Constant conditions
    >>> atm = AtmosphericConditions(T_air=300.0, wind_speed=3.0)
    >>> T = atm.get_air_temperature(datetime.now())

    >>> # Time-varying temperature with diurnal cycle
    >>> def T_diurnal(t):
    ...     hour = t.hour + t.minute/60.0
    ...     return 295.0 + 10.0 * np.sin(np.pi * (hour - 6) / 12)
    >>> atm = AtmosphericConditions(T_air=T_diurnal, wind_speed=3.0)
    """

    def __init__(self,
                 T_air: Union[float, Callable],
                 wind_speed: Union[float, Callable],
                 relative_humidity: Union[float, Callable] = 0.3,
                 reference_height: float = 2.0,
                 cloud_fraction: Union[float, Callable] = 0.0):

        self.T_air_spec = T_air
        self.wind_speed_spec = wind_speed
        self.relative_humidity_spec = relative_humidity
        self.cloud_fraction_spec = cloud_fraction
        self.reference_height = reference_height

    def _evaluate(self, spec: Union[float, Callable], time: datetime) -> float:
        """
        Evaluate a specification (constant or callable) at given time

        Parameters
        ----------
        spec : float or callable
            Value specification
        time : datetime
            Time at which to evaluate

        Returns
        -------
        value : float
            Evaluated value
        """
        if callable(spec):
            return spec(time)
        else:
            return float(spec)

    def get_air_temperature(self, time: datetime) -> float:
        """
        Get air temperature at specified time

        Parameters
        ----------
        time : datetime
            Time of interest

        Returns
        -------
        T_air : float
            Air temperature [K]
        """
        return self._evaluate(self.T_air_spec, time)

    def get_wind_speed(self, time: datetime, height: Optional[float] = None) -> float:
        """
        Get wind speed at specified time and height

        Parameters
        ----------
        time : datetime
            Time of interest
        height : float, optional
            Height above ground [m]. If None, returns reference height value.

        Returns
        -------
        wind_speed : float
            Wind speed [m/s]

        Notes
        -----
        If height is specified and differs from reference height, logarithmic
        wind profile is used for extrapolation (requires roughness_length).
        """
        U_ref = self._evaluate(self.wind_speed_spec, time)

        if height is None or height == self.reference_height:
            return U_ref
        else:
            # Would need roughness_length to extrapolate
            # For now, just return reference value
            # TODO: Add height adjustment when roughness is available
            return U_ref

    def get_relative_humidity(self, time: datetime) -> float:
        """
        Get relative humidity at specified time

        Parameters
        ----------
        time : datetime
            Time of interest

        Returns
        -------
        RH : float
            Relative humidity [0-1]
        """
        return self._evaluate(self.relative_humidity_spec, time)

    def get_cloud_fraction(self, time: datetime) -> float:
        """
        Get cloud cover fraction at specified time

        Parameters
        ----------
        time : datetime
            Time of interest

        Returns
        -------
        N : float
            Cloud fraction [0-1]
        """
        return self._evaluate(self.cloud_fraction_spec, time)

    def get_vapor_pressure(self, time: datetime) -> float:
        """
        Calculate water vapor pressure from temperature and relative humidity

        Parameters
        ----------
        time : datetime
            Time of interest

        Returns
        -------
        e_a : float
            Actual vapor pressure [Pa]

        Notes
        -----
        e_a = RH * e_sat(T)
        where e_sat is saturation vapor pressure
        """
        T_air = self.get_air_temperature(time)
        RH = self.get_relative_humidity(time)
        e_sat = saturation_vapor_pressure(T_air)
        return RH * e_sat

    def get_sky_temperature(self, time: datetime,
                           model: str = 'brunt') -> float:
        """
        Calculate effective sky temperature for longwave radiation

        Parameters
        ----------
        time : datetime
            Time of interest
        model : str, optional
            Sky temperature model to use. Options:
            - 'simple': T_sky = T_air - 20K
            - 'swinbank': Swinbank (1963) correlation
            - 'brunt': Brunt (1932) with vapor pressure
            - 'idso': Idso-Jackson (1969), best for deserts
            Default is 'brunt'.

        Returns
        -------
        T_sky : float
            Effective sky temperature [K]

        Notes
        -----
        Sky temperature is critical for nighttime radiative cooling.
        Clear sky is typically 20-40K colder than air temperature.
        """
        T_air = self.get_air_temperature(time)

        if model == 'simple':
            return sky_temperature_simple(T_air)

        elif model == 'swinbank':
            return sky_temperature_swinbank(T_air)

        elif model in ['brunt', 'idso']:
            e_a = self.get_vapor_pressure(time)
            N = self.get_cloud_fraction(time)

            if model == 'brunt':
                return sky_temperature_brunt(T_air, e_a, N)
            else:  # idso
                return sky_temperature_idso_jackson(T_air, e_a, N)

        else:
            raise ValueError(f"Unknown sky temperature model: {model}")

    def get_convection_coefficient(self,
                                   time: datetime,
                                   surface_roughness: float = 0.01,
                                   correlation: str = 'watmuff') -> float:
        """
        Calculate convective heat transfer coefficient

        Parameters
        ----------
        time : datetime
            Time of interest
        surface_roughness : float, optional
            Surface roughness length [m], default 0.01 (short grass)
        correlation : str, optional
            Correlation to use. Options:
            - 'mcadams': McAdams (1954), simple and robust
            - 'jurges': Jurges (1924), for building surfaces
            - 'watmuff': Watmuff et al. (1977), for solar collectors
            Default is 'watmuff' (best for terrain).

        Returns
        -------
        h_conv : float
            Convective heat transfer coefficient [W/(m²·K)]

        Notes
        -----
        Coefficient depends on wind speed and surface characteristics.
        Typical range: 5-40 W/(m²·K)
        Used in: Q_conv = h_conv * (T_air - T_surface)
        """
        U = self.get_wind_speed(time)

        if correlation == 'mcadams':
            return convection_coefficient_mcadams(U)
        elif correlation == 'jurges':
            return convection_coefficient_jurges(U)
        elif correlation == 'watmuff':
            return convection_coefficient_watmuff(U, characteristic_length=1.0)
        else:
            raise ValueError(f"Unknown convection correlation: {correlation}")

    def get_atmospheric_state(self, time: datetime) -> dict:
        """
        Get complete atmospheric state at specified time

        Parameters
        ----------
        time : datetime
            Time of interest

        Returns
        -------
        state : dict
            Dictionary containing:
            - 'T_air': Air temperature [K]
            - 'wind_speed': Wind speed [m/s]
            - 'RH': Relative humidity [0-1]
            - 'cloud_fraction': Cloud cover [0-1]
            - 'vapor_pressure': Water vapor pressure [Pa]
            - 'T_sky_brunt': Sky temperature (Brunt model) [K]
            - 'T_sky_idso': Sky temperature (Idso model) [K]
            - 'h_conv': Convective coefficient [W/(m²·K)]
        """
        return {
            'T_air': self.get_air_temperature(time),
            'wind_speed': self.get_wind_speed(time),
            'RH': self.get_relative_humidity(time),
            'cloud_fraction': self.get_cloud_fraction(time),
            'vapor_pressure': self.get_vapor_pressure(time),
            'T_sky_brunt': self.get_sky_temperature(time, model='brunt'),
            'T_sky_idso': self.get_sky_temperature(time, model='idso'),
            'h_conv': self.get_convection_coefficient(time),
        }


# =============================================================================
# HELPER FUNCTIONS FOR DIURNAL VARIATIONS
# =============================================================================

def create_diurnal_temperature(T_mean: float, T_amplitude: float,
                               sunrise_hour: float = 6.0,
                               sunset_hour: float = 18.0) -> Callable:
    """
    Create a simple diurnal temperature variation function

    Parameters
    ----------
    T_mean : float
        Daily mean temperature [K]
    T_amplitude : float
        Temperature amplitude (peak-to-peak / 2) [K]
    sunrise_hour : float, optional
        Hour of sunrise (local time), default 6.0
    sunset_hour : float, optional
        Hour of sunset (local time), default 18.0

    Returns
    -------
    T_func : callable
        Function that takes datetime and returns temperature [K]

    Notes
    -----
    Simple sinusoidal variation with:
    - Minimum at sunrise
    - Maximum ~2-3 hours after solar noon

    T(t) = T_mean + T_amp * sin(π * (hour - sunrise) / 12 - π/2)
    """
    solar_noon = (sunrise_hour + sunset_hour) / 2.0

    def T_diurnal(time: datetime) -> float:
        hour = time.hour + time.minute / 60.0
        # Phase shifted so minimum is at sunrise
        phase = np.pi * (hour - sunrise_hour) / 12.0 - np.pi / 2.0
        return T_mean + T_amplitude * np.sin(phase)

    return T_diurnal


def create_diurnal_wind(wind_mean: float, wind_amplitude: float = None) -> Callable:
    """
    Create a simple diurnal wind speed variation function

    Parameters
    ----------
    wind_mean : float
        Daily mean wind speed [m/s]
    wind_amplitude : float, optional
        Wind speed amplitude [m/s]. If None, uses 0.3 * wind_mean

    Returns
    -------
    wind_func : callable
        Function that takes datetime and returns wind speed [m/s]

    Notes
    -----
    Simple variation with:
    - Minimum wind at sunrise (stable boundary layer)
    - Maximum wind in afternoon (convective boundary layer)

    Ensures wind speed is always positive.
    """
    if wind_amplitude is None:
        wind_amplitude = 0.3 * wind_mean

    def wind_diurnal(time: datetime) -> float:
        hour = time.hour + time.minute / 60.0
        # Maximum around 14:00, minimum around 6:00
        phase = np.pi * (hour - 6.0) / 12.0 - np.pi / 2.0
        wind = wind_mean + wind_amplitude * np.sin(phase)
        return max(0.1, wind)  # Ensure positive

    return wind_diurnal
