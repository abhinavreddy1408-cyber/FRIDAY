"""
F.R.I.D.A.Y. Weather Skill
Provides current weather conditions, forecasts, and weather alerts
through voice commands using the Open-Meteo API (free, no API key required)
with optional OpenWeatherMap integration for enhanced data.

Supported commands:
  - "what's the weather"
  - "weather in [city]"
  - "weather today / tomorrow / this week"
  - "will it rain today"
  - "what's the temperature"
  - "weather forecast"
  - "is it going to be hot / cold / sunny / rainy"
  - "what should I wear today"
  - "weather alerts"

Uses IP-based geolocation as a fallback when no location is specified.
All data is cached to avoid redundant API calls.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from friday.actions import Action, ActionResult, ActionStatus

if TYPE_CHECKING:
    import re

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_CACHE_TTL = 600  # seconds (10 minutes)
_REQUEST_TIMEOUT = 10  # seconds
_DEFAULT_LOCATION = "London"

# Open-Meteo (free, no key required)
_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# IP geolocation (free)
_GEOIP_URL = "https://ipapi.co/json/"


# ── Enums ─────────────────────────────────────────────────────────────────────


class WeatherStatus(Enum):
    SUCCESS = auto()
    FETCH_FAILED = auto()
    PARSE_FAILED = auto()
    LOCATION_ERROR = auto()
    CACHE_HIT = auto()
    UNAVAILABLE = auto()


class WeatherCondition(Enum):
    """WMO weather interpretation codes mapped to friendly conditions."""

    CLEAR = auto()
    PARTLY_CLOUDY = auto()
    CLOUDY = auto()
    FOG = auto()
    DRIZZLE = auto()
    RAIN = auto()
    SNOW = auto()
    THUNDERSTORM = auto()
    UNKNOWN = auto()


# ── WMO Code Mapping ──────────────────────────────────────────────────────────

_WMO_CONDITIONS: dict[int, tuple[WeatherCondition, str]] = {
    0: (WeatherCondition.CLEAR, "clear skies"),
    1: (WeatherCondition.CLEAR, "mainly clear"),
    2: (WeatherCondition.PARTLY_CLOUDY, "partly cloudy"),
    3: (WeatherCondition.CLOUDY, "overcast"),
    45: (WeatherCondition.FOG, "foggy"),
    48: (WeatherCondition.FOG, "icy fog"),
    51: (WeatherCondition.DRIZZLE, "light drizzle"),
    53: (WeatherCondition.DRIZZLE, "moderate drizzle"),
    55: (WeatherCondition.DRIZZLE, "heavy drizzle"),
    61: (WeatherCondition.RAIN, "light rain"),
    63: (WeatherCondition.RAIN, "moderate rain"),
    65: (WeatherCondition.RAIN, "heavy rain"),
    71: (WeatherCondition.SNOW, "light snow"),
    73: (WeatherCondition.SNOW, "moderate snow"),
    75: (WeatherCondition.SNOW, "heavy snow"),
    77: (WeatherCondition.SNOW, "snow grains"),
    80: (WeatherCondition.RAIN, "light showers"),
    81: (WeatherCondition.RAIN, "moderate showers"),
    82: (WeatherCondition.RAIN, "violent showers"),
    85: (WeatherCondition.SNOW, "snow showers"),
    86: (WeatherCondition.SNOW, "heavy snow showers"),
    95: (WeatherCondition.THUNDERSTORM, "thunderstorm"),
    96: (WeatherCondition.THUNDERSTORM, "thunderstorm with hail"),
    99: (WeatherCondition.THUNDERSTORM, "severe thunderstorm"),
}


def _decode_wmo(code: int) -> tuple[WeatherCondition, str]:
    """Return the condition enum and description for a WMO code."""
    return _WMO_CONDITIONS.get(code, (WeatherCondition.UNKNOWN, "unknown conditions"))


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Location:
    """
    Geographic location resolved from a place name or IP address.

    Attributes:
        name:      Display name (city, country).
        latitude:  WGS84 latitude.
        longitude: WGS84 longitude.
        timezone:  IANA timezone string.
        country:   Country name.
    """

    name: str
    latitude: float
    longitude: float
    timezone: str
    country: str

    def __str__(self) -> str:
        return f"{self.name}, {self.country}" if self.country else self.name


@dataclass(frozen=True)
class CurrentWeather:
    """
    Current weather conditions at a location.

    Attributes:
        location:       The resolved location.
        temperature_c:  Temperature in Celsius.
        feels_like_c:   Apparent temperature in Celsius.
        humidity_pct:   Relative humidity percentage.
        wind_speed_kph: Wind speed in km/h.
        wind_dir:       Wind direction in degrees.
        condition:      Classified weather condition.
        description:    Human-readable condition description.
        uv_index:       UV index (0-11+).
        visibility_km:  Visibility in kilometres.
        fetched_at:     Unix timestamp of data retrieval.
    """

    location: Location
    temperature_c: float
    feels_like_c: float
    humidity_pct: float
    wind_speed_kph: float
    wind_dir: float
    condition: WeatherCondition
    description: str
    uv_index: float
    visibility_km: float
    fetched_at: float

    @property
    def temperature_f(self) -> float:
        return round(self.temperature_c * 9 / 5 + 32, 1)

    @property
    def is_cold(self) -> bool:
        return self.temperature_c < 10

    @property
    def is_hot(self) -> bool:
        return self.temperature_c > 28

    @property
    def is_raining(self) -> bool:
        return self.condition in (
            WeatherCondition.RAIN,
            WeatherCondition.DRIZZLE,
            WeatherCondition.THUNDERSTORM,
        )

    def to_speech(self) -> str:
        """Format as a natural TTS-ready weather report."""
        parts = [
            f"Currently in {self.location.name}:",
            f"{self.description}, {self.temperature_c:.0f} degrees Celsius",
            f"({self.temperature_f:.0f} Fahrenheit).",
            f"Feels like {self.feels_like_c:.0f} degrees.",
            f"Humidity {self.humidity_pct:.0f} percent.",
            f"Wind at {self.wind_speed_kph:.0f} kilometres per hour.",
        ]
        return " ".join(parts)

    def clothing_advice(self) -> str:
        """Return a clothing recommendation based on current conditions."""
        if self.temperature_c < 0:
            base = "heavy winter coat, gloves, and a hat"
        elif self.temperature_c < 10:
            base = "a warm coat and layers"
        elif self.temperature_c < 16:
            base = "a jacket or light coat"
        elif self.temperature_c < 22:
            base = "a light jacket or jumper"
        else:
            base = "light, comfortable clothing"

        rain_advice = " Bring an umbrella — rain is expected." if self.is_raining else ""
        uv_advice = " Don't forget sunscreen — UV index is high." if self.uv_index >= 6 else ""

        return f"I'd suggest {base}.{rain_advice}{uv_advice}"


@dataclass(frozen=True)
class DailyForecast:
    """
    Single-day weather forecast.

    Attributes:
        date:          Date string (YYYY-MM-DD).
        max_temp_c:    Maximum temperature in Celsius.
        min_temp_c:    Minimum temperature in Celsius.
        condition:     Dominant weather condition.
        description:   Human-readable condition string.
        precipitation_mm: Total precipitation in mm.
        wind_max_kph:  Maximum wind speed in km/h.
        uv_index:      Peak UV index.
        sunrise:       Sunrise time string (HH:MM).
        sunset:        Sunset time string (HH:MM).
    """

    date: str
    max_temp_c: float
    min_temp_c: float
    condition: WeatherCondition
    description: str
    precipitation_mm: float
    wind_max_kph: float
    uv_index: float
    sunrise: str
    sunset: str

    def to_speech(self, label: str = "") -> str:
        """Format as a TTS-ready daily summary."""
        prefix = f"{label}: " if label else ""
        rain = f", {self.precipitation_mm:.0f}mm of rain" if self.precipitation_mm > 0 else ""
        return (
            f"{prefix}{self.description}, "
            f"high of {self.max_temp_c:.0f}, low of {self.min_temp_c:.0f} degrees{rain}."
        )


@dataclass
class WeatherCache:
    """
    In-memory cache entry for weather data.

    Attributes:
        data:       Cached data payload.
        fetched_at: Unix timestamp of cache population.
        location:   Location string used as cache key.
    """

    data: Any
    fetched_at: float
    location: str

    def is_valid(self, ttl: float = _CACHE_TTL) -> bool:
        return (time.time() - self.fetched_at) < ttl


# ── HTTP Utilities ────────────────────────────────────────────────────────────


def _fetch_json(url: str, params: dict | None = None) -> dict | None:
    """
    Perform a GET request and return the parsed JSON response.

    Args:
        url:    Base URL.
        params: Optional query parameters dict.

    Returns:
        Parsed JSON dict or None on failure.
    """
    if params:
        url = f"{url}?{urlencode(params)}"

    try:
        req = Request(url, headers={"User-Agent": "FRIDAY-Assistant/1.0"})
        with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        logger.warning("[Weather] Network error fetching '%s': %s", url, exc)
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("[Weather] JSON parse error from '%s': %s", url, exc)
        return None
    except Exception as exc:
        logger.warning("[Weather] Unexpected fetch error: %s", exc)
        return None


# ── Geocoder ──────────────────────────────────────────────────────────────────


class _Geocoder:
    """
    Resolves place names and IP addresses to Location objects.
    Uses Open-Meteo's geocoding API and ipapi.co for IP geolocation.
    """

    _cache: dict[str, Location] = {}  # Module-level cache across instances  # noqa: RUF012

    def resolve(self, place_name: str) -> Location | None:
        """
        Resolve a place name to a Location.

        Args:
            place_name: City name, "auto" for IP-based, or "me" for current.

        Returns:
            Location object or None if resolution fails.
        """
        key = place_name.lower().strip()

        if key in self._cache:
            return self._cache[key]

        if key in ("auto", "me", "here", "my location"):
            location = self._from_ip()
        else:
            location = self._from_name(place_name)

        if location:
            self._cache[key] = location

        return location

    def _from_name(self, name: str) -> Location | None:
        """Geocode a place name using Open-Meteo geocoding."""
        data = _fetch_json(
            _GEOCODING_URL,
            {
                "name": name,
                "count": 1,
                "language": "en",
                "format": "json",
            },
        )

        if not data or not data.get("results"):
            logger.warning("[Weather] Geocoding failed for '%s'.", name)
            return None

        result = data["results"][0]
        return Location(
            name=result.get("name", name),
            latitude=float(result["latitude"]),
            longitude=float(result["longitude"]),
            timezone=result.get("timezone", "UTC"),
            country=result.get("country", ""),
        )

    def _from_ip(self) -> Location | None:
        """Determine location from the public IP address."""
        data = _fetch_json(_GEOIP_URL)
        if not data or "latitude" not in data:
            logger.warning("[Weather] IP geolocation failed.")
            return None

        return Location(
            name=data.get("city", "Unknown"),
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            timezone=data.get("timezone", "UTC"),
            country=data.get("country_name", ""),
        )


# ── Weather Fetcher ───────────────────────────────────────────────────────────


class _WeatherFetcher:
    """
    Fetches and parses weather data from the Open-Meteo API.
    Returns typed CurrentWeather and DailyForecast objects.
    """

    # Variables to request from the API
    _CURRENT_VARS = [  # noqa: RUF012
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "weather_code",
        "uv_index",
        "visibility",
    ]

    _DAILY_VARS = [  # noqa: RUF012
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "uv_index_max",
        "sunrise",
        "sunset",
    ]

    def fetch_current(self, location: Location) -> CurrentWeather | None:
        """
        Fetch current weather conditions for a location.

        Args:
            location: Resolved Location object.

        Returns:
            CurrentWeather or None on failure.
        """
        data = _fetch_json(
            _WEATHER_URL,
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": ",".join(self._CURRENT_VARS),
                "timezone": location.timezone,
                "wind_speed_unit": "kmh",
            },
        )

        if not data or "current" not in data:
            return None

        try:
            return self._parse_current(data["current"], location)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("[Weather] Current weather parse failed: %s", exc)
            return None

    def fetch_forecast(
        self,
        location: Location,
        days: int = 7,
    ) -> list[DailyForecast]:
        """
        Fetch a daily forecast for the next N days.

        Args:
            location: Resolved Location object.
            days:     Number of forecast days (1-16).

        Returns:
            List of DailyForecast objects.
        """
        data = _fetch_json(
            _WEATHER_URL,
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "daily": ",".join(self._DAILY_VARS),
                "timezone": location.timezone,
                "forecast_days": min(days, 16),
                "wind_speed_unit": "kmh",
            },
        )

        if not data or "daily" not in data:
            return []

        try:
            return self._parse_daily(data["daily"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            logger.warning("[Weather] Forecast parse failed: %s", exc)
            return []

    @staticmethod
    def _parse_current(current: dict, location: Location) -> CurrentWeather:
        """Parse the current weather dict into a CurrentWeather object."""
        wmo_code = int(current.get("weather_code", 0))
        condition, description = _decode_wmo(wmo_code)

        return CurrentWeather(
            location=location,
            temperature_c=float(current.get("temperature_2m", 0)),
            feels_like_c=float(current.get("apparent_temperature", 0)),
            humidity_pct=float(current.get("relative_humidity_2m", 0)),
            wind_speed_kph=float(current.get("wind_speed_10m", 0)),
            wind_dir=float(current.get("wind_direction_10m", 0)),
            condition=condition,
            description=description,
            uv_index=float(current.get("uv_index", 0)),
            visibility_km=float(current.get("visibility", 0)) / 1000,
            fetched_at=time.time(),
        )

    @staticmethod
    def _parse_daily(daily: dict) -> list[DailyForecast]:
        """Parse the daily forecast arrays into DailyForecast objects."""
        dates = daily.get("time", [])
        codes = daily.get("weather_code", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        wind_max = daily.get("wind_speed_10m_max", [])
        uv_max = daily.get("uv_index_max", [])
        sunrises = daily.get("sunrise", [])
        sunsets = daily.get("sunset", [])

        forecasts: list[DailyForecast] = []

        for i, date in enumerate(dates):
            wmo_code = int(codes[i]) if i < len(codes) else 0
            condition, description = _decode_wmo(wmo_code)

            # Extract HH:MM from ISO datetime string
            sunrise = sunrises[i].split("T")[-1][:5] if i < len(sunrises) else ""
            sunset = sunsets[i].split("T")[-1][:5] if i < len(sunsets) else ""

            forecasts.append(
                DailyForecast(
                    date=date,
                    max_temp_c=float(max_temps[i]) if i < len(max_temps) else 0,
                    min_temp_c=float(min_temps[i]) if i < len(min_temps) else 0,
                    condition=condition,
                    description=description,
                    precipitation_mm=float(precip[i]) if i < len(precip) else 0,
                    wind_max_kph=float(wind_max[i]) if i < len(wind_max) else 0,
                    uv_index=float(uv_max[i]) if i < len(uv_max) else 0,
                    sunrise=sunrise,
                    sunset=sunset,
                )
            )

        return forecasts


# ── Weather Skill ─────────────────────────────────────────────────────────────


class WeatherSkill:
    """
    Weather information skill for F.R.I.D.A.Y.

    Provides current conditions, multi-day forecasts, clothing advice,
    and condition-specific queries. Uses Open-Meteo (no API key required)
    with IP-based geolocation fallback.

    Usage:
        skill = WeatherSkill()
        skill.setup()
        skill.register(action_router)
    """

    def __init__(self, default_location: str = _DEFAULT_LOCATION) -> None:
        self._default_location = default_location
        self._geocoder = _Geocoder()
        self._fetcher = _WeatherFetcher()
        self._ready = False

        # Caches keyed by location name
        self._current_cache: dict[str, WeatherCache] = {}
        self._forecast_cache: dict[str, WeatherCache] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise the weather skill.
        Performs a test geocode to verify connectivity.

        Returns:
            True — always succeeds (network errors are handled per-request).
        """
        self._ready = True
        logger.info("[Weather] Weather skill ready (Open-Meteo, no API key required).")
        return True

    def register(self, router: Any) -> None:
        """Register all weather actions with the ActionRouter."""
        actions = [
            Action(
                name="weather_current",
                patterns=[
                    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?weather(?:\s+(?:like|today|now|currently))?\b",
                    r"\bweather\s+(?:in\s+|for\s+)?(?P<location>.+?)(?:\s+(?:today|now|currently))?\s*$",
                    r"\bcurrent\s+(?:weather|conditions?)\b",
                    r"\bhow(?:'s|\s+is)\s+(?:it|the\s+weather)\s+(?:outside|today)?\b",
                ],
                handler=self._handle_current,
                description="Get current weather conditions.",
            ),
            Action(
                name="weather_forecast",
                patterns=[
                    r"\bweather\s+(?:forecast|this\s+week|for\s+the\s+week)\b",
                    r"\b(?:weekly|7.day|seven.day)\s+(?:weather\s+)?forecast\b",
                    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?weather\s+(?:going\s+to\s+be|like)\s+(?:this\s+week|tomorrow|next\s+few\s+days)\b",
                    r"\bforecast\s+(?:for\s+)?(?P<location>.+)?\b",
                ],
                handler=self._handle_forecast,
                description="Get a multi-day weather forecast.",
            ),
            Action(
                name="weather_tomorrow",
                patterns=[
                    r"\bweather\s+(?:for\s+)?tomorrow\b",
                    r"\bwhat(?:'s|\s+is)\s+(?:it|the\s+weather)\s+(?:going\s+to\s+be\s+)?(?:like\s+)?tomorrow\b",
                    r"\btomorrow(?:'s)?\s+weather\b",
                ],
                handler=self._handle_tomorrow,
                description="Get tomorrow's weather forecast.",
            ),
            Action(
                name="weather_rain",
                patterns=[
                    r"\bwill\s+it\s+rain(?:\s+today|\s+tomorrow|\s+this\s+week)?\b",
                    r"\bis\s+it\s+going\s+to\s+rain\b",
                    r"\bdo\s+i\s+need\s+an?\s+umbrella\b",
                    r"\bwill\s+there\s+be\s+rain\b",
                ],
                handler=self._handle_rain_check,
                description="Check whether rain is expected.",
            ),
            Action(
                name="weather_temperature",
                patterns=[
                    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?temperature\b",
                    r"\bhow\s+(?:hot|cold|warm)\s+is\s+it\b",
                    r"\btemperature\s+(?:today|outside|right\s+now)?\b",
                ],
                handler=self._handle_temperature,
                description="Get the current temperature.",
            ),
            Action(
                name="weather_clothing",
                patterns=[
                    r"\bwhat\s+should\s+i\s+wear\b",
                    r"\bwhat\s+(?:to\s+wear|clothes?\s+(?:should\s+i\s+wear|do\s+i\s+need))\b",
                    r"\bdo\s+i\s+need\s+a\s+(?:coat|jacket|jumper|umbrella)\b",
                    r"\bhow\s+should\s+i\s+dress\b",
                ],
                handler=self._handle_clothing,
                description="Get clothing recommendations based on current weather.",
            ),
            Action(
                name="weather_uv",
                patterns=[
                    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:uv|uv\s+index|ultraviolet)\b",
                    r"\bdo\s+i\s+need\s+sunscreen\b",
                    r"\buv\s+(?:level|index|today)?\b",
                ],
                handler=self._handle_uv,
                description="Get the UV index.",
            ),
            Action(
                name="weather_wind",
                patterns=[
                    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?wind\s*(?:speed|like|today)?\b",
                    r"\bhow\s+(?:windy|breezy)\s+is\s+it\b",
                    r"\bwind\s+(?:speed|conditions?)\b",
                ],
                handler=self._handle_wind,
                description="Get current wind conditions.",
            ),
        ]

        for action in actions:
            router.register(action)

        logger.info("[Weather] %d actions registered.", len(actions))

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_current(self, match: re.Match) -> ActionResult:
        """Report current weather conditions."""
        start = time.perf_counter()
        location_str = self._extract_location(match)
        weather = self._get_current(location_str)

        if not weather:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"I couldn't fetch weather data for {location_str}. Check your internet connection.",
                action="weather_current",
            )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=weather.to_speech(),
            action="weather_current",
            duration=time.perf_counter() - start,
        )

    def _handle_forecast(self, match: re.Match) -> ActionResult:
        """Report a 7-day weather forecast."""
        start = time.perf_counter()
        location_str = self._extract_location(match)
        forecast = self._get_forecast(location_str, days=7)

        if not forecast:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"I couldn't fetch the forecast for {location_str}.",
                action="weather_forecast",
            )

        # Build a 3-day spoken summary
        location = self._geocoder.resolve(location_str)
        loc_name = str(location) if location else location_str

        from datetime import datetime as dt

        labels = ["Today", "Tomorrow"]
        parts = [f"Here's the forecast for {loc_name}."]

        for i, day in enumerate(forecast[:5]):
            label = (
                labels[i] if i < len(labels) else dt.strptime(day.date, "%Y-%m-%d").strftime("%A")
            )
            parts.append(day.to_speech(label))

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=" ".join(parts),
            action="weather_forecast",
            duration=time.perf_counter() - start,
        )

    def _handle_tomorrow(self, match: re.Match) -> ActionResult:
        """Report tomorrow's weather."""
        start = time.perf_counter()
        location_str = self._extract_location(match)
        forecast = self._get_forecast(location_str, days=2)

        if not forecast or len(forecast) < 2:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"I couldn't fetch tomorrow's forecast for {location_str}.",
                action="weather_tomorrow",
            )

        tomorrow = forecast[1]
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Tomorrow's weather: {tomorrow.to_speech('Tomorrow')}",
            action="weather_tomorrow",
            duration=time.perf_counter() - start,
        )

    def _handle_rain_check(self, match: re.Match) -> ActionResult:
        """Check whether rain is expected today."""
        start = time.perf_counter()
        location_str = self._extract_location(match)
        weather = self._get_current(location_str)
        forecast = self._get_forecast(location_str, days=1)

        if not weather:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I couldn't check rain conditions.",
                action="weather_rain",
            )

        if weather.is_raining:
            msg = f"Yes, it's currently raining in {weather.location.name}. Take an umbrella."
        elif forecast and forecast[0].precipitation_mm > 1:
            msg = (
                f"Rain is expected today in {weather.location.name}, "
                f"with {forecast[0].precipitation_mm:.0f}mm forecast. I'd bring an umbrella."
            )
        else:
            msg = f"No rain expected in {weather.location.name} today. You should be fine."

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="weather_rain",
            duration=time.perf_counter() - start,
        )

    def _handle_temperature(self, match: re.Match) -> ActionResult:
        """Report the current temperature."""
        start = time.perf_counter()
        location_str = self._extract_location(match)
        weather = self._get_current(location_str)

        if not weather:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I couldn't fetch the temperature.",
                action="weather_temperature",
            )

        msg = (
            f"It's currently {weather.temperature_c:.0f} degrees Celsius "
            f"({weather.temperature_f:.0f} Fahrenheit) in {weather.location.name}. "
            f"Feels like {weather.feels_like_c:.0f} degrees."
        )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="weather_temperature",
            duration=time.perf_counter() - start,
        )

    def _handle_clothing(self, match: re.Match) -> ActionResult:
        """Provide clothing recommendations."""
        start = time.perf_counter()
        location_str = self._extract_location(match)
        weather = self._get_current(location_str)

        if not weather:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I couldn't get the weather data to make a recommendation.",
                action="weather_clothing",
            )

        msg = (
            f"It's {weather.temperature_c:.0f} degrees with {weather.description} "
            f"in {weather.location.name}. {weather.clothing_advice()}"
        )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="weather_clothing",
            duration=time.perf_counter() - start,
        )

    def _handle_uv(self, match: re.Match) -> ActionResult:
        """Report the UV index."""
        start = time.perf_counter()
        location_str = self._extract_location(match)
        weather = self._get_current(location_str)

        if not weather:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I couldn't fetch UV data.",
                action="weather_uv",
            )

        uv = weather.uv_index
        level = (
            "low"
            if uv < 3
            else "moderate"
            if uv < 6
            else "high"
            if uv < 8
            else "very high"
            if uv < 11
            else "extreme"
        )

        advice = (
            " No sunscreen needed."
            if uv < 3
            else " SPF 15+ sunscreen recommended."
            if uv < 6
            else " SPF 30+ sunscreen recommended."
            if uv < 8
            else " SPF 50+ and protective clothing."
            if uv < 11
            else " Avoid sun exposure if possible."
        )

        msg = f"The UV index in {weather.location.name} is {uv:.0f} — {level}.{advice}"

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="weather_uv",
            duration=time.perf_counter() - start,
        )

    def _handle_wind(self, match: re.Match) -> ActionResult:
        """Report current wind conditions."""
        start = time.perf_counter()
        location_str = self._extract_location(match)
        weather = self._get_current(location_str)

        if not weather:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I couldn't fetch wind data.",
                action="weather_wind",
            )

        speed = weather.wind_speed_kph
        level = (
            "calm"
            if speed < 10
            else "light breeze"
            if speed < 20
            else "moderate"
            if speed < 40
            else "strong"
            if speed < 60
            else "very strong"
            if speed < 80
            else "storm-force"
        )

        msg = (
            f"Wind in {weather.location.name}: {speed:.0f} km/h — {level}. "
            f"Direction: {self._wind_direction(weather.wind_dir)}."
        )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="weather_wind",
            duration=time.perf_counter() - start,
        )

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _get_current(self, location_str: str) -> CurrentWeather | None:
        """
        Fetch current weather with cache support.

        Args:
            location_str: Location name or "auto" for IP-based.

        Returns:
            CurrentWeather or None on failure.
        """
        key = location_str.lower()
        cache = self._current_cache.get(key)

        if cache and cache.is_valid():
            logger.debug("[Weather] Cache hit for '%s'.", location_str)
            return cache.data

        location = self._geocoder.resolve(location_str)
        if not location:
            return None

        weather = self._fetcher.fetch_current(location)
        if weather:
            self._current_cache[key] = WeatherCache(
                data=weather,
                fetched_at=time.time(),
                location=location_str,
            )

        return weather

    def _get_forecast(
        self,
        location_str: str,
        days: int = 7,
    ) -> list[DailyForecast] | None:
        """
        Fetch a daily forecast with cache support.

        Args:
            location_str: Location name.
            days:         Number of forecast days.

        Returns:
            List of DailyForecast or None on failure.
        """
        key = f"{location_str.lower()}:{days}"
        cache = self._forecast_cache.get(key)

        if cache and cache.is_valid():
            return cache.data

        location = self._geocoder.resolve(location_str)
        if not location:
            return None

        forecast = self._fetcher.fetch_forecast(location, days=days)
        if forecast:
            self._forecast_cache[key] = WeatherCache(
                data=forecast,
                fetched_at=time.time(),
                location=location_str,
            )

        return forecast if forecast else None

    def _extract_location(self, match: re.Match) -> str:
        """
        Extract the location string from a regex match.
        Falls back to the configured default location.

        Args:
            match: The regex match object from pattern matching.

        Returns:
            Location string to use for API calls.
        """
        try:
            location = (match.group("location") or "").strip()
            if location:
                return location
        except IndexError:
            pass
        return self._default_location

    @staticmethod
    def _wind_direction(degrees: float) -> str:
        """Convert wind direction in degrees to a compass direction string."""
        directions = [
            "North",
            "North-Northeast",
            "Northeast",
            "East-Northeast",
            "East",
            "East-Southeast",
            "Southeast",
            "South-Southeast",
            "South",
            "South-Southwest",
            "Southwest",
            "West-Southwest",
            "West",
            "West-Northwest",
            "Northwest",
            "North-Northwest",
        ]
        idx = round(degrees / 22.5) % 16
        return directions[idx]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    def set_default_location(self, location: str) -> None:
        """
        Update the default location used when no location is specified.

        Args:
            location: New default location name.
        """
        self._default_location = location
        logger.info("[Weather] Default location updated to '%s'.", location)

    def clear_cache(self) -> None:
        """Invalidate all cached weather data."""
        self._current_cache.clear()
        self._forecast_cache.clear()
        logger.debug("[Weather] Cache cleared.")
