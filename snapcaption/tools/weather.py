"""Weather at the photo's place and time via Open-Meteo (free, no key)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import requests

from ..schemas import Weather
from . import ToolError

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_WINDOW_DAYS = 80  # the forecast endpoint serves roughly the last 92 days

WMO_CODES = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


def _at(series, idx):
    try:
        return series[idx]
    except (TypeError, IndexError):
        return None


def fetch_weather(
    lat: float,
    lon: float,
    taken_at: str,
    *,
    unit: str = "fahrenheit",
    today: date | None = None,
    timeout: float = 8.0,
) -> Weather:
    try:
        taken = datetime.fromisoformat(taken_at)
    except ValueError as e:
        raise ToolError("bad timestamp") from e

    today = today or date.today()
    recent = (today - taken.date()).days <= _FORECAST_WINDOW_DAYS
    url = FORECAST_URL if recent else ARCHIVE_URL

    params = {
        "latitude": round(lat, 2),  # about 1 km is plenty for weather
        "longitude": round(lon, 2),
        "start_date": taken.date().isoformat(),
        "end_date": taken.date().isoformat(),
        "hourly": "temperature_2m,weather_code,cloud_cover",
        "daily": "sunrise,sunset",
        "temperature_unit": unit,
        "timezone": "auto",  # hourly times come back in the photo's local time
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise ToolError(f"weather unavailable ({e.__class__.__name__})") from e

    hourly = (data or {}).get("hourly") or {}
    times = hourly.get("time") or []

    nearest = taken.replace(minute=0, second=0, microsecond=0)
    if taken.minute >= 30:
        nearest += timedelta(hours=1)
    wanted = [nearest.strftime("%Y-%m-%dT%H:00"), taken.strftime("%Y-%m-%dT%H:00")]
    idx = next((times.index(w) for w in wanted if w in times), None)
    if idx is None:
        raise ToolError("no weather data for that hour")

    temp = _at(hourly.get("temperature_2m"), idx)
    code = _at(hourly.get("weather_code"), idx)
    cloud = _at(hourly.get("cloud_cover"), idx)
    condition = WMO_CODES.get(int(code)) if code is not None else None
    symbol = "F" if unit == "fahrenheit" else "C"

    bits = []
    if condition:
        bits.append(condition)
    if temp is not None:
        bits.append(f"{round(temp)}°{symbol}")

    daily = (data or {}).get("daily") or {}
    return Weather(
        summary=", ".join(bits),
        temp=temp,
        unit=symbol,
        condition=condition,
        cloud_cover=int(cloud) if cloud is not None else None,
        sunrise=_at(daily.get("sunrise"), 0),
        sunset=_at(daily.get("sunset"), 0),
    )
