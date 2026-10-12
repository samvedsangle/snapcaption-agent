"""Reverse geocoding via OpenStreetMap Nominatim (free, no key).

Only a coarse label is ever returned: venue or neighbourhood, city, region. Street names
and house numbers are deliberately dropped.
"""

from __future__ import annotations

import threading
import time

import requests

from ..schemas import Place
from . import ToolError

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_VENUE_CATEGORIES = {
    "tourism",
    "leisure",
    "amenity",
    "shop",
    "natural",
    "historic",
    "man_made",
    "building",
}
_MIN_INTERVAL = 1.1  # Nominatim usage policy: at most one request per second
_cache: dict[tuple[float, float], Place] = {}
_lock = threading.Lock()
_last_call = 0.0


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _first(addr: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if addr.get(key):
            return str(addr[key])
    return None


def parse_nominatim(data: dict) -> Place:
    addr = data.get("address") or {}
    city = _first(addr, ("city", "town", "village", "municipality", "hamlet", "county"))
    neighborhood = _first(addr, ("neighbourhood", "suburb", "quarter", "city_district"))
    region = addr.get("state") or addr.get("region")
    country = addr.get("country")

    venue = None
    if data.get("category") in _VENUE_CATEGORIES and data.get("name"):
        venue = str(data["name"])

    parts: list[str] = []
    for part in (venue or neighborhood, city, region):
        if part and part.lower() not in (p.lower() for p in parts):
            parts.append(part)
    label = ", ".join(parts) or (country or "")

    return Place(
        label=label,
        venue=venue,
        neighborhood=neighborhood,
        city=city,
        region=region,
        country=country,
        source="exif",
    )


def reverse_geocode(
    lat: float,
    lon: float,
    *,
    user_agent: str = "snapcaption/0.1",
    timeout: float = 8.0,
) -> Place:
    global _last_call
    key = (round(lat, 4), round(lon, 4))  # about 10 m; also the cache key

    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "format": "jsonv2",
                "lat": key[0],
                "lon": key[1],
                "zoom": 18,
                "addressdetails": 1,
            },
            headers={"User-Agent": user_agent, "Accept-Language": "en"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise ToolError(f"geocoder unavailable ({e.__class__.__name__})") from e

    if not isinstance(data, dict) or data.get("error"):
        raise ToolError("no address found for these coordinates")

    place = parse_nominatim(data)
    if not place.label:
        raise ToolError("no usable place name")

    with _lock:
        _cache[key] = place
    return place
