"""Reverse geocoding via OpenStreetMap Nominatim (free, no key).

Only a coarse label is ever returned: venue or neighbourhood, city, region. Street names
and house numbers are deliberately dropped.
"""

from __future__ import annotations

import re
import threading
import time

import requests

from ..schemas import Place
from . import ToolError

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
_PARK_SEARCH_RADIUS = 0.005  # degrees (about 500 m) around the photo
# Free public Overpass servers. They are often busy, so each gets a short try in turn.
OVERPASS_URLS = (
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
OVERPASS_URL = OVERPASS_URLS[0]
_OVERPASS_TRY_TIMEOUT = 3.0
_OVERPASS_BUDGET = 6.0  # seconds in total; this runs while the vision model is busy anyway
# Venues that already name the place well; anything else (a food stand, a shop, a street)
# triggers a check for an enclosing park or landmark.
_LANDMARK_CATEGORIES = {"tourism", "leisure", "historic", "natural"}
# Enclosing areas worth naming, most specific kind first.
_LANDMARK_QUERY = """[out:json][timeout:3];
is_in({lat},{lon})->.a;
(
  area.a["tourism"~"^(attraction|theme_park|zoo|museum|aquarium|viewpoint)$"]["name"];
  area.a["historic"]["name"];
  area.a["leisure"~"^(park|garden|nature_reserve|stadium|marina|beach_resort)$"]["name"];
  area.a["natural"~"^(beach|bay|peak)$"]["name"];
);
out tags;"""
_LANDMARK_PRIORITY = ("tourism", "historic", "leisure", "natural")
# Administrative districts that read badly in a caption, e.g. "Downtown PID".
_DISTRICT_SUFFIX = re.compile(r"\s*\b(PID|BID|TIF|Public Improvement District|Business Improvement District)$", re.I)
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


def _district(addr: dict) -> str | None:
    for key in ("neighbourhood", "suburb", "quarter", "city_district"):
        name = addr.get(key)
        if name and not _DISTRICT_SUFFIX.search(str(name)):
            return str(name)
    return None


def parse_nominatim(data: dict) -> Place:
    addr = data.get("address") or {}
    city = _first(addr, ("city", "town", "village", "municipality", "hamlet", "county"))
    neighborhood = _district(addr)
    region = addr.get("state") or addr.get("region")
    country = addr.get("country")

    venue = None
    if data.get("category") in _VENUE_CATEGORIES and data.get("name"):
        venue = str(data["name"])

    label = _label(venue or neighborhood, city, region) or (country or "")

    return Place(
        label=label,
        venue=venue,
        neighborhood=neighborhood,
        city=city,
        region=region,
        country=country,
        source="exif",
    )


def _label(*parts: str | None) -> str:
    out: list[str] = []
    for part in parts:
        if part and part.lower() not in (p.lower() for p in out):
            out.append(part)
    return ", ".join(out)


def _with_venue(place: Place, venue: str) -> Place:
    return place.model_copy(
        update={"venue": venue, "label": _label(venue, place.city, place.region) or place.label}
    )


def _wait_for_nominatim() -> None:
    """Nominatim usage policy: at most one request per second from this app."""
    global _last_call
    with _lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _park_containing(lat: float, lon: float, *, user_agent: str) -> str | None:
    """Smallest named park whose bounding box contains the point (Nominatim search)."""
    d = _PARK_SEARCH_RADIUS
    _wait_for_nominatim()
    try:
        resp = requests.get(
            NOMINATIM_SEARCH_URL,
            params={
                "format": "jsonv2",
                "q": "park",
                "bounded": 1,
                "limit": 10,
                "viewbox": f"{lon - d},{lat + d},{lon + d},{lat - d}",
            },
            headers={"User-Agent": user_agent, "Accept-Language": "en"},
            timeout=6.0,
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception:  # noqa: BLE001 - only an upgrade to the label
        return None
    if not isinstance(results, list):
        return None

    best, best_area = None, float("inf")
    for r in results:
        try:
            south, north, west, east = (float(x) for x in r["boundingbox"])
        except (KeyError, TypeError, ValueError):
            continue
        if r.get("name") and south <= lat <= north and west <= lon <= east:
            area = (north - south) * (east - west)
            if area < best_area:
                best, best_area = str(r["name"]), area
    return best


def _overpass_elements(query: str, user_agent: str) -> list | None:
    deadline = time.monotonic() + _OVERPASS_BUDGET
    for url in OVERPASS_URLS:
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            break
        try:
            resp = requests.get(
                url,
                params={"data": query},
                headers={"User-Agent": user_agent},
                timeout=min(_OVERPASS_TRY_TIMEOUT, remaining),
            )
            resp.raise_for_status()
            return resp.json().get("elements") or []
        except Exception:  # noqa: BLE001 - busy server: try the next one
            continue
    return None


def enclosing_landmark(lat: float, lon: float, *, user_agent: str) -> str | None:
    """Name of a park or landmark the point lies inside. Never raises.

    Nominatim's park search answers reliably; the busier Overpass servers are the fallback
    and also cover attractions, historic sites and beaches."""
    park = _park_containing(lat, lon, user_agent=user_agent)
    if park:
        return park
    elements = _overpass_elements(_LANDMARK_QUERY.format(lat=lat, lon=lon), user_agent)
    if not elements:
        return None
    for kind in _LANDMARK_PRIORITY:
        for el in elements:
            tags = (el.get("tags") or {}) if isinstance(el, dict) else {}
            if tags.get(kind) and tags.get("name"):
                return str(tags["name"])
    return None


def reverse_geocode(
    lat: float,
    lon: float,
    *,
    user_agent: str = "snapcaption/0.1",
    timeout: float = 8.0,
) -> Place:
    key = (round(lat, 4), round(lon, 4))  # about 10 m; also the cache key

    with _lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached
    _wait_for_nominatim()

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
    if not (place.venue and data.get("category") in _LANDMARK_CATEGORIES):
        landmark = enclosing_landmark(key[0], key[1], user_agent=user_agent)
        if landmark:
            place = _with_venue(place, landmark)

    with _lock:
        _cache[key] = place
    return place
