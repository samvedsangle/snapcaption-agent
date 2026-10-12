from datetime import date

import pytest
import requests

from snapcaption.tools import ToolError, geo, weather
from snapcaption.tools.timectx import build_time_context


class FakeResp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.payload


@pytest.fixture(autouse=True)
def _fresh_geo(monkeypatch):
    monkeypatch.setattr(geo, "_MIN_INTERVAL", 0)
    geo.clear_cache()
    yield
    geo.clear_cache()


def patch_get(monkeypatch, module, payload=None, error=None, status=200):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers})
        if error:
            raise error
        return FakeResp(payload, status)

    monkeypatch.setattr(module.requests, "get", fake_get)
    return calls


PARK_JSON = {
    "category": "leisure",
    "type": "park",
    "name": "Klyde Warren Park",
    "address": {
        "leisure": "Klyde Warren Park",
        "road": "Woodall Rodgers Freeway",
        "house_number": "2012",
        "neighbourhood": "Arts District",
        "city": "Dallas",
        "county": "Dallas County",
        "state": "Texas",
        "country": "United States",
    },
}


# ------------------------------------------------------------------ geocoder
def test_geocode_parses_venue_and_hides_street(monkeypatch):
    calls = patch_get(monkeypatch, geo, PARK_JSON)
    place = geo.reverse_geocode(32.7894, -96.8014, user_agent="ua-test")
    assert place.label == "Klyde Warren Park, Dallas, Texas"
    assert place.venue == "Klyde Warren Park"
    assert place.neighborhood == "Arts District"
    assert place.source == "exif"
    assert "Woodall" not in place.label and "2012" not in place.label
    assert calls[0]["headers"]["User-Agent"] == "ua-test"


def test_geocode_street_only_uses_neighbourhood(monkeypatch):
    patch_get(
        monkeypatch,
        geo,
        {
            "category": "highway",
            "name": "Main Street",
            "address": {"road": "Main Street", "suburb": "Deep Ellum", "city": "Dallas", "state": "Texas"},
        },
    )
    place = geo.reverse_geocode(32.78, -96.78)
    assert place.venue is None
    assert place.label == "Deep Ellum, Dallas, Texas"


def test_geocode_dedupes_repeated_names(monkeypatch):
    patch_get(monkeypatch, geo, {"category": "place", "address": {"city": "Paris", "suburb": "Paris", "country": "France"}})
    assert geo.reverse_geocode(48.85, 2.35).label == "Paris"


def test_geocode_caches_by_rounded_coordinates(monkeypatch):
    calls = patch_get(monkeypatch, geo, PARK_JSON)
    geo.reverse_geocode(32.78941, -96.80141)
    geo.reverse_geocode(32.78942, -96.80142)  # same 4-decimal cell
    assert len(calls) == 1


def test_geocode_errors_become_tool_errors(monkeypatch):
    patch_get(monkeypatch, geo, error=requests.ConnectionError("offline"))
    with pytest.raises(ToolError):
        geo.reverse_geocode(1.0, 1.0)


def test_geocode_no_address_is_a_tool_error(monkeypatch):
    patch_get(monkeypatch, geo, {"error": "Unable to geocode"})
    with pytest.raises(ToolError):
        geo.reverse_geocode(1.0, 1.0)


def test_geocode_http_error(monkeypatch):
    patch_get(monkeypatch, geo, {}, status=429)
    with pytest.raises(ToolError):
        geo.reverse_geocode(1.0, 1.0)


FOOD_STAND_JSON = {
    "category": "amenity",
    "type": "fast_food",
    "name": "Freshnez",
    "address": {
        "amenity": "Freshnez",
        "road": "Woodall Rodgers Freeway",
        "neighbourhood": "Klyde Warren Park-Arts District PID",
        "quarter": "Downtown PID",
        "city": "Dallas",
        "state": "Texas",
        "country": "United States",
    },
}


def patch_routes(monkeypatch, nominatim, overpass):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        reply = nominatim if url == geo.NOMINATIM_URL else overpass
        if isinstance(reply, Exception):
            raise reply
        return FakeResp(reply)

    monkeypatch.setattr(geo.requests, "get", fake_get)
    return calls


def test_small_venue_is_upgraded_to_the_enclosing_park(monkeypatch):
    overpass = {"elements": [{"tags": {"leisure": "park", "name": "Klyde Warren Park"}}]}
    calls = patch_routes(monkeypatch, FOOD_STAND_JSON, overpass)
    place = geo.reverse_geocode(32.7894, -96.8014)
    assert place.label == "Klyde Warren Park, Dallas, Texas"
    assert place.venue == "Klyde Warren Park"
    assert calls == [geo.NOMINATIM_URL, geo.OVERPASS_URL]


def test_landmark_priority_prefers_the_attraction(monkeypatch):
    elements = [
        {"tags": {"leisure": "park", "name": "Fair Park"}},
        {"tags": {"tourism": "museum", "name": "Hall of State"}},
    ]
    patch_get(monkeypatch, geo, {"elements": elements})
    assert geo.enclosing_landmark(1.0, 1.0, user_agent="ua") == "Hall of State"


def test_busy_overpass_server_falls_through_to_a_mirror(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        if url == geo.OVERPASS_URLS[0]:
            return FakeResp({}, status=500)
        return FakeResp({"elements": [{"tags": {"leisure": "park", "name": "Klyde Warren Park"}}]})

    monkeypatch.setattr(geo.requests, "get", fake_get)
    assert geo.enclosing_landmark(32.7894, -96.8014, user_agent="ua") == "Klyde Warren Park"
    assert calls == list(geo.OVERPASS_URLS[:2])


def test_landmark_lookup_failure_keeps_the_nominatim_answer(monkeypatch):
    patch_routes(monkeypatch, FOOD_STAND_JSON, requests.Timeout("slow"))
    place = geo.reverse_geocode(32.7894, -96.8014)
    assert place.label == "Freshnez, Dallas, Texas"
    assert place.neighborhood is None  # "... PID" districts are not caption material


def test_parks_skip_the_landmark_lookup(monkeypatch):
    calls = patch_routes(monkeypatch, PARK_JSON, {"elements": []})
    geo.reverse_geocode(32.7894, -96.8014)
    assert calls == [geo.NOMINATIM_URL]


# ------------------------------------------------------------------- weather
def weather_payload(day="2026-10-10"):
    codes = [3] * 24
    codes[18] = 0
    clouds = [100] * 24
    clouds[18] = 5
    return {
        "hourly": {
            "time": [f"{day}T{h:02d}:00" for h in range(24)],
            "temperature_2m": [60.0 + h for h in range(24)],
            "weather_code": codes,
            "cloud_cover": clouds,
        },
        "daily": {"sunrise": [f"{day}T07:21"], "sunset": [f"{day}T18:40"]},
    }


def test_weather_picks_the_right_hour(monkeypatch):
    calls = patch_get(monkeypatch, weather, weather_payload())
    w = weather.fetch_weather(32.7894, -96.8014, "2026-10-10T18:05:00", today=date(2026, 10, 12))
    assert w.summary == "clear sky, 78°F"
    assert w.cloud_cover == 5
    assert w.sunset == "2026-10-10T18:40"
    params = calls[0]["params"]
    assert calls[0]["url"] == weather.FORECAST_URL
    assert params["latitude"] == 32.79 and params["longitude"] == -96.8  # coarsened
    assert params["timezone"] == "auto"


def test_weather_rounds_to_nearest_hour(monkeypatch):
    patch_get(monkeypatch, weather, weather_payload())
    w = weather.fetch_weather(1.0, 1.0, "2026-10-10T18:40:00", today=date(2026, 10, 12))
    assert w.summary == "overcast, 79°F"  # nearest hour is 19:00


def test_weather_uses_archive_for_old_photos(monkeypatch):
    calls = patch_get(monkeypatch, weather, weather_payload("2025-01-01"))
    weather.fetch_weather(1.0, 1.0, "2025-01-01T18:05:00", today=date(2026, 10, 12))
    assert calls[0]["url"] == weather.ARCHIVE_URL


def test_weather_celsius(monkeypatch):
    calls = patch_get(monkeypatch, weather, weather_payload())
    w = weather.fetch_weather(1.0, 1.0, "2026-10-10T18:05:00", unit="celsius", today=date(2026, 10, 12))
    assert w.summary.endswith("°C")
    assert calls[0]["params"]["temperature_unit"] == "celsius"


def test_weather_errors(monkeypatch):
    patch_get(monkeypatch, weather, error=requests.Timeout("slow"))
    with pytest.raises(ToolError):
        weather.fetch_weather(1.0, 1.0, "2026-10-10T18:05:00", today=date(2026, 10, 12))


def test_weather_missing_hour(monkeypatch):
    patch_get(monkeypatch, weather, {"hourly": {"time": []}})
    with pytest.raises(ToolError):
        weather.fetch_weather(1.0, 1.0, "2026-10-10T18:05:00", today=date(2026, 10, 12))


def test_weather_bad_timestamp():
    with pytest.raises(ToolError):
        weather.fetch_weather(1.0, 1.0, "yesterday-ish")


# -------------------------------------------------------------- time context
def test_golden_hour_before_sunset():
    t = build_time_context("2026-10-10T18:05:00", "2026-10-10T07:21", "2026-10-10T18:40")
    assert t.known and t.golden_hour and not t.blue_hour
    assert t.part_of_day == "evening" and t.weekday == "Saturday" and t.season == "autumn"


def test_blue_hour_after_sunset():
    t = build_time_context("2026-10-10T18:55:00", "2026-10-10T07:21", "2026-10-10T18:40")
    assert t.blue_hour and not t.golden_hour


def test_golden_hour_after_sunrise():
    assert build_time_context("2026-10-10T07:40:00", "2026-10-10T07:21", "2026-10-10T18:40").golden_hour


def test_midday_is_neither():
    t = build_time_context("2026-10-10T12:30:00", "2026-10-10T07:21", "2026-10-10T18:40")
    assert not t.golden_hour and not t.blue_hour and t.part_of_day == "midday"


def test_unknown_time_is_not_invented():
    assert build_time_context(None).known is False


def test_no_sun_times_means_no_golden_claim():
    assert build_time_context("2026-10-10T18:05:00").golden_hour is False


def test_southern_hemisphere_season_flips():
    assert build_time_context("2026-12-25T10:00:00", lat=-33.9).season == "summer"
    assert build_time_context("2026-12-25T10:00:00", lat=40.0).season == "winter"
