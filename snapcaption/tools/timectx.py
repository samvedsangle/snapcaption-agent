"""Turn a capture time (plus sunrise/sunset when known) into caption-friendly context."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from ..schemas import TimeContext

_SEASONS = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}
_FLIP = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "midday"
    if 14 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def build_time_context(
    taken_at: Optional[str],
    sunrise: Optional[str] = None,
    sunset: Optional[str] = None,
    lat: Optional[float] = None,
) -> TimeContext:
    t = _parse(taken_at)
    if t is None:
        return TimeContext()  # unknown: never pretend

    rise, sset = _parse(sunrise), _parse(sunset)
    golden = blue = False
    if sset:
        golden = golden or (sset - timedelta(minutes=60) <= t <= sset)
        blue = blue or (sset < t <= sset + timedelta(minutes=30))
    if rise:
        golden = golden or (rise <= t <= rise + timedelta(minutes=60))
        blue = blue or (rise - timedelta(minutes=30) <= t < rise)

    season = _SEASONS[t.month]
    if lat is not None and lat < 0:
        season = _FLIP[season]

    return TimeContext(
        known=True,
        local_time=t.strftime("%H:%M"),
        weekday=t.strftime("%A"),
        part_of_day=_part_of_day(t.hour),
        golden_hour=golden,
        blue_hour=blue,
        season=season,
    )
