"""Small data shapes shared by the tools, the pipeline and the front ends."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ExifInfo(BaseModel):
    has_gps: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None
    taken_at: Optional[str] = None  # local ISO time, as the camera wrote it


class Place(BaseModel):
    label: str = ""
    venue: Optional[str] = None
    neighborhood: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    source: str = "none"  # "exif" (geocoded), "user" (typed in) or "none"


class Weather(BaseModel):
    summary: str = ""
    temp: Optional[float] = None
    unit: str = "F"
    condition: Optional[str] = None
    cloud_cover: Optional[int] = None
    sunrise: Optional[str] = None
    sunset: Optional[str] = None


class TimeContext(BaseModel):
    known: bool = False
    local_time: Optional[str] = None
    weekday: Optional[str] = None
    part_of_day: Optional[str] = None
    golden_hour: bool = False
    blue_hour: bool = False
    season: Optional[str] = None
