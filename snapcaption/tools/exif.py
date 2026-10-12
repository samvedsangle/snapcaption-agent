"""Read GPS position and capture time from a photo's EXIF block."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from ..imaging import open_image
from ..schemas import ExifInfo

_GPS_IFD = 0x8825
_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 0x9003
_DATETIME_DIGITIZED = 0x9004
_DATETIME = 0x0132


def _to_degrees(dms, ref) -> Optional[float]:
    try:
        d, m, s = (float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (d, m, s)):
        return None
    value = d + m / 60.0 + s / 3600.0
    if str(ref).strip().upper() in ("S", "W"):
        value = -value
    return value


def _parse_datetime(raw) -> Optional[str]:
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", "ignore")
    text = str(raw).strip().strip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return None


def read_exif(data: bytes) -> ExifInfo:
    """Never raises: a photo without usable EXIF just yields an empty ExifInfo."""
    try:
        exif = open_image(data, load=False).getexif()
    except ValueError:
        return ExifInfo()

    info = ExifInfo()

    try:
        sub = exif.get_ifd(_EXIF_IFD)
        info.taken_at = (
            _parse_datetime(sub.get(_DATETIME_ORIGINAL))
            or _parse_datetime(sub.get(_DATETIME_DIGITIZED))
            or _parse_datetime(exif.get(_DATETIME))
        )
    except Exception:
        pass

    try:
        gps = exif.get_ifd(_GPS_IFD)
        lat = _to_degrees(gps.get(2), gps.get(1)) if gps.get(2) else None
        lon = _to_degrees(gps.get(4), gps.get(3)) if gps.get(4) else None
        valid = (
            lat is not None
            and lon is not None
            and abs(lat) <= 90
            and abs(lon) <= 180
            and not (lat == 0 and lon == 0)  # (0, 0) is what broken cameras write
        )
        if valid:
            info.has_gps = True
            info.lat = round(lat, 6)
            info.lon = round(lon, 6)
    except Exception:
        pass

    return info
