import io

import pytest
from PIL import Image

from helpers import make_jpeg
from snapcaption.imaging import open_image, prepare_for_llm
from snapcaption.schemas import ExifInfo
from snapcaption.tools.exif import read_exif


def test_reads_gps_and_capture_time(jpeg_gps):
    info = read_exif(jpeg_gps)
    assert info.has_gps
    assert info.lat == pytest.approx(32.7894, abs=1e-3)
    assert info.lon == pytest.approx(-96.8014, abs=1e-3)
    assert info.taken_at == "2026-10-10T18:05:00"


def test_southern_and_eastern_hemispheres():
    info = read_exif(make_jpeg(lat=-33.8688, lon=151.2093))
    assert info.lat == pytest.approx(-33.8688, abs=1e-3)
    assert info.lon == pytest.approx(151.2093, abs=1e-3)


def test_photo_without_exif(jpeg_plain):
    info = read_exif(jpeg_plain)
    assert not info.has_gps
    assert info.lat is None and info.taken_at is None


def test_time_without_gps():
    info = read_exif(make_jpeg(taken="2026:03:01 09:30:00"))
    assert not info.has_gps
    assert info.taken_at == "2026-03-01T09:30:00"


def test_garbage_bytes_never_raise():
    assert read_exif(b"definitely not an image") == ExifInfo()


def test_null_island_is_rejected():
    assert not read_exif(make_jpeg(lat=0.0, lon=0.0)).has_gps


def test_open_image_rejects_non_images():
    with pytest.raises(ValueError):
        open_image(b"hello")


def test_prepare_downsizes_and_strips_metadata():
    original = make_jpeg(lat=32.7894, lon=-96.8014, taken="2026:10:10 18:05:00", size=(3000, 2000))
    assert read_exif(original).has_gps

    jpeg, mime = prepare_for_llm(original)
    assert mime == "image/jpeg"
    img = Image.open(io.BytesIO(jpeg))
    assert max(img.size) <= 1024
    assert len(img.getexif()) == 0
    assert not read_exif(jpeg).has_gps


def test_prepare_applies_orientation():
    # Orientation 6 means "rotate 90 degrees to display", so a 100x50 file shows as 50x100.
    jpeg, _ = prepare_for_llm(make_jpeg(size=(100, 50), orientation=6))
    assert Image.open(io.BytesIO(jpeg)).size == (50, 100)


def test_prepare_handles_rgba_png():
    buf = io.BytesIO()
    Image.new("RGBA", (40, 40), (10, 20, 30, 128)).save(buf, "PNG")
    jpeg, _ = prepare_for_llm(buf.getvalue())
    assert Image.open(io.BytesIO(jpeg)).mode == "RGB"


def test_prepare_rejects_non_images():
    with pytest.raises(ValueError):
        prepare_for_llm(b"nope")
