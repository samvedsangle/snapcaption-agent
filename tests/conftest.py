import pytest

from helpers import make_deps, make_jpeg


@pytest.fixture
def jpeg_plain():
    return make_jpeg()


@pytest.fixture
def jpeg_gps():
    # Dallas Arts District, 6:05 pm local, shortly before sunset
    return make_jpeg(lat=32.7894, lon=-96.8014, taken="2026:10:10 18:05:00")


@pytest.fixture
def deps():
    return make_deps()
