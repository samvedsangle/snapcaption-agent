"""Shared test helpers: synthetic photos, a fake LLM, and fake geocode/weather."""

import io
import re

from PIL import Image

from snapcaption.pipeline import Deps
from snapcaption.schemas import Place, Weather

WORDS = [
    "amber", "breeze", "candle", "drift", "ember", "fable", "glow", "harbor", "indigo",
    "juniper", "kettle", "lantern", "meadow", "nectar", "orchid", "pebble", "quartz",
    "ripple", "saffron", "thistle", "umber", "velvet", "willow", "xenon", "yarrow",
    "zephyr", "anchor", "beacon", "cobalt", "dune",
]  # fmt: skip
STYLES = ["witty", "minimal", "storytelling", "question_hook", "song_vibe"]


def _dms(value: float):
    value = abs(value)
    deg = int(value)
    minutes = int((value - deg) * 60)
    seconds = round((value - deg - minutes / 60) * 3600, 4)
    return (float(deg), float(minutes), seconds)


def make_jpeg(*, lat=None, lon=None, taken=None, size=(64, 48), orientation=None) -> bytes:
    img = Image.new("RGB", size, (200, 120, 60))
    exif = Image.Exif()
    if taken:
        exif[0x8769] = {0x9003: taken}
    if lat is not None and lon is not None:
        exif[0x8825] = {
            1: "N" if lat >= 0 else "S",
            2: _dms(lat),
            3: "E" if lon >= 0 else "W",
            4: _dms(lon),
        }
    if orientation:
        exif[0x0112] = orientation
    kwargs = {"exif": exif} if len(exif) or taken or lat is not None else {}
    buf = io.BytesIO()
    img.save(buf, "JPEG", **kwargs)
    return buf.getvalue()


class FakeLLM:
    def __init__(self, *, fail_critic=False, empty_captions=False, vision_error=None):
        self.prompts: list[str] = []
        self.vision_calls = 0
        self.fail_critic = fail_critic
        self.empty_captions = empty_captions
        self.vision_error = vision_error
        self.vision = {
            "scene": "A city skyline at golden hour with a few people in a park",
            "setting": "outdoor",
            "activity": "strolling",
            "people_count": 3,
            "subjects": ["skyline", "park"],
            "objects": ["trees"],
            "mood": ["relaxed", "warm"],
            "aesthetic": ["golden hour", "urban"],
            "time_of_day_visual": "golden hour",
            "landmark_guess": None,
            "visible_text": "",
            "caption_angles": ["weekend reset", "city glow"],
        }

    def vision_json(self, image_bytes, mime, prompt, temperature=0.2):
        self.vision_calls += 1
        self.prompts.append(prompt)
        if self.vision_error:
            raise self.vision_error
        return self.vision

    def text_json(self, prompt, temperature=0.7):
        self.prompts.append(prompt)
        if "TASK: WRITE_CAPTIONS" in prompt:
            if self.empty_captions:
                return {"captions": [], "hashtags": {}}
            captions = []
            for s, style in enumerate(STYLES):
                for k in range(5):
                    idx = s * 5 + k
                    words = " ".join(WORDS[idx : idx + 4])
                    captions.append({"text": f"{style} {words}", "style": style})
            return {
                "captions": captions,
                "hashtags": {
                    "broad": ["travel", "photography", "citylife", "weekend"],
                    "mid": ["urbanexplorer", "skylinelovers", "cityscapes", "goldenhourphotography"],
                    "niche": ["klydewarrenpark", "dallasskyline", "dfwphotographer", "texasgoldenhour"],
                },
            }
        if "TASK: SCORE_CAPTIONS" in prompt:
            if self.fail_critic:
                raise RuntimeError("critic down")
            n = len(re.findall(r"^\d+\. \[", prompt, flags=re.M))
            return {
                "scores": [
                    {"i": i, "voice": 6, "hook": 9 - (i % 5), "originality": 7} for i in range(n)
                ]
            }
        raise AssertionError("unexpected prompt")


PARK = Place(
    label="Klyde Warren Park, Dallas, Texas",
    venue="Klyde Warren Park",
    neighborhood="Arts District",
    city="Dallas",
    region="Texas",
    country="United States",
    source="exif",
)

CLEAR_EVENING = Weather(
    summary="clear sky, 72°F",
    temp=72.0,
    unit="F",
    condition="clear sky",
    cloud_cover=5,
    sunrise="2026-10-10T07:21",
    sunset="2026-10-10T18:40",
)


class Calls:
    def __init__(self):
        self.geocode: list = []
        self.weather: list = []


def make_deps(*, llm=None, geocode_error=None, weather_error=None):
    calls = Calls()

    def geocode(lat, lon):
        calls.geocode.append((lat, lon))
        if geocode_error:
            raise geocode_error
        return PARK

    def weather(lat, lon, taken):
        calls.weather.append((lat, lon, taken))
        if weather_error:
            raise weather_error
        return CLEAR_EVENING

    deps = Deps(llm=llm or FakeLLM(), geocode=geocode, weather=weather)
    deps.calls = calls  # type: ignore[attr-defined]
    return deps
