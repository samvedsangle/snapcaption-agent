import json

import pytest

from helpers import FakeLLM, make_deps, make_jpeg
from snapcaption.pipeline import PipelineError, run_pipeline
from snapcaption.tools import ToolError


def test_happy_path_with_gps(jpeg_gps, deps):
    result = run_pipeline(jpeg_gps, deps=deps)

    captions = result["captions"]
    assert len(captions) == 10
    assert [c["rank"] for c in captions] == list(range(1, 11))
    styles = [c["style"] for c in captions]
    assert all(styles.count(s) <= 3 for s in set(styles))  # diversity cap

    ctx = result["context"]
    assert ctx["place"] == "Klyde Warren Park, Dallas, Texas"
    assert ctx["place_source"] == "exif"
    assert ctx["weather"] == "clear sky, 72°F"
    assert ctx["time"]["golden_hour"] is True  # 18:05, sunset 18:40
    assert ctx["vibe"][:2] == ["relaxed", "warm"]

    tags = result["hashtags"]["all"]
    assert len(tags) == 8 and "#dallas" in tags and "#klydewarrenpark" in tags
    assert len(set(tags)) == len(tags)

    assert len(result["choices"]) == 10
    assert result["choices"][0].startswith(captions[0]["text"])
    assert result["choices"][0].endswith(tags[-1])
    assert result["warnings"] == []
    assert deps.llm.vision_calls == 1


def test_every_node_is_timed(jpeg_gps, deps):
    timings = run_pipeline(jpeg_gps, deps=deps)["timings"]
    for node in (
        "read_exif", "analyze_image", "lookup_place", "lookup_weather",
        "write_captions", "score_captions", "finalize", "total",
    ):  # fmt: skip
        assert node in timings


def test_coordinates_never_reach_the_model_or_the_result(jpeg_gps, deps):
    result = run_pipeline(jpeg_gps, deps=deps)
    haystack = "\n".join(deps.llm.prompts) + json.dumps(result)
    for needle in ("32.78", "96.80", "-96.8", "32.7894"):
        assert needle not in haystack
    assert "taken_at" not in json.dumps(result)


def test_no_exif_and_no_hint(jpeg_plain, deps):
    result = run_pipeline(jpeg_plain, deps=deps)
    assert len(result["captions"]) == 10
    assert result["context"]["place"] == "" and result["context"]["place_source"] == "none"
    assert result["context"]["time"]["known"] is False
    assert any("no location data" in w for w in result["warnings"])
    assert deps.calls.geocode == [] and deps.calls.weather == []
    # without a real timestamp the writer falls back to what the image itself shows
    assert "time_of_day_from_image" in deps.llm.prompts[-2]


def test_place_hint_beats_gps(jpeg_gps, deps):
    result = run_pipeline(jpeg_gps, place_hint="Joe's Coffee, Austin", deps=deps)
    assert result["context"]["place"] == "Joe's Coffee, Austin"
    assert result["context"]["place_source"] == "user"
    assert deps.calls.geocode == []
    assert "#joescoffee" in result["hashtags"]["all"] and "#austin" in result["hashtags"]["all"]


def test_geocoder_failure_degrades_gracefully(jpeg_gps):
    deps = make_deps(geocode_error=ToolError("geocoder unavailable"))
    result = run_pipeline(jpeg_gps, deps=deps)
    assert len(result["captions"]) == 10
    assert result["context"]["place"] == ""
    assert any("place name" in w for w in result["warnings"])
    assert result["context"]["weather"] == "clear sky, 72°F"  # the other branch still ran


def test_weather_failure_degrades_gracefully(jpeg_gps):
    deps = make_deps(weather_error=ToolError("weather unavailable"))
    result = run_pipeline(jpeg_gps, deps=deps)
    assert len(result["captions"]) == 10
    assert result["context"]["weather"] == ""
    assert result["context"]["time"]["known"] is True  # time still comes from EXIF
    assert result["context"]["time"]["golden_hour"] is False  # no sunset data, so no claim
    assert any("weather" in w for w in result["warnings"])


def test_critic_failure_falls_back_to_rules(jpeg_gps):
    deps = make_deps(llm=FakeLLM(fail_critic=True))
    result = run_pipeline(jpeg_gps, deps=deps)
    assert len(result["captions"]) == 10
    assert any("simple rules" in w for w in result["warnings"])


def test_empty_generation_is_a_clear_error(jpeg_gps):
    with pytest.raises(PipelineError):
        run_pipeline(jpeg_gps, deps=make_deps(llm=FakeLLM(empty_captions=True)))


def test_vision_failure_propagates(jpeg_gps):
    with pytest.raises(RuntimeError, match="vision down"):
        run_pipeline(jpeg_gps, deps=make_deps(llm=FakeLLM(vision_error=RuntimeError("vision down"))))


def test_voice_samples_reach_the_writer_and_critic(jpeg_gps, deps):
    run_pipeline(jpeg_gps, voice_samples=["sunday reset, no notes", "coffee first, opinions later"], deps=deps)
    writer, critic = deps.llm.prompts[-2], deps.llm.prompts[-1]
    for prompt in (writer, critic):
        assert "coffee first, opinions later" in prompt
        assert "No samples yet" not in prompt


def test_without_samples_the_writer_is_told_so(jpeg_gps, deps):
    run_pipeline(jpeg_gps, deps=deps)
    assert "No samples yet" in deps.llm.prompts[-2]


def test_non_images_fail_fast(deps):
    with pytest.raises(ValueError):
        run_pipeline(b"not a photo", deps=deps)
    assert deps.llm.vision_calls == 0  # never spent a model call


def test_n_and_platform_are_normalised(jpeg_gps):
    big = run_pipeline(jpeg_gps, n=50, deps=make_deps())
    assert len(big["captions"]) == 15
    small = run_pipeline(jpeg_gps, n=1, deps=make_deps())
    assert len(small["captions"]) == 3
    weird = run_pipeline(jpeg_gps, platform="MySpace", deps=make_deps())
    assert weird["platform"] == "instagram"


def test_hashtag_budget_follows_the_platform(jpeg_gps):
    assert len(run_pipeline(jpeg_gps, platform="x", deps=make_deps())["hashtags"]["all"]) == 2
    assert len(run_pipeline(jpeg_gps, platform="tiktok", deps=make_deps())["hashtags"]["all"]) == 5


def test_photo_taken_without_sun_data_still_gets_time_of_day():
    deps = make_deps()
    jpeg = make_jpeg(taken="2026:10:10 18:05:00")  # time but no GPS
    result = run_pipeline(jpeg, deps=deps)
    assert result["context"]["time"]["part_of_day"] == "evening"
    assert deps.calls.weather == []
