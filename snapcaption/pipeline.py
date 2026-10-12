"""The LangGraph pipeline.

    read_exif ─┬─ lookup_place ───┐
               └─ lookup_weather ─┼─ write_captions ─ score_captions ─ finalize
    analyze_image ────────────────┘

Image analysis runs alongside the EXIF → place / weather branch; writing waits for all three.
"""

from __future__ import annotations

import operator
import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import prompts
from .hashtags import build_hashtags
from .imaging import open_image, prepare_for_llm
from .platforms import get_platform, normalize_platform
from .ranking import select_top
from .schemas import ExifInfo, Place, TimeContext, Weather
from .tools import ToolError
from .tools.exif import read_exif
from .tools.timectx import build_time_context

MIN_CAPTIONS, MAX_CAPTIONS = 3, 15
PER_STYLE_CAP = 3
# Models sometimes echo Instagram's "... more" fold after a finished sentence.
_TRAILING_MORE = re.compile(r"([.!?…])\s*(?:\.\.\.|…)?\s*more$", re.I)


class PipelineError(RuntimeError):
    """The pipeline could not produce captions (e.g. the model returned none)."""


@dataclass
class Deps:
    llm: Any  # has vision_json(...) and text_json(...)
    geocode: Callable[[float, float], Place]
    weather: Callable[[float, float, str], Weather]
    # The critic call adds little with a small local model and roughly doubles its run time;
    # without it, captions are ranked by the plain-code rules alone.
    use_critic: bool = True


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class State(TypedDict, total=False):
    # inputs
    image: bytes
    place_hint: str
    platform: str
    n: int
    voice_samples: list[str]
    # intermediate
    exif: ExifInfo
    vision: dict
    place: Optional[Place]
    weather: Optional[Weather]
    time: TimeContext
    candidates: list[dict]
    hashtag_pool: dict
    scores: dict
    # output
    result: dict
    warnings: Annotated[list[str], operator.add]
    timings: Annotated[dict, _merge]


def _as_list(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _timed(name: str, fn: Callable[[State], dict]) -> Callable[[State], dict]:
    def node(state: State) -> dict:
        start = time.perf_counter()
        update = fn(state)
        update["timings"] = {name: round((time.perf_counter() - start) * 1000, 1)}
        return update

    return node


def _build_graph(deps: Deps):
    def node_read_exif(state: State) -> dict:
        return {"exif": read_exif(state["image"])}

    def node_analyze_image(state: State) -> dict:
        jpeg, mime = prepare_for_llm(state["image"])
        vision = deps.llm.vision_json(jpeg, mime, prompts.VISION_PROMPT, schema=prompts.VISION_SCHEMA)
        return {"vision": vision if isinstance(vision, dict) else {}}

    def node_lookup_place(state: State) -> dict:
        hint = (state.get("place_hint") or "").strip()
        if hint:
            return {"place": Place(label=hint, source="user")}
        exif = state["exif"]
        if not exif.has_gps:
            return {
                "place": None,
                "warnings": [
                    "This photo has no location data, so captions skip the place. "
                    "Add the place yourself to include it."
                ],
            }
        try:
            return {"place": deps.geocode(exif.lat, exif.lon)}
        except ToolError as e:
            return {
                "place": None,
                "warnings": [f"Couldn't look up a place name for this photo ({e}). Add the place yourself."],
            }

    def node_lookup_weather(state: State) -> dict:
        exif = state["exif"]
        weather, warnings = None, []
        if exif.has_gps and exif.taken_at:
            try:
                weather = deps.weather(exif.lat, exif.lon, exif.taken_at)
            except ToolError as e:
                warnings.append(f"Couldn't get the weather for this photo ({e}).")
        time_ctx = build_time_context(
            exif.taken_at,
            weather.sunrise if weather else None,
            weather.sunset if weather else None,
            lat=exif.lat,
        )
        return {"weather": weather, "time": time_ctx, "warnings": warnings}

    def node_write_captions(state: State) -> dict:
        spec = get_platform(state["platform"])
        vision = state.get("vision") or {}
        place, weather, time_ctx = state.get("place"), state.get("weather"), state["time"]

        context: dict[str, Any] = {}
        if place and place.label:
            context["place"] = place.label
        if weather and weather.summary:
            context["weather"] = weather.summary
        if time_ctx.known:
            context["time"] = time_ctx.model_dump(exclude={"known"})
        else:  # no timestamp: only the light in the image tells us anything
            context["time_of_day_from_image"] = vision.get("time_of_day_visual") or "unclear"

        prompt = prompts.writer_prompt(
            vision=vision,
            context=context,
            platform_name=spec.name,
            ideal_length=spec.ideal_length,
            platform_notes=spec.notes,
            voice_samples=state.get("voice_samples"),
            count=max(25, state["n"] * 2 + 5),
        )
        resp = deps.llm.text_json(prompt, temperature=0.9, schema=prompts.WRITER_SCHEMA)

        candidates = []
        for item in resp.get("captions") or []:
            if isinstance(item, str):
                item = {"text": item, "style": "other"}
            if not isinstance(item, dict):
                continue
            text = _TRAILING_MORE.sub(r"\1", str(item.get("text") or "").strip())
            if text:
                candidates.append({"text": text, "style": str(item.get("style") or "other")})
        if not candidates:
            raise PipelineError("The model returned no captions. Try again.")
        return {"candidates": candidates, "hashtag_pool": resp.get("hashtags") or {}}

    def node_score_captions(state: State) -> dict:
        if not deps.use_critic:
            return {"scores": {}}
        candidates = state["candidates"]
        prompt = prompts.critic_prompt(
            candidates=candidates,
            voice_samples=state.get("voice_samples"),
            platform_name=get_platform(state["platform"]).name,
        )
        scores: dict[int, dict] = {}
        try:
            resp = deps.llm.text_json(prompt, temperature=0.2, schema=prompts.CRITIC_SCHEMA)
            for item in resp.get("scores") or []:
                try:
                    i = int(item["i"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= i < len(candidates):
                    scores[i] = item
        except Exception:  # noqa: BLE001 - ranking must survive a critic failure
            scores = {}
        if not scores:
            return {
                "scores": {},
                "warnings": ["The critic model was unavailable, so captions were ranked with simple rules."],
            }
        return {"scores": scores}

    def node_finalize(state: State) -> dict:
        platform = state["platform"]
        captions = select_top(
            state["candidates"],
            state.get("scores") or {},
            k=state["n"],
            per_style_cap=PER_STYLE_CAP,
            platform=platform,
            has_voice=bool(state.get("voice_samples")),
        )
        place = state.get("place")
        hashtags = build_hashtags(state.get("hashtag_pool"), place, platform)
        tag_line = " ".join(hashtags["all"])
        choices = [f"{c['text']}\n\n{tag_line}" if tag_line else c["text"] for c in captions]

        vision = state.get("vision") or {}
        vibe: list[str] = []
        for word in _as_list(vision.get("mood")) + _as_list(vision.get("aesthetic")):
            if word not in vibe:
                vibe.append(word)
        weather = state.get("weather")
        return {
            "result": {
                "platform": platform,
                "captions": captions,
                "hashtags": hashtags,
                "choices": choices,
                "context": {
                    "place": place.label if place else "",
                    "place_source": place.source if place else "none",
                    "weather": weather.summary if weather else "",
                    "time": state["time"].model_dump(),
                    "vibe": vibe[:5],
                    "scene": str(vision.get("scene") or ""),
                },
            }
        }

    g = StateGraph(State)
    for name, fn in (
        ("read_exif", node_read_exif),
        ("analyze_image", node_analyze_image),
        ("lookup_place", node_lookup_place),
        ("lookup_weather", node_lookup_weather),
        ("write_captions", node_write_captions),
        ("score_captions", node_score_captions),
        ("finalize", node_finalize),
    ):
        g.add_node(name, _timed(name, fn))
    g.add_edge(START, "read_exif")
    g.add_edge(START, "analyze_image")
    g.add_edge("read_exif", "lookup_place")
    g.add_edge("read_exif", "lookup_weather")
    g.add_edge(["analyze_image", "lookup_place", "lookup_weather"], "write_captions")
    g.add_edge("write_captions", "score_captions")
    g.add_edge("score_captions", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def run_pipeline(
    image: bytes,
    *,
    deps: Deps,
    place_hint: str = "",
    platform: str = "instagram",
    n: int = 10,
    voice_samples: Optional[list[str]] = None,
) -> dict:
    """Photo bytes in, ranked captions + hashtags + context out. Raises ValueError for non-images."""
    open_image(image)  # fail fast, before spending a model call
    start = time.perf_counter()
    state = _build_graph(deps).invoke(
        {
            "image": image,
            "place_hint": place_hint or "",
            "platform": normalize_platform(platform),
            "n": max(MIN_CAPTIONS, min(MAX_CAPTIONS, int(n or 10))),
            "voice_samples": [s for s in (voice_samples or []) if s and s.strip()],
            "warnings": [],
            "timings": {},
        }
    )
    result = dict(state["result"])
    result["warnings"] = list(state.get("warnings") or [])
    result["timings"] = {**state.get("timings", {}), "total": round((time.perf_counter() - start) * 1000, 1)}
    return result
