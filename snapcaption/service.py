"""The one entry point both front ends (Streamlit and FastAPI) call."""

from __future__ import annotations

import threading
from typing import Optional

from . import prompts
from .config import Settings, load_settings
from .imaging import open_image
from .llm import get_llm
from .llm import LLMError
from .pipeline import Deps, run_pipeline
from .store import Store
from .tools.geo import reverse_geocode
from .tools.weather import fetch_weather

VOICE_SAMPLES_USED = 12

_stores: dict[str, Store] = {}
_stores_lock = threading.Lock()


def get_store(settings: Optional[Settings] = None) -> Store:
    path = (settings or load_settings()).db_path
    with _stores_lock:
        if path not in _stores:
            _stores[path] = Store(path)
        return _stores[path]


def default_deps(settings: Settings) -> Deps:
    return Deps(
        llm=get_llm(settings),
        geocode=lambda lat, lon: reverse_geocode(lat, lon, user_agent=settings.nominatim_user_agent),
        weather=lambda lat, lon, taken: fetch_weather(lat, lon, taken, unit=settings.temp_unit),
        use_critic=settings.provider != "ollama",
    )


def caption_photo(
    data: bytes,
    *,
    filename: str = "",
    user_id: str = "default",
    platform: str = "instagram",
    place_hint: str = "",
    n: int = 10,
    note: str = "",
    language: str = "English",
    settings: Optional[Settings] = None,
    deps: Optional[Deps] = None,
) -> dict:
    """Validate the upload, run the pipeline with the user's voice, and save the batch."""
    settings = settings or load_settings()
    if not data:
        raise ValueError("No image received.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ValueError(f"Photo is over the {settings.max_upload_mb} MB limit.")
    open_image(data, load=False)  # a bad file is the user's problem to fix first, before config

    deps = deps or default_deps(settings)
    user_id = (user_id or "").strip() or "default"
    store = get_store(settings)

    result = run_pipeline(
        data,
        deps=deps,
        place_hint=place_hint,
        platform=platform,
        n=n,
        voice_samples=store.list_voice_samples(user_id, limit=VOICE_SAMPLES_USED),
        note=note,
        language=language,
    )
    result["generation_id"] = store.save_generation(user_id, result["platform"], result)
    return result


def rewrite_caption(
    text: str,
    how: str,
    *,
    user_id: str = "default",
    platform: str = "instagram",
    language: str = "English",
    settings: Optional[Settings] = None,
    deps: Optional[Deps] = None,
) -> list[str]:
    """Three new takes on one caption: shorter, funnier, more personal, with emoji, or polished."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Send the caption to rewrite.")
    if how not in prompts.REWRITES:
        raise ValueError(f"'how' must be one of: {', '.join(prompts.REWRITES)}.")
    settings = settings or load_settings()
    deps = deps or default_deps(settings)
    user_id = (user_id or "").strip() or "default"
    prompt = prompts.rewrite_prompt(
        caption=text,
        how=how,
        voice_samples=get_store(settings).list_voice_samples(user_id, limit=VOICE_SAMPLES_USED),
        platform_name=platform,
        language=(language or "").strip() or "English",
    )
    resp = deps.llm.text_json(prompt, temperature=0.9, schema=prompts.REWRITE_SCHEMA)
    options: list[str] = []
    for option in resp.get("options") or []:
        option = str(option or "").strip().strip('"')
        if option and option != text and option not in options:
            options.append(option)
    if not options:
        raise LLMError("The model didn't return any rewrites. Try again.")
    return options[:3]


def render_text(result: dict) -> str:
    place = (result.get("context") or {}).get("place")
    lines = [f"Captions for {place}" if place else "Captions", ""]
    lines += [f"{c['rank']}. {c['text']}" for c in result.get("captions", [])]
    tags = (result.get("hashtags") or {}).get("all") or []
    if tags:
        lines += ["", "Hashtags: " + " ".join(tags)]
    if result.get("alt_text"):
        lines += ["", f"Alt text: {result['alt_text']}"]
    for warning in result.get("warnings") or []:
        lines += ["", f"Note: {warning}"]
    return "\n".join(lines) + "\n"
