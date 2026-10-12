"""Settings from environment variables (and a local .env file, if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=False)  # real environment variables always win


@dataclass
class Settings:
    provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    api_key: str = ""  # protects the HTTP API; empty means open
    db_path: str = "data/snapcaption.db"
    nominatim_user_agent: str = "snapcaption/0.1"
    temp_unit: str = "fahrenheit"
    max_upload_mb: int = 15


def _int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


def _temp_unit(raw: str) -> str:
    return "celsius" if raw.strip().lower() in ("c", "celsius", "metric") else "fahrenheit"


def load_settings() -> Settings:
    env = os.environ.get
    d = Settings()
    return Settings(
        provider=(env("LLM_PROVIDER") or d.provider).strip().lower(),
        gemini_api_key=(env("GEMINI_API_KEY") or "").strip(),
        gemini_model=(env("GEMINI_MODEL") or d.gemini_model).strip(),
        anthropic_api_key=(env("ANTHROPIC_API_KEY") or "").strip(),
        anthropic_model=(env("ANTHROPIC_MODEL") or d.anthropic_model).strip(),
        api_key=(env("API_KEY") or "").strip(),
        db_path=(env("DB_PATH") or d.db_path).strip(),
        nominatim_user_agent=(env("NOMINATIM_USER_AGENT") or d.nominatim_user_agent).strip(),
        temp_unit=_temp_unit(env("TEMP_UNIT") or d.temp_unit),
        max_upload_mb=_int("MAX_UPLOAD_MB", d.max_upload_mb),
    )
