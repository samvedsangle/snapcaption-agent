"""Per-platform caption length and hashtag budgets. Tune these from your Results data."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Platform:
    name: str
    ideal_length: int  # characters; longer captions are penalised in ranking
    hashtag_tiers: dict = field(default_factory=dict)  # tier -> how many tags to aim for
    notes: str = ""

    @property
    def hashtag_count(self) -> int:
        return sum(self.hashtag_tiers.values())


PLATFORMS: dict[str, Platform] = {
    "instagram": Platform(
        "instagram",
        150,
        {"location": 2, "niche": 2, "mid": 2, "broad": 2},
        "First line is the hook; the rest hides behind 'more'.",
    ),
    "tiktok": Platform(
        "tiktok",
        100,
        {"location": 1, "niche": 2, "mid": 1, "broad": 1},
        "Short and punchy; reads over a video.",
    ),
    "threads": Platform(
        "threads",
        200,
        {"niche": 1},
        "Conversational; one topic tag at most.",
    ),
    "x": Platform(
        "x",
        200,
        {"location": 1, "niche": 1},
        "Tight; hashtags sparingly.",
    ),
    "linkedin": Platform(
        "linkedin",
        250,
        {"niche": 1, "mid": 1, "broad": 1},
        "Professional but human; no slang overload.",
    ),
}
DEFAULT_PLATFORM = "instagram"


def normalize_platform(name: str | None) -> str:
    key = (name or "").strip().lower()
    if key == "twitter":
        key = "x"
    return key if key in PLATFORMS else DEFAULT_PLATFORM


def get_platform(name: str | None) -> Platform:
    return PLATFORMS[normalize_platform(name)]
