"""Hashtag clean-up, location tags and per-platform budgets. Plain code, no model."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

from .platforms import get_platform
from .schemas import Place

TIERS = ("location", "niche", "mid", "broad")
_NEAR_DUPLICATE = 0.8  # "klyderenpark" (a typo) or "klydewarren" vs "klydewarrenpark", "cityscapes" vs "cityscape"
_MIN_LEN, _MAX_LEN = 2, 30
_BAIT = re.compile(r"^(like|follow|comment|share|tag|sub)s?(4|for)(like|follow|comment|share|tag|sub)s?$")
_BANNED = {
    "f4f", "l4l", "s4s", "c4c", "followme", "likeme", "tagsforlikes", "instalike",
    "likes", "like", "follow", "followback", "likesforlikes",
}  # fmt: skip


def sanitize_tag(raw) -> Optional[str]:
    """Lowercase letters and digits only; None for junk or engagement bait."""
    if not isinstance(raw, str):
        return None
    text = unicodedata.normalize("NFC", raw).lower()
    tag = "".join(ch for ch in text if ch.isalnum())
    if not (_MIN_LEN <= len(tag) <= _MAX_LEN) or tag.isdigit():
        return None
    if tag in _BANNED or _BAIT.match(tag):
        return None
    return tag


def _dedupe(tags) -> list[str]:
    out: list[str] = []
    for tag in tags:
        if tag and tag not in out:
            out.append(tag)
    return out


def location_tags(place: Optional[Place]) -> list[str]:
    if place is None:
        return []
    if place.source == "user":
        return _dedupe(sanitize_tag(part) for part in place.label.split(","))

    candidates = [place.city, place.venue]
    if place.city and place.region:
        candidates.append(f"{place.city}{place.region}")
    candidates.append(place.neighborhood)
    tags = _dedupe(sanitize_tag(c) for c in candidates)
    if not tags:
        tags = _dedupe([sanitize_tag(place.region), sanitize_tag(place.country)])[:1]
    return tags


def _pool_list(value) -> list[str]:
    if isinstance(value, str):
        items = re.split(r"[,\s]+", value)
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return []
    return _dedupe(sanitize_tag(item) for item in items)


def _near_duplicate(tag: str, chosen: set[str]) -> bool:
    return any(SequenceMatcher(None, tag, other).ratio() >= _NEAR_DUPLICATE for other in chosen)


def build_hashtags(pool, place: Optional[Place], platform: str) -> dict[str, list[str]]:
    """Pick tags tier by tier: each tier's quota first, then fill any leftover budget."""
    spec = get_platform(platform)
    pool = pool if isinstance(pool, dict) else {}
    sources = {"location": location_tags(place)}
    for tier in ("niche", "mid", "broad"):
        sources[tier] = _pool_list(pool.get(tier))

    total = spec.hashtag_count
    chosen: dict[str, list[str]] = {tier: [] for tier in TIERS}
    seen: set[str] = set()

    def take(tier: str, limit: int) -> None:
        for tag in sources[tier]:
            if len(seen) >= total or len(chosen[tier]) >= limit:
                return
            if tag not in seen and not _near_duplicate(tag, seen):
                seen.add(tag)
                chosen[tier].append(tag)

    for tier in TIERS:
        take(tier, spec.hashtag_tiers.get(tier, 0))
    for tier in TIERS:
        take(tier, total)

    result = {tier: [f"#{t}" for t in chosen[tier]] for tier in TIERS}
    result["all"] = [tag for tier in TIERS for tag in result[tier]]
    return result
