"""Turn critic scores into a ranked, de-duplicated, style-diverse top list."""

from __future__ import annotations

import re
from collections import Counter

from .platforms import get_platform

WEIGHTS_WITH_VOICE = {"voice": 0.4, "hook": 0.35, "originality": 0.25}
WEIGHTS_NO_VOICE = {"voice": 0.0, "hook": 0.6, "originality": 0.4}
NEUTRAL_SCORE = 5.0
DUPLICATE_SIMILARITY = 0.8  # word-set overlap at which two captions count as the same

CLICHES = (
    "best life", "good vibes only", "no filter", "nofilter", "making memories",
    "views for days", "adventure awaits", "wanderlust", "blessed", "living my best",
    "life is better", "chasing sunsets", "but first, coffee", "happy place",
)  # fmt: skip
PENALTIES = {"too_short": 10.0, "very_long": 3.0, "long": 1.0, "cliche": 2.0, "hashtag": 1.5}
_WORD = re.compile(r"\w+")


def heuristic_penalty(text: str, platform: str = "instagram") -> tuple[float, list[str]]:
    flags: list[str] = []
    clean = (text or "").strip()
    limit = get_platform(platform).ideal_length

    if len(clean) < 8:
        flags.append("too_short")
    if len(clean) > limit * 1.5:
        flags.append("very_long")
    elif len(clean) > limit:
        flags.append("long")
    lower = clean.lower()
    if any(c in lower for c in CLICHES):
        flags.append("cliche")
    if re.search(r"(^|\s)#\w", clean):
        flags.append("hashtag")  # tags go in the hashtag block, not the caption
    return sum(PENALTIES[f] for f in flags), flags


def _score(raw: dict | None, key: str) -> float:
    try:
        value = float((raw or {}).get(key, NEUTRAL_SCORE))
    except (TypeError, ValueError):
        return NEUTRAL_SCORE
    return min(10.0, max(0.0, value))


def _similar(a: set[str], b: set[str]) -> bool:
    if not a and not b:
        return True
    return len(a & b) / len(a | b) >= DUPLICATE_SIMILARITY


def select_top(
    candidates: list[dict],
    scores: dict[int, dict],
    k: int = 10,
    *,
    per_style_cap: int = 3,
    platform: str = "instagram",
    has_voice: bool = True,
) -> list[dict]:
    """Rank candidates. `scores` maps candidate index -> {voice, hook, originality} (0-10)."""
    weights = WEIGHTS_WITH_VOICE if has_voice else WEIGHTS_NO_VOICE
    scored = []
    for i, cand in enumerate(candidates):
        text = str(cand.get("text") or "").strip()
        if not text:
            continue
        raw = scores.get(i)
        base = sum(w * _score(raw, key) for key, w in weights.items())
        penalty, flags = heuristic_penalty(text, platform)
        scored.append(
            {
                "text": text,
                "style": str(cand.get("style") or "other").strip().lower() or "other",
                "score": round(base - penalty, 2),
                "flags": flags,
                "_words": set(_WORD.findall(text.lower())),
            }
        )
    scored.sort(key=lambda c: c["score"], reverse=True)

    unique: list[dict] = []
    for cand in scored:
        if not any(_similar(cand["_words"], kept["_words"]) for kept in unique):
            unique.append(cand)

    picked: list[dict] = []
    per_style: Counter = Counter()
    for cand in unique:  # best first, at most `per_style_cap` of each style
        if len(picked) >= k:
            break
        if per_style[cand["style"]] < per_style_cap:
            picked.append(cand)
            per_style[cand["style"]] += 1
    taken = {id(c) for c in picked}
    for cand in unique:  # not enough variety: relax the cap
        if len(picked) >= k:
            break
        if id(cand) not in taken:
            picked.append(cand)

    picked.sort(key=lambda c: c["score"], reverse=True)
    return [
        {"rank": r, "text": c["text"], "style": c["style"], "score": c["score"], "flags": c["flags"]}
        for r, c in enumerate(picked, start=1)
    ]
