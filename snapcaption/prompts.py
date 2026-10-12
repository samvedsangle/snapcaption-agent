"""All prompt text lives here. The model never sees coordinates or raw EXIF."""

from __future__ import annotations

import json

STYLES = {
    "witty": "clever wordplay or a light joke",
    "minimal": "three to six words, understated",
    "storytelling": "a tiny moment or memory, first person",
    "question_hook": "ends on a question that invites comments",
    "song_vibe": "the feel of a lyric without quoting a real song",
    "deadpan": "dry, matter-of-fact humour",
    "poetic": "one vivid image, no clichés",
    "hype": "high energy, celebratory",
}

VISION_PROMPT = """You are looking at a photo someone wants to post on social media.
Describe only what is visible. Do not identify real people by name or guess anyone's age,
ethnicity or other sensitive traits.

Return one JSON object with exactly these keys:
{
  "scene": "one sentence describing the photo",
  "setting": "indoor | outdoor | mixed",
  "activity": "what is happening, or empty string",
  "people_count": 0,
  "subjects": ["main subjects"],
  "objects": ["notable objects"],
  "mood": ["2-4 mood words"],
  "aesthetic": ["2-4 visual style words, e.g. golden hour, moody, pastel"],
  "time_of_day_visual": "what the light suggests: morning | midday | golden hour | blue hour | night | unclear",
  "landmark_guess": "a famous landmark only if you are confident, otherwise null",
  "visible_text": "any readable text in the photo, or empty string",
  "caption_angles": ["3-5 short ideas a caption could play on"]
}"""


def _voice_block(voice_samples: list[str] | None) -> str:
    if not voice_samples:
        return "No samples yet. Write in a natural, casual, human voice."
    lines = "\n".join(f"- {s}" for s in voice_samples)
    return f"Captions this person actually wrote (match their tone, length and punctuation):\n{lines}"


def writer_prompt(
    *,
    vision: dict,
    context: dict,
    platform_name: str,
    ideal_length: int,
    platform_notes: str,
    voice_samples: list[str] | None,
    count: int,
) -> str:
    style_lines = "\n".join(f"- {name}: {desc}" for name, desc in STYLES.items())
    return f"""TASK: WRITE_CAPTIONS

You write social media captions for one photo. Platform: {platform_name}.
{platform_notes} Aim for under {ideal_length} characters.

WHAT THE PHOTO SHOWS
{json.dumps(vision, ensure_ascii=False, indent=2)}

CONTEXT WE KNOW FOR SURE
{json.dumps(context, ensure_ascii=False, indent=2)}

VOICE
{_voice_block(voice_samples)}

STYLES (spread the captions across these; at least 5 different styles)
{style_lines}

RULES
- Write {count} distinct captions. Vary the opening words.
- Only mention the place, weather or time if it is in CONTEXT. Never invent a location.
- Only say "golden hour" or "blue hour" if CONTEXT says so (or, when there is no time,
  if time_of_day_from_image says so).
- No hashtags inside captions. At most one emoji per caption.
- Avoid clichés like "living my best life", "good vibes only", "no filter".

HASHTAGS
Suggest hashtags without the # sign in three tiers:
- broad: big, general tags (5)
- mid: medium-sized community tags (5)
- niche: specific to this place, subject or scene (5)

Return one JSON object:
{{"captions": [{{"text": "...", "style": "one of the style names"}}],
  "hashtags": {{"broad": [], "mid": [], "niche": []}}}}"""


def critic_prompt(*, candidates: list[dict], voice_samples: list[str] | None, platform_name: str) -> str:
    listing = "\n".join(f"{i}. [{c['style']}] {c['text']}" for i, c in enumerate(candidates))
    return f"""TASK: SCORE_CAPTIONS

You are a tough social media editor. Score each {platform_name} caption from 0 to 10 on:
- voice: sounds like the person below (if there are no samples, score natural human tone)
- hook: would make someone stop scrolling
- originality: fresh, not generic or clichéd

VOICE
{_voice_block(voice_samples)}

CAPTIONS
{listing}

Return one JSON object with a score for every caption, using its number as "i":
{{"scores": [{{"i": 0, "voice": 0, "hook": 0, "originality": 0}}]}}"""
