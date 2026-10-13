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


def _list_of(item: dict) -> dict:
    return {"type": "array", "items": item}


_STR = {"type": "string"}

# JSON schemas for each step. Providers that support constrained output (Ollama) use them;
# small local models otherwise tend to answer "{}".
VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": _STR,
        "setting": _STR,
        "activity": _STR,
        "people_count": {"type": "integer"},
        "subjects": _list_of(_STR),
        "objects": _list_of(_STR),
        "mood": _list_of(_STR),
        "aesthetic": _list_of(_STR),
        "time_of_day_visual": _STR,
        "landmark_guess": {"type": ["string", "null"]},
        "visible_text": _STR,
        "caption_angles": _list_of(_STR),
    },
    "required": ["scene", "mood", "aesthetic", "time_of_day_visual", "caption_angles"],
}
WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "captions": _list_of(
            {"type": "object", "properties": {"text": _STR, "style": _STR}, "required": ["text", "style"]}
        ),
        "hashtags": {
            "type": "object",
            "properties": {tier: _list_of(_STR) for tier in ("broad", "mid", "niche")},
            "required": ["broad", "mid", "niche"],
        },
        "alt_text": _STR,
    },
    "required": ["captions", "hashtags", "alt_text"],
}
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": _list_of(
            {
                "type": "object",
                "properties": {k: {"type": "number"} for k in ("i", "voice", "hook", "originality")},
                "required": ["i", "voice", "hook", "originality"],
            }
        )
    },
    "required": ["scores"],
}


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
    language: str = "English",
) -> str:
    language_rule = (
        ""
        if language.lower() == "english"
        else f"\n- Write every caption in {language}, the way native speakers caption their photos."
    )
    style_lines = "\n".join(f"- {name}: {desc}" for name, desc in STYLES.items())
    return f"""TASK: WRITE_CAPTIONS

You are the person who took this photo, writing your own caption for {platform_name}.
{platform_notes} Aim for under {ideal_length} characters.

WHAT THE PHOTO SHOWS
{json.dumps(vision, ensure_ascii=False, indent=2)}

CONTEXT WE KNOW FOR SURE
{json.dumps(context, ensure_ascii=False, indent=2)}

VOICE
{_voice_block(voice_samples)}

STYLES (spread the captions across these; at least 5 different styles)
{style_lines}

SOUND LIKE A PERSON, NOT A BRAND OR AN AI
- Write the way people actually caption their own photos: casual, a little specific,
  sometimes self-aware or funny. Lowercase and loose punctuation are fine.
- Anchor each caption in a concrete detail from WHAT THE PHOTO SHOWS or CONTEXT (the food
  trucks, the cold, a sign, what you were doing) rather than generic beauty words.
- A small real-life thought beats a slogan: "came for tacos, stayed for the skyline" works,
  "Explore. Discover. Repeat." doesn't.
- Avoid: slogans of three one-word sentences, "lost in the beauty", "this view is
  everything", "soaking up", "urban oasis", "amidst", "embrace", "making memories",
  "living my best life", "good vibes only", "no filter", and em dashes.

RULES
- Write {count} distinct captions. Vary the opening words and the length.{language_rule}
- If CONTEXT has what_the_person_says_about_it, that is the real story of the photo:
  build the captions around it.
- Only mention the place, weather or time if it is in CONTEXT. Never invent a location.
- Only say "golden hour" or "blue hour" if CONTEXT says so (or, when there is no time,
  if time_of_day_from_image says so).
- No hashtags inside captions. At most one emoji per caption, and most need none.
- The caption text is exactly what gets posted: no labels, quotes or trailing words.

HASHTAGS
Suggest hashtags without the # sign in three tiers:
- broad: big, general tags (5)
- mid: medium-sized community tags (5)
- niche: specific to this place, subject or scene (5)

ALT TEXT
One plain sentence describing the photo for people using screen readers: who or what is in
it and what is happening, under 125 characters, no hashtags, not starting with "Image of".
Write it in the same language as the captions.

Return one JSON object:
{{"captions": [{{"text": "...", "style": "one of the style names"}}],
  "hashtags": {{"broad": [], "mid": [], "niche": []}},
  "alt_text": "..."}}"""


def critic_prompt(*, candidates: list[dict], voice_samples: list[str] | None, platform_name: str) -> str:
    listing = "\n".join(f"{i}. [{c['style']}] {c['text']}" for i, c in enumerate(candidates))
    return f"""TASK: SCORE_CAPTIONS

You are a tough social media editor. Score each {platform_name} caption from 0 to 10 on:
- voice: sounds like a real person wrote it about their own day (and like the person below,
  if there are samples); slogans, brand-speak and AI-sounding lines score low
- hook: would make someone stop scrolling
- originality: specific to this photo; would not fit any random city or park picture

VOICE
{_voice_block(voice_samples)}

CAPTIONS
{listing}

Return one JSON object with a score for every caption, using its number as "i":
{{"scores": [{{"i": 0, "voice": 0, "hook": 0, "originality": 0}}]}}"""


REWRITES = {
    "shorter": "Make it shorter and punchier; keep the core idea.",
    "funnier": "Make it funnier, the way a friend with good timing would say it.",
    "more_personal": "Make it more personal: a first-person feeling or a small real detail.",
    "add_emoji": "Keep the wording close, and add one or two emoji that fit naturally.",
    "more_professional": "Make it polished enough for LinkedIn, but still human and warm.",
}
REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"options": _list_of(_STR)},
    "required": ["options"],
}


def rewrite_prompt(
    *, caption: str, how: str, voice_samples: list[str] | None, platform_name: str, language: str = "English"
) -> str:
    language_rule = "" if language.lower() == "english" else f"\n- Keep it in {language}."
    return f"""TASK: REWRITE_CAPTION

Rewrite this {platform_name} caption three different ways.
{REWRITES[how]}

CAPTION
{caption}

VOICE
{_voice_block(voice_samples)}

RULES
- Each option must sound like a real person wrote it about their own photo.
- Don't add facts that aren't in the caption. No hashtags.{language_rule}

Return one JSON object: {{"options": ["...", "...", "..."]}}"""
