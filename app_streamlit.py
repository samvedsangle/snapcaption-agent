"""SnapCaption web app: drop a photo, captions appear on their own.

Run locally:  streamlit run app_streamlit.py
"""

import hashlib
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SnapCaption", page_icon="📸", layout="centered")


def _secrets_to_env() -> None:
    """Streamlit Cloud keeps secrets in st.secrets; the core package reads env vars."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float)) and key not in os.environ:
                os.environ[key] = str(value)
    except Exception:  # no secrets file locally; .env is used instead
        pass


_secrets_to_env()

from snapcaption.llm import LLMError  # noqa: E402
from snapcaption.pipeline import PipelineError  # noqa: E402
from snapcaption.platforms import PLATFORMS  # noqa: E402
from snapcaption.service import caption_photo, get_store  # noqa: E402


@st.cache_data(show_spinner=False, ttl=3600, max_entries=16)
def _generate(image_hash: str, _image: bytes, user_id: str, platform: str, place: str, nonce: int):
    # image_hash / nonce are part of the cache key; the raw bytes are excluded from hashing.
    return caption_photo(_image, user_id=user_id, platform=platform, place_hint=place)


def _log_pick(generation_id: int, rank: int) -> None:
    try:
        get_store().record_pick(generation_id, rank=rank)
        st.session_state["toast"] = f"Logged caption #{rank}. It shows up in Results."
    except Exception as e:  # noqa: BLE001
        st.session_state["toast"] = f"Couldn't log that pick ({e})."


def render_captions(user_id: str, include_tags: bool) -> None:
    toast = st.session_state.pop("toast", None)
    if toast:
        st.toast(toast)

    platform = st.radio(
        "Posting to", list(PLATFORMS), horizontal=True, label_visibility="collapsed"
    )
    upload = st.file_uploader(
        "Drop a photo",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
        label_visibility="collapsed",
    )
    place = st.text_input(
        "📍 Place (optional)",
        placeholder="e.g. Klyde Warren Park, Dallas",
        help="Only needed when the photo has no location. Press Enter to regenerate.",
    )

    if upload is None:
        st.info("Drop a photo above. Your captions and hashtags will appear here by themselves.")
        return

    data = upload.getvalue()  # the original bytes, so EXIF is intact
    try:
        st.image(data, width=220)
    except Exception:  # noqa: BLE001 - e.g. HEIC the browser can't preview
        pass

    try:
        with st.spinner("Reading the photo, place and light..."):
            result = _generate(
                hashlib.sha256(data).hexdigest()[:16],
                data,
                user_id,
                platform,
                place.strip(),
                st.session_state.get("nonce", 0),
            )
    except ValueError as e:
        st.error(str(e))
        return
    except (LLMError, PipelineError) as e:
        st.error(f"The caption model had a problem: {e}")
        return
    except Exception as e:  # noqa: BLE001
        st.error("Something went wrong while generating captions.")
        st.exception(e)
        return

    ctx = result["context"]
    time_ctx = ctx.get("time") or {}
    chips = []
    if ctx.get("place"):
        chips.append(f"📍 {ctx['place']}")
    if time_ctx.get("golden_hour"):
        chips.append("🌇 golden hour")
    elif time_ctx.get("part_of_day"):
        chips.append(f"🕒 {time_ctx['part_of_day']}")
    if ctx.get("weather"):
        chips.append(f"🌤 {ctx['weather']}")
    if ctx.get("vibe"):
        chips.append("✨ " + ", ".join(ctx["vibe"][:3]))
    if chips:
        st.caption("  ·  ".join(chips))
    for warning in result["warnings"]:
        st.info(warning)

    tag_line = " ".join(result["hashtags"]["all"])
    generation_id = result["generation_id"]
    for cap in result["captions"]:
        with st.container(border=True):
            body = f"{cap['text']}\n\n{tag_line}" if include_tags and tag_line else cap["text"]
            st.code(body, language=None, wrap_lines=True)  # built-in copy button
            left, right = st.columns([3, 1])
            left.caption(f"{cap['rank']}. {cap['style'].replace('_', ' ')}")
            right.button(
                "Use this",
                key=f"use_{generation_id}_{cap['rank']}",
                on_click=_log_pick,
                args=(generation_id, cap["rank"]),
            )

    if tag_line:
        st.subheader("Hashtags")
        tiers = result["hashtags"]
        for name in ("location", "niche", "mid", "broad"):
            if tiers[name]:
                st.caption(f"{name}: " + " ".join(tiers[name]))
        st.code(tag_line, language=None, wrap_lines=True)

    if st.button("🔄 Another batch"):
        st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
        st.rerun()


def render_voice(user_id: str) -> None:
    store = get_store()
    st.write(
        "Paste a few of your real past captions, one per line. New captions are written "
        "to sound like you."
    )
    raw = st.text_area(
        "Past captions", height=180, label_visibility="collapsed", key="voice_text"
    )
    if st.button("Save samples"):
        added = store.add_voice_samples(user_id, raw.splitlines())
        st.success(f"Saved {added} new sample(s).")

    samples = store.list_voice_samples(user_id, limit=50)
    st.caption(f"{len(samples)} saved. The 12 most recent are used.")
    for sample in samples[:12]:
        st.write(f"• {sample}")
    if samples and st.button("Clear all samples"):
        store.clear_voice_samples(user_id)
        st.rerun()


def render_stats(user_id: str) -> None:
    store = get_store()
    m = store.metrics(user_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Batches", m["generations"])
    c2.metric("Captions used", m["picks"])
    c3.metric("Top-3 pick rate", f"{m['top3_pick_rate']:.0%}" if m["picks"] else "n/a")
    st.caption(
        "Top-3 pick rate: how often the caption you used was one of the top three ranked. "
        "Higher means the ranking matches your taste."
    )
    if m["picks_by_style"]:
        st.bar_chart(pd.Series(m["picks_by_style"], name="times used"))

    st.subheader("Log how a post did")
    recent = store.recent_picks(user_id, limit=10)
    if not recent:
        st.write("Use a caption first (tap “Use this”), then log its likes and reach here.")
    else:
        labels = {f"{p['id']} · #{p['rank']} · {p['text'][:50]}": p["id"] for p in recent}
        choice = st.selectbox("Which caption did you post?", list(labels))
        likes = st.number_input("Likes", min_value=0, step=1)
        reach = st.number_input("Reach (optional)", min_value=0, step=1)
        if st.button("Save result"):
            store.record_outcome(labels[choice], int(likes), int(reach) or None)
            st.success("Saved.")
    if m["avg_likes_by_style"]:
        st.write("Average likes by style")
        st.dataframe(pd.Series(m["avg_likes_by_style"], name="avg likes"))


def main() -> None:
    with st.sidebar:
        st.header("📸 SnapCaption")
        start_user = st.query_params.get("u", "default")
        user_id = (
            st.text_input(
                "Your handle",
                value=start_user,
                help="Keeps your voice samples and stats separate. Bookmark the page after setting it.",
            ).strip()
            or "default"
        )
        if user_id != start_user:
            st.query_params["u"] = user_id
        include_tags = st.toggle("Copy hashtags with each caption", value=True)
        st.caption("Photos aren't stored. Location is reduced to a place name.")

    tab_captions, tab_voice, tab_stats = st.tabs(["Captions", "My voice", "Results"])
    with tab_captions:
        render_captions(user_id, include_tags)
    with tab_voice:
        render_voice(user_id)
    with tab_stats:
        render_stats(user_id)


main()
