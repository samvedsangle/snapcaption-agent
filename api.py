"""HTTP API for share-sheet shortcuts (iPhone Shortcuts, Android HTTP Shortcuts) and anything else.

Run locally:  uvicorn api:app --reload
"""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from snapcaption import service
from snapcaption.config import load_settings
from snapcaption.llm import LLMError
from snapcaption.pipeline import PipelineError

app = FastAPI(title="SnapCaption", version="0.1.0")


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = load_settings().api_key
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/caption")
def caption(
    file: UploadFile = File(...),
    user_id: str = Form("default"),
    platform: str = Form("instagram"),
    place: str = Form(""),
    note: str = Form(""),
    language: str = Form("English"),
    n: int = Form(10),
    format: str = Query("json", pattern="^(json|text)$"),
    _: None = Depends(require_key),
):
    """Upload a photo, get ranked captions and hashtags.

    `note` is what the moment is ("my sister's birthday dinner"); `language` picks the
    caption language. format=json returns captions, hashtags, `choices` (caption + hashtags,
    ready to paste), `alt_text` and `generation_id`. format=text returns a plain-text list.
    """
    limit = load_settings().max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="Photo is too large.")
    try:
        result = service.caption_photo(
            data,
            filename=file.filename or "",
            user_id=user_id,
            platform=platform,
            place_hint=place,
            n=n,
            note=note,
            language=language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (LLMError, PipelineError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    if format == "text":
        return PlainTextResponse(service.render_text(result))
    return result


class RewriteIn(BaseModel):
    text: str
    how: str = "shorter"  # shorter | funnier | more_personal | add_emoji | more_professional
    user_id: str = "default"
    platform: str = "instagram"
    language: str = "English"


@app.post("/rewrite")
def rewrite(body: RewriteIn, _: None = Depends(require_key)) -> dict:
    """Three new versions of one caption."""
    try:
        options = service.rewrite_caption(
            body.text, body.how, user_id=body.user_id, platform=body.platform, language=body.language
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"options": options}


class PickIn(BaseModel):
    generation_id: int
    rank: Optional[int] = None
    text: Optional[str] = None  # alternative to rank: the chosen text from `choices`
    final_text: Optional[str] = None  # what you actually posted, if you edited it


@app.post("/pick")
def pick(body: PickIn, _: None = Depends(require_key)) -> dict:
    try:
        pick_id = service.get_store().record_pick(
            body.generation_id, rank=body.rank, text=body.text, final_text=body.final_text
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"pick_id": pick_id}


class OutcomeIn(BaseModel):
    pick_id: int
    likes: Optional[int] = None
    reach: Optional[int] = None


@app.post("/outcome")
def outcome(body: OutcomeIn, _: None = Depends(require_key)) -> dict:
    try:
        service.get_store().record_outcome(body.pick_id, body.likes, body.reach)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"ok": True}


class VoiceIn(BaseModel):
    user_id: str = "default"
    samples: list[str]


@app.post("/voice")
def voice(body: VoiceIn, _: None = Depends(require_key)) -> dict:
    added = service.get_store().add_voice_samples(body.user_id.strip() or "default", body.samples)
    return {"added": added}


@app.get("/metrics")
def metrics(user_id: Optional[str] = None, _: None = Depends(require_key)) -> dict:
    return service.get_store().metrics(user_id)
