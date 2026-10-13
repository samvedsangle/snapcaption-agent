import pytest
from fastapi.testclient import TestClient

from helpers import FakeLLM, make_deps, make_jpeg
from snapcaption import service
from snapcaption.config import Settings
from snapcaption.llm import LLMError


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    return tmp_path


@pytest.fixture
def client(env, monkeypatch):
    deps = make_deps()
    monkeypatch.setattr("snapcaption.service.default_deps", lambda settings: deps)
    from api import app

    return TestClient(app), deps


def post_photo(client, photo, key="secret", **kwargs):
    headers = {"X-API-Key": key} if key else {}
    return client.post(
        "/caption",
        files={"file": ("photo.jpg", photo, "image/jpeg")},
        headers=headers,
        **kwargs,
    )


# ------------------------------------------------------------------- service
def test_service_saves_the_batch_and_uses_saved_voice(env, jpeg_gps):
    settings = Settings(db_path=str(env / "svc.db"))
    deps = make_deps()
    store = service.get_store(settings)
    store.add_voice_samples("sam", ["coffee first, opinions later"])

    result = service.caption_photo(jpeg_gps, user_id="sam", settings=settings, deps=deps)
    assert isinstance(result["generation_id"], int)
    assert store.get_generation(result["generation_id"])["user_id"] == "sam"
    assert "coffee first, opinions later" in deps.llm.prompts[-2]


def test_service_validates_input(env, jpeg_gps):
    settings = Settings(db_path=str(env / "svc.db"), max_upload_mb=1)
    with pytest.raises(ValueError, match="No image"):
        service.caption_photo(b"", settings=settings, deps=make_deps())
    with pytest.raises(ValueError, match="limit"):
        service.caption_photo(b"0" * (1024 * 1024 + 1), settings=settings, deps=make_deps())


def test_service_without_a_key_explains_what_to_do(env, jpeg_gps):
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        service.caption_photo(jpeg_gps, settings=Settings(db_path=str(env / "x.db")))


def test_non_image_is_reported_before_the_missing_key(env):
    with pytest.raises(ValueError, match="photo"):
        service.caption_photo(b"hello", settings=Settings(db_path=str(env / "x.db")))


def test_render_text(env, jpeg_gps):
    result = service.caption_photo(jpeg_gps, settings=Settings(db_path=str(env / "r.db")), deps=make_deps())
    text = service.render_text(result)
    assert text.startswith("Captions") and "1. " in text and "Hashtags:" in text
    assert "Klyde Warren Park" in text


# ----------------------------------------------------------------------- API
def test_health_needs_no_key(client):
    api, _ = client
    assert api.get("/health").json() == {"status": "ok"}


def test_caption_requires_the_api_key(client, jpeg_gps):
    api, _ = client
    assert post_photo(api, jpeg_gps, key=None).status_code == 401
    assert post_photo(api, jpeg_gps, key="wrong").status_code == 401


def test_caption_json(client, jpeg_gps):
    api, _ = client
    r = post_photo(api, jpeg_gps, data={"user_id": "sam", "platform": "instagram"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["captions"]) == 10 and len(body["choices"]) == 10
    assert isinstance(body["generation_id"], int)
    assert body["context"]["place"].startswith("Klyde Warren Park")


def test_caption_text_format(client, jpeg_gps):
    api, _ = client
    r = post_photo(api, jpeg_gps, params={"format": "text"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "1. " in r.text and "#dallas" in r.text


def test_caption_rejects_non_images(client):
    api, _ = client
    r = post_photo(api, b"hello world")
    assert r.status_code == 400 and "photo" in r.json()["detail"]


def test_caption_rejects_oversized_uploads(client):
    api, _ = client
    assert post_photo(api, b"0" * (1024 * 1024 + 10)).status_code == 413


def test_caption_without_configured_key_is_open(client, jpeg_gps, monkeypatch):
    api, _ = client
    monkeypatch.setenv("API_KEY", "")
    assert post_photo(api, jpeg_gps, key=None).status_code == 200


def test_model_problems_are_bad_gateway(env, monkeypatch, jpeg_gps):
    def boom(settings):
        raise LLMError("GEMINI_API_KEY is not set.")

    monkeypatch.setattr("snapcaption.service.default_deps", boom)
    from api import app

    r = post_photo(TestClient(app), jpeg_gps)
    assert r.status_code == 502 and "GEMINI_API_KEY" in r.json()["detail"]


def test_empty_generation_is_bad_gateway(env, monkeypatch, jpeg_gps):
    deps = make_deps(llm=FakeLLM(empty_captions=True))
    monkeypatch.setattr("snapcaption.service.default_deps", lambda settings: deps)
    from api import app

    assert post_photo(TestClient(app), jpeg_gps).status_code == 502


def test_pick_outcome_and_metrics_flow(client, jpeg_gps):
    api, _ = client
    auth = {"X-API-Key": "secret"}
    body = post_photo(api, jpeg_gps, data={"user_id": "sam"}).json()
    gid = body["generation_id"]

    by_rank = api.post("/pick", json={"generation_id": gid, "rank": 2}, headers=auth)
    assert by_rank.status_code == 200
    # a shortcut only knows the text the user chose from `choices`
    by_text = api.post("/pick", json={"generation_id": gid, "text": body["choices"][0]}, headers=auth)
    assert by_text.status_code == 200

    out = api.post("/outcome", json={"pick_id": by_text.json()["pick_id"], "likes": 120, "reach": 900}, headers=auth)
    assert out.status_code == 200

    m = api.get("/metrics", params={"user_id": "sam"}, headers=auth).json()
    assert m["picks"] == 2 and m["top3_pick_rate"] == 1.0 and m["avg_likes_by_style"]


def test_pick_errors(client, jpeg_gps):
    api, _ = client
    auth = {"X-API-Key": "secret"}
    assert api.post("/pick", json={"generation_id": 999, "rank": 1}, headers=auth).status_code == 404
    gid = post_photo(api, jpeg_gps).json()["generation_id"]
    assert api.post("/pick", json={"generation_id": gid, "rank": 99}, headers=auth).status_code == 400
    assert api.post("/outcome", json={"pick_id": 999, "likes": 1}, headers=auth).status_code == 404
    assert api.post("/pick", json={"generation_id": gid, "rank": 1}).status_code == 401


def test_voice_endpoint_changes_the_next_batch(client, jpeg_gps):
    api, deps = client
    auth = {"X-API-Key": "secret"}
    r = api.post("/voice", json={"user_id": "sam", "samples": ["sunday reset, no notes"]}, headers=auth)
    assert r.json() == {"added": 1}
    post_photo(api, jpeg_gps, data={"user_id": "sam"})
    assert "sunday reset, no notes" in deps.llm.prompts[-2]
    post_photo(api, jpeg_gps, data={"user_id": "someone-else"})
    assert "sunday reset, no notes" not in deps.llm.prompts[-2]


def test_rewrite_endpoint(client):
    api, deps = client
    auth = {"X-API-Key": "secret"}
    r = api.post("/rewrite", json={"text": "came for tacos", "how": "funnier"}, headers=auth)
    assert r.status_code == 200 and r.json() == {"options": ["take one", "take two", "take three"]}
    assert "funnier" in deps.llm.prompts[-1].lower()
    assert api.post("/rewrite", json={"text": "x", "how": "louder"}, headers=auth).status_code == 400
    assert api.post("/rewrite", json={"text": "", "how": "shorter"}, headers=auth).status_code == 400
    assert api.post("/rewrite", json={"text": "x"}).status_code == 401


def test_caption_note_language_and_alt_text(client, jpeg_gps):
    api, deps = client
    r = post_photo(api, jpeg_gps, data={"note": "first day at the new job", "language": "Hindi"})
    body = r.json()
    assert r.status_code == 200 and body["language"] == "Hindi" and body["alt_text"]
    writer = deps.llm.prompts[-2]
    assert "first day at the new job" in writer and "in Hindi" in writer
