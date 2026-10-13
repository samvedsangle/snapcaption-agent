from pathlib import Path

from streamlit.testing.v1 import AppTest

from helpers import make_deps, make_jpeg
from snapcaption.store import Store

APP = str(Path(__file__).resolve().parents[1] / "app_streamlit.py")


def new_app(monkeypatch, tmp_path, deps=None):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    if deps is not None:
        monkeypatch.setattr("snapcaption.service.default_deps", lambda settings: deps)
    return AppTest.from_file(APP, default_timeout=60)


def test_app_renders_without_a_photo(monkeypatch, tmp_path):
    at = new_app(monkeypatch, tmp_path).run()
    assert not at.exception
    assert len(at.tabs) == 3
    assert any("Drop a photo" in i.value for i in at.info)


def test_handle_comes_from_the_url(monkeypatch, tmp_path):
    at = new_app(monkeypatch, tmp_path)
    at.query_params["u"] = "samved"
    at.run()
    assert not at.exception
    assert at.sidebar.text_input[0].value == "samved"


def test_upload_shows_captions_then_logs_a_pick(monkeypatch, tmp_path):
    deps = make_deps()
    photo = make_jpeg(lat=32.7894, lon=-96.8014, taken="2026:10:10 18:05:00", size=(66, 50))
    at = new_app(monkeypatch, tmp_path, deps).run()

    at.file_uploader[0].set_value(("dallas.jpg", photo, "image/jpeg")).run()
    assert not at.exception

    # ten caption cards plus one hashtag block, each with a built-in copy button
    assert len(at.code) == 11
    assert "#dallas" in at.code[0].value  # hashtags are copied with the caption by default
    chips = " ".join(c.value for c in at.caption)
    assert "Klyde Warren Park, Dallas, Texas" in chips and "golden hour" in chips

    use = [b for b in at.button if b.label == "Use this"]
    assert len(use) == 10
    use[1].click().run()
    assert not at.exception
    assert Store(str(tmp_path / "app.db")).metrics("default")["picks"] == 1
    assert deps.llm.vision_calls == 1  # the rerun after the click came from cache


def test_bad_upload_shows_a_friendly_error(monkeypatch, tmp_path):
    at = new_app(monkeypatch, tmp_path, make_deps()).run()
    at.file_uploader[0].set_value(("notes.jpg", b"definitely not a photo", "image/jpeg")).run()
    assert not at.exception
    assert any("photo" in e.value for e in at.error)
    assert len(at.code) == 0


def test_missing_api_key_is_explained_in_the_ui(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    at = new_app(monkeypatch, tmp_path).run()
    photo = make_jpeg(size=(67, 51))
    at.file_uploader[0].set_value(("p.jpg", photo, "image/jpeg")).run()
    assert not at.exception
    assert any("GEMINI_API_KEY" in e.value for e in at.error)
    assert any(b.label == "🔄 Try again" for b in at.button)
