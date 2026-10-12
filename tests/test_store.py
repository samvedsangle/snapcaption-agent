import pytest

from snapcaption.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "data" / "t.db"))  # also proves the parent dir is created


def result_with(n=10):
    return {
        "context": {"place": "Dallas"},
        "captions": [
            {"rank": r, "text": f"caption number {r}", "style": "witty" if r % 2 else "minimal", "score": 9 - r}
            for r in range(1, n + 1)
        ],
    }


def test_voice_samples_dedupe_and_newest_first(store):
    assert store.add_voice_samples("sam", ["first one", "second one", "first one", " ", "x"]) == 2
    assert store.list_voice_samples("sam") == ["second one", "first one"]
    assert store.list_voice_samples("other") == []
    store.clear_voice_samples("sam")
    assert store.list_voice_samples("sam") == []


def test_voice_sample_limit(store):
    store.add_voice_samples("sam", [f"caption {i}" for i in range(20)])
    assert len(store.list_voice_samples("sam", limit=12)) == 12


def test_pick_by_rank_and_metrics(store):
    gid = store.save_generation("sam", "instagram", result_with())
    store.record_pick(gid, rank=2)
    store.record_pick(gid, rank=7)
    m = store.metrics("sam")
    assert m["generations"] == 1 and m["picks"] == 2
    assert m["pick_rate"] == 2.0
    assert m["top3_pick_rate"] == 0.5
    assert m["avg_pick_rank"] == 4.5
    assert m["picks_by_style"] == {"minimal": 1, "witty": 1}


def test_pick_by_text_prefers_longest_match(store):
    gid = store.save_generation("sam", "instagram", result_with())
    # "caption number 1" is also a prefix of "caption number 10"
    pick_id = store.record_pick(gid, text="caption number 10\n\n#travel #city")
    assert store.recent_picks("sam")[0]["id"] == pick_id
    assert store.recent_picks("sam")[0]["rank"] == 10


def test_pick_errors(store):
    with pytest.raises(KeyError):
        store.record_pick(999, rank=1)
    gid = store.save_generation("sam", "instagram", result_with())
    with pytest.raises(ValueError):
        store.record_pick(gid, rank=99)
    with pytest.raises(ValueError):
        store.record_pick(gid, text="something unrelated")


def test_edited_final_text_becomes_a_voice_sample(store):
    gid = store.save_generation("sam", "instagram", result_with())
    store.record_pick(gid, rank=1, final_text="caption number 1, but in my words")
    assert store.list_voice_samples("sam") == ["caption number 1, but in my words"]
    store.record_pick(gid, rank=2, final_text="caption number 2")  # unchanged: not saved
    assert len(store.list_voice_samples("sam")) == 1


def test_outcomes_upsert_and_average_by_style(store):
    gid = store.save_generation("sam", "instagram", result_with())
    a = store.record_pick(gid, rank=1)  # witty
    b = store.record_pick(gid, rank=3)  # witty
    store.record_outcome(a, 100, 900)
    store.record_outcome(b, 50, None)
    store.record_outcome(b, 60, 500)  # logging again overwrites
    assert store.metrics("sam")["avg_likes_by_style"] == {"witty": 80.0}
    with pytest.raises(KeyError):
        store.record_outcome(12345, 1, 1)


def test_metrics_are_scoped_per_user(store):
    g1 = store.save_generation("sam", "instagram", result_with())
    g2 = store.save_generation("kim", "instagram", result_with())
    store.record_pick(g1, rank=1)
    store.record_pick(g2, rank=9)
    assert store.metrics("sam")["top3_pick_rate"] == 1.0
    assert store.metrics("kim")["top3_pick_rate"] == 0.0
    assert store.metrics()["picks"] == 2


def test_empty_metrics(store):
    m = store.metrics("nobody")
    assert m["generations"] == 0 and m["top3_pick_rate"] == 0.0 and m["picks_by_style"] == {}
