from snapcaption.hashtags import build_hashtags, location_tags, sanitize_tag
from snapcaption.ranking import heuristic_penalty, select_top
from snapcaption.schemas import Place

PARK = Place(
    label="Klyde Warren Park, Dallas, Texas",
    venue="Klyde Warren Park",
    neighborhood="Arts District",
    city="Dallas",
    region="Texas",
    source="exif",
)


def good(i=9):
    return {"voice": i, "hook": i, "originality": i}


# ------------------------------------------------------------------- ranking
def test_style_cap_is_enforced():
    cands = [{"text": f"witty line {w}", "style": "witty"} for w in "abcdefgh"]
    cands += [{"text": f"minimal {w} zz", "style": "minimal"} for w in "wxyz"]
    scores = {i: good(9) for i in range(8)} | {i: good(3) for i in range(8, 12)}
    top = select_top(cands, scores, k=6, per_style_cap=3)
    styles = [c["style"] for c in top]
    assert len(top) == 6 and styles.count("witty") == 3 and styles.count("minimal") == 3


def test_cap_relaxes_when_variety_runs_out():
    cands = [{"text": f"only style {w} here", "style": "witty"} for w in "abcdefg"]
    top = select_top(cands, {i: good(7) for i in range(7)}, k=5, per_style_cap=3)
    assert len(top) == 5


def test_ranks_are_sequential_and_sorted():
    cands = [{"text": f"{s} caption {s}{s}", "style": s} for s in ["a", "b", "c", "d"]]
    scores = {0: good(2), 1: good(9), 2: good(5), 3: good(7)}
    top = select_top(cands, scores, k=4)
    assert [c["rank"] for c in top] == [1, 2, 3, 4]
    assert [c["text"][0] for c in top] == ["b", "d", "c", "a"]


def test_near_duplicates_are_dropped():
    cands = [
        {"text": "Golden hour hits different here", "style": "witty"},
        {"text": "golden hour hits different here!", "style": "hype"},
        {"text": "Quiet city, loud sky", "style": "minimal"},
    ]
    top = select_top(cands, {0: good(8), 1: good(7), 2: good(6)}, k=3)
    assert len(top) == 2


def test_cliche_and_hashtag_are_penalised():
    cands = [
        {"text": "living my best life downtown", "style": "hype"},
        {"text": "honestly the skyline did all the work", "style": "deadpan"},
        {"text": "weekend mood #goldenhour", "style": "witty"},
    ]
    top = select_top(cands, {i: good(7) for i in range(3)}, k=3)
    assert top[0]["text"].startswith("honestly")
    assert "cliche" in next(c for c in top if "best life" in c["text"])["flags"]


def test_machine_sounding_lines_are_penalised():
    _, flags = heuristic_penalty("Explore. Discover. Repeat.", "instagram")
    assert "slogan" in flags
    _, flags = heuristic_penalty("Lost in the beauty of it all", "instagram")
    assert "cliche" in flags
    assert heuristic_penalty("came for tacos, stayed for the skyline", "instagram") == (0, [])


def test_length_penalty_depends_on_platform():
    text = "x" * 170
    ig, _ = heuristic_penalty(text, "instagram")  # limit 150: long
    tt, _ = heuristic_penalty(text, "tiktok")  # limit 100: over 1.5x, very long
    assert 0 < ig < tt


def test_too_short_is_rejected():
    assert heuristic_penalty("ok", "instagram")[0] >= 10


def test_missing_scores_fall_back_to_neutral():
    cands = [{"text": f"candidate number {w} xx", "style": "witty"} for w in "abcd"]
    assert len(select_top(cands, {}, k=3, per_style_cap=5)) == 3


def test_voice_score_ignored_without_voice_samples():
    cands = [{"text": "alpha beta gamma", "style": "a"}, {"text": "delta epsilon zeta", "style": "b"}]
    scores = {0: {"voice": 10, "hook": 3, "originality": 3}, 1: {"voice": 1, "hook": 6, "originality": 6}}
    assert select_top(cands, scores, k=2, has_voice=True)[0]["text"].startswith("alpha")
    assert select_top(cands, scores, k=2, has_voice=False)[0]["text"].startswith("delta")


# ------------------------------------------------------------------ hashtags
def test_sanitize_tag():
    assert sanitize_tag("#Golden Hour!") == "goldenhour"
    assert sanitize_tag("Café_Life") == "cafélife"
    for bad in ("f4f", "Like4Like", "x", "12345", "a" * 31, None, 42):
        assert sanitize_tag(bad) is None


def test_budget_is_spread_across_tiers():
    pool = {
        "broad": ["travel", "photography", "citylife", "weekend"],
        "mid": ["urbanexplorer", "skylinelovers", "cityscapes", "goldenhourphotography"],
        "niche": ["dallasskyline", "dfwphotographer", "texasgoldenhour", "arttexas"],
    }
    tags = build_hashtags(pool, PARK, "instagram")
    assert len(tags["all"]) == 8
    assert all(len(tags[t]) == 2 for t in ("broad", "mid", "niche", "location"))
    assert all(t.startswith("#") for t in tags["all"])


def test_x_gets_two_tags_location_first():
    tags = build_hashtags({"niche": ["dallasskyline"], "broad": ["travel"]}, PARK, "x")
    assert len(tags["all"]) == 2
    assert tags["location"] and tags["niche"]


def test_tag_lives_in_one_tier_only():
    tags = build_hashtags({"broad": ["dallas", "travel"]}, PARK, "instagram")
    assert tags["all"].count("#dallas") == 1
    assert "#dallas" in tags["location"]


def test_engagement_bait_is_removed():
    tags = build_hashtags({"broad": ["follow4follow", "travel", "likeforlike"]}, None, "instagram")
    assert tags["all"] == ["#travel"]


def test_string_and_garbage_pools_are_tolerated():
    tags = build_hashtags({"broad": "travel, photo  citylife", "mid": 7, "niche": None}, None, "instagram")
    assert tags["broad"] == ["#travel", "#photo", "#citylife"]
    assert build_hashtags(None, None, "instagram")["all"] == []


def test_user_typed_place_becomes_location_tags():
    place = Place(label="Klyde Warren Park, Dallas", source="user")
    assert location_tags(place) == ["klydewarrenpark", "dallas"]


def test_location_tags_from_geocoded_place():
    tags = location_tags(PARK)
    assert tags[:2] == ["dallas", "klydewarrenpark"]
    assert "dallastexas" in tags


def test_country_is_the_last_resort_location_tag():
    assert location_tags(Place(label="Iceland", country="Iceland", source="exif")) == ["iceland"]
    assert location_tags(Place()) == []


def test_misspelled_and_plural_near_copies_are_dropped():
    tags = build_hashtags({"niche": ["klyderenpark", "klydewarren", "cityscapes", "cityscape", "dallasparks"]}, PARK, "instagram")
    assert "#klyderenpark" not in tags["all"] and "#klydewarren" not in tags["all"] and "#klydewarrenpark" in tags["all"]
    assert tags["all"].count("#cityscape") + tags["all"].count("#cityscapes") == 1
    assert "#dallasparks" in tags["all"]  # related, but not a copy of #dallas
