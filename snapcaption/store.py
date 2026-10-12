"""SQLite storage: voice samples, caption batches, picks and post results.

No photos and no coordinates are stored, only caption text and a coarse place label.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MIN_SAMPLE_LEN = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, text)
);
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    place TEXT,
    captions_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES generations(id),
    rank INTEGER NOT NULL,
    style TEXT,
    text TEXT NOT NULL,
    final_text TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    pick_id INTEGER PRIMARY KEY REFERENCES picks(id),
    likes INTEGER,
    reach INTEGER,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_generations_user ON generations(user_id);
CREATE INDEX IF NOT EXISTS idx_picks_generation ON picks(generation_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _strip_hashtags(text: str) -> str:
    return re.sub(r"(^|\s)#\w+", " ", text).strip()


class Store:
    def __init__(self, path: str):
        self.path = path
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        # One short-lived connection per operation: safe across Streamlit/FastAPI threads.
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------ voice samples
    def add_voice_samples(self, user_id: str, samples) -> int:
        added = 0
        with self._conn() as conn:
            for raw in samples:
                text = str(raw or "").strip()
                if len(text) < MIN_SAMPLE_LEN:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO voice_samples (user_id, text, created_at) VALUES (?, ?, ?)",
                    (user_id, text, _now()),
                )
                added += cur.rowcount
        return added

    def list_voice_samples(self, user_id: str, limit: int = 12) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT text FROM voice_samples WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [r["text"] for r in rows]

    def clear_voice_samples(self, user_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM voice_samples WHERE user_id = ?", (user_id,))

    # -------------------------------------------------------------- generations
    def save_generation(self, user_id: str, platform: str, result: dict) -> int:
        captions = [
            {"rank": c["rank"], "text": c["text"], "style": c.get("style"), "score": c.get("score")}
            for c in result.get("captions", [])
        ]
        place = (result.get("context") or {}).get("place") or None
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO generations (user_id, platform, place, captions_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, platform, place, json.dumps(captions, ensure_ascii=False), _now()),
            )
            return int(cur.lastrowid)

    def get_generation(self, generation_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM generations WHERE id = ?", (generation_id,)).fetchone()
        if row is None:
            return None
        gen = dict(row)
        gen["captions"] = json.loads(gen.pop("captions_json"))
        return gen

    # -------------------------------------------------------------------- picks
    def record_pick(
        self,
        generation_id: int,
        *,
        rank: Optional[int] = None,
        text: Optional[str] = None,
        final_text: Optional[str] = None,
    ) -> int:
        """Log which caption was used, by rank or by the (possibly hashtag-suffixed) text."""
        gen = self.get_generation(generation_id)
        if gen is None:
            raise KeyError(f"No caption batch with id {generation_id}.")
        captions = gen["captions"]

        if rank is not None:
            chosen = next((c for c in captions if c["rank"] == rank), None)
            if chosen is None:
                raise ValueError(f"Rank must be between 1 and {len(captions)}.")
        elif text:
            given = text.strip()
            matches = [c for c in captions if given.startswith(c["text"].strip())]
            if not matches:
                raise ValueError("That text doesn't match any caption in this batch.")
            chosen = max(matches, key=lambda c: len(c["text"]))
        else:
            raise ValueError("Send either rank or text.")

        final = (final_text or "").strip() or None
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO picks (generation_id, rank, style, text, final_text, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (generation_id, chosen["rank"], chosen.get("style"), chosen["text"], final, _now()),
            )
            pick_id = int(cur.lastrowid)

        if final:  # an edited caption is the best voice sample there is
            edited = _strip_hashtags(final)
            if edited and edited != chosen["text"].strip():
                self.add_voice_samples(gen["user_id"], [edited])
        return pick_id

    def recent_picks(self, user_id: str, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT p.id, p.rank, p.style, p.text, p.final_text, p.created_at, "
                "o.likes, o.reach FROM picks p "
                "JOIN generations g ON g.id = p.generation_id "
                "LEFT JOIN outcomes o ON o.pick_id = p.id "
                "WHERE g.user_id = ? ORDER BY p.id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------- outcomes
    def record_outcome(self, pick_id: int, likes: Optional[int], reach: Optional[int]) -> None:
        with self._conn() as conn:
            if conn.execute("SELECT 1 FROM picks WHERE id = ?", (pick_id,)).fetchone() is None:
                raise KeyError(f"No pick with id {pick_id}.")
            conn.execute(
                "INSERT INTO outcomes (pick_id, likes, reach, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(pick_id) DO UPDATE SET likes = excluded.likes, "
                "reach = excluded.reach, updated_at = excluded.updated_at",
                (pick_id, likes, reach, _now()),
            )

    # ------------------------------------------------------------------ metrics
    def metrics(self, user_id: Optional[str] = None) -> dict:
        where, args = ("WHERE g.user_id = ?", (user_id,)) if user_id else ("", ())
        with self._conn() as conn:
            generations = conn.execute(
                f"SELECT COUNT(*) FROM generations g {where}", args
            ).fetchone()[0]
            picks = conn.execute(
                "SELECT p.rank, p.style, o.likes FROM picks p "
                "JOIN generations g ON g.id = p.generation_id "
                f"LEFT JOIN outcomes o ON o.pick_id = p.id {where}",
                args,
            ).fetchall()

        n = len(picks)
        by_style: dict[str, int] = {}
        likes_by_style: dict[str, list[int]] = {}
        for p in picks:
            style = p["style"] or "other"
            by_style[style] = by_style.get(style, 0) + 1
            if p["likes"] is not None:
                likes_by_style.setdefault(style, []).append(p["likes"])

        return {
            "generations": generations,
            "picks": n,
            "pick_rate": round(n / generations, 3) if generations else 0.0,
            "top3_pick_rate": round(sum(1 for p in picks if p["rank"] <= 3) / n, 3) if n else 0.0,
            "avg_pick_rank": round(sum(p["rank"] for p in picks) / n, 2) if n else None,
            "picks_by_style": dict(sorted(by_style.items())),
            "avg_likes_by_style": {
                s: round(sum(v) / len(v), 1) for s, v in sorted(likes_by_style.items())
            },
        }
