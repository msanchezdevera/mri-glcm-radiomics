"""Persistent memory backed by SQLite.

Two layers:
  - short-term: the message history of each session (windowed when sent to the LLM)
  - long-term: every case analyzed through the API, queryable by the agent
Traces of every agent turn are stored too, which is what evals and debugging read.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY, filename TEXT, created_at TEXT,
    features TEXT, prediction TEXT, note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT, idx INTEGER, message TEXT, PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS traces (
    session_id TEXT, created_at TEXT, trace TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    # --- long-term: cases ---------------------------------------------------
    def add_case(self, filename: str, features: dict, prediction: dict) -> str:
        case_id = uuid.uuid4().hex[:8]
        with self._lock:
            self._db.execute("INSERT INTO cases VALUES (?,?,?,?,?,'')",
                             (case_id, filename, _now(), json.dumps(features), json.dumps(prediction)))
            self._db.commit()
        return case_id

    def get_case(self, case_id: str) -> dict | None:
        row = self._db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return self._case(row) if row else None

    def list_cases(self, label: str | None = None, limit: int = 10) -> list[dict]:
        rows = self._db.execute("SELECT * FROM cases ORDER BY created_at DESC, rowid DESC").fetchall()
        cases = [self._case(r, with_features=False) for r in rows]
        if label:
            cases = [c for c in cases if c["prediction"]["label"] == label]
        return cases[:limit]

    def annotate_case(self, case_id: str, note: str) -> bool:
        with self._lock:
            cur = self._db.execute("UPDATE cases SET note=? WHERE id=?", (note, case_id))
            self._db.commit()
        return cur.rowcount == 1

    @staticmethod
    def _case(row: sqlite3.Row, with_features: bool = True) -> dict:
        case = {"case_id": row["id"], "filename": row["filename"], "created_at": row["created_at"],
                "prediction": json.loads(row["prediction"]), "note": row["note"]}
        if with_features:
            case["features"] = json.loads(row["features"])
        return case

    # --- short-term: session messages ---------------------------------------
    def history(self, session_id: str) -> list[dict]:
        rows = self._db.execute("SELECT message FROM messages WHERE session_id=? ORDER BY idx",
                                (session_id,)).fetchall()
        return [json.loads(r["message"]) for r in rows]

    def append(self, session_id: str, messages: list[dict]) -> None:
        with self._lock:
            start = self._db.execute("SELECT COUNT(*) FROM messages WHERE session_id=?",
                                     (session_id,)).fetchone()[0]
            self._db.executemany("INSERT INTO messages VALUES (?,?,?)",
                                 [(session_id, start + i, json.dumps(m)) for i, m in enumerate(messages)])
            self._db.commit()

    # --- traces -------------------------------------------------------------
    def save_trace(self, session_id: str, trace: list[dict]) -> None:
        with self._lock:
            self._db.execute("INSERT INTO traces VALUES (?,?,?)", (session_id, _now(), json.dumps(trace)))
            self._db.commit()


def window(history: list[dict], max_messages: int = 30) -> list[dict]:
    """Keep the most recent messages, cutting only at a user turn so that a tool
    call is never separated from its result."""
    if len(history) <= max_messages:
        return history
    tail = history[-max_messages:]
    for i, m in enumerate(tail):
        if m["role"] == "user":
            return tail[i:]
    return tail
