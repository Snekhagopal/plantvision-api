"""SQLite storage for past analyses.

Stores only what is needed to review a past result: time, plant name, status,
predicted condition, score and the recommendations text. Images, free-text
notes and any user identifiers are never stored.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    plant_name      TEXT NOT NULL,
    status          TEXT NOT NULL,
    status_label    TEXT NOT NULL,
    predicted_label TEXT,
    condition_name  TEXT,
    confidence      REAL,
    summary         TEXT NOT NULL,
    recommendations TEXT NOT NULL
)
"""


class HistoryError(RuntimeError):
    """Raised for any database failure; message is safe to show users."""


@dataclass
class HistoryRecord:
    id: int
    created_at: str
    plant_name: str
    status: str
    status_label: str
    predicted_label: str | None
    condition_name: str | None
    confidence: float | None
    summary: str
    recommendations: dict


class HistoryStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(SCHEMA)
        except (sqlite3.Error, OSError) as exc:
            raise HistoryError("The history database could not be opened.") from exc

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _run(self, sql: str, params=()):
        try:
            with self._connect() as conn:
                cur = conn.execute(sql, params)
                return cur.fetchall(), cur.lastrowid, cur.rowcount
        except sqlite3.Error as exc:
            raise HistoryError("A history database operation failed.") from exc

    def add(self, plant_name: str, status: str, status_label: str, summary: str, recommendations: dict,
            predicted_label: str | None = None, condition_name: str | None = None,
            confidence: float | None = None) -> int:
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _, rowid, _ = self._run(
            "INSERT INTO analyses (created_at, plant_name, status, status_label, predicted_label, condition_name,"
            " confidence, summary, recommendations) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (created, plant_name[:80], status, status_label, predicted_label, condition_name, confidence,
             summary, json.dumps(recommendations)),
        )
        return int(rowid)

    def recent(self, limit: int = 100) -> list[HistoryRecord]:
        rows, _, _ = self._run("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (int(limit),))
        return [self._record(r) for r in rows]

    def get(self, record_id: int) -> HistoryRecord | None:
        rows, _, _ = self._run("SELECT * FROM analyses WHERE id = ?", (int(record_id),))
        return self._record(rows[0]) if rows else None

    def delete(self, record_id: int) -> bool:
        _, _, count = self._run("DELETE FROM analyses WHERE id = ?", (int(record_id),))
        return count > 0

    def clear(self) -> int:
        _, _, count = self._run("DELETE FROM analyses")
        return count

    @staticmethod
    def _record(row) -> HistoryRecord:
        try:
            recs = json.loads(row[9])
        except (TypeError, ValueError):
            recs = {}
        return HistoryRecord(*row[:9], recommendations=recs)
