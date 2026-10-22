import sqlite3

import pytest

from src.database.history import HistoryError, HistoryStore


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "h.db")


def add(store, plant="Tomato", **kw):
    return store.add(plant, "likely_match", "Likely match", "summary", {"summary": "s", "sections": []}, **kw)


def test_insert_and_retrieve(store):
    rid = add(store, predicted_label="Tomato___Early_blight", condition_name="Tomato early blight", confidence=0.91)
    rec = store.get(rid)
    assert rec.plant_name == "Tomato"
    assert rec.condition_name == "Tomato early blight"
    assert rec.confidence == pytest.approx(0.91)
    assert rec.recommendations == {"summary": "s", "sections": []}
    assert rec.created_at.endswith("+00:00")


def test_recent_returns_newest_first(store):
    ids = [add(store, plant=p) for p in ("A", "B", "C")]
    assert [r.id for r in store.recent()] == ids[::-1]
    assert len(store.recent(limit=2)) == 2


def test_delete_and_clear(store):
    a, b = add(store), add(store)
    assert store.delete(a) is True
    assert store.get(a) is None
    assert store.delete(a) is False
    assert store.clear() == 1
    assert store.recent() == []


def test_schema_has_no_image_or_personal_columns(store):
    with sqlite3.connect(store.path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(analyses)")}
    assert not cols & {"image", "image_path", "email", "user", "notes", "location"}


def test_unopenable_database_raises_history_error(tmp_path):
    directory = tmp_path / "is_a_dir"
    directory.mkdir()
    with pytest.raises(HistoryError):
        HistoryStore(directory)


def test_operation_failure_raises_history_error(store):
    with sqlite3.connect(store.path) as conn:
        conn.execute("DROP TABLE analyses")
    with pytest.raises(HistoryError):
        store.recent()
