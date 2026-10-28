"""SQLite persistence layer (one short-lived connection per operation)."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    book_id TEXT PRIMARY KEY, title TEXT NOT NULL, author TEXT DEFAULT '', genre TEXT DEFAULT 'Unknown',
    subgenre TEXT DEFAULT '', isbn TEXT DEFAULT '', description TEXT DEFAULT '', publication_year INTEGER,
    language TEXT DEFAULT 'Unknown', source TEXT DEFAULT 'manual', scan_confidence REAL,
    availability TEXT DEFAULT 'Unknown', norm_title TEXT, norm_surname TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_books_norm ON books(norm_title);
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY, name TEXT NOT NULL, pref_genres TEXT DEFAULT '', pref_authors TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS reading_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
    status TEXT NOT NULL, rating INTEGER, liked INTEGER, read_date TEXT, UNIQUE(user_id, book_id));
CREATE TABLE IF NOT EXISTS recommendation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, book_id TEXT NOT NULL, score REAL,
    explanation TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, selected INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS scan_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP, input_type TEXT,
    filename TEXT, items_found INTEGER, items_saved INTEGER, notes TEXT);
"""


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def reset(self):
        with self.connect() as c:
            for t in ("recommendation_log", "reading_history", "scan_logs", "users", "books"):
                c.execute(f"DELETE FROM {t}")
