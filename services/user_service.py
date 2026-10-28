"""Users, reading history (kept separate from the scanned catalogue) and recommendation log."""
import pandas as pd

import config
from services.db import Database
from services.text_utils import is_blank

HISTORY_COLUMNS = ["user_id", "book_id", "title", "author", "genre", "status", "rating", "liked", "read_date"]


def _split(s):
    return [x.strip() for x in str(s or "").split("|") if x.strip()]


class UserService:
    def __init__(self, db: Database):
        self.db = db

    def create_user(self, name, pref_genres=(), pref_authors=(), user_id=None) -> str:
        if is_blank(name):
            raise ValueError("User name is required.")
        with self.db.connect() as c:
            uid = user_id
            if not uid or c.execute("SELECT 1 FROM users WHERE user_id=?", (uid,)).fetchone():
                nums = [int(r[0][1:]) for r in c.execute("SELECT user_id FROM users") if r[0][1:].isdigit()]
                uid = f"U{(max(nums) if nums else 0) + 1:03d}"
            c.execute("INSERT INTO users (user_id,name,pref_genres,pref_authors) VALUES (?,?,?,?)",
                      (uid, str(name).strip(), "|".join(pref_genres), "|".join(pref_authors)))
        return uid

    def list_users(self) -> pd.DataFrame:
        with self.db.connect() as c:
            rows = [dict(r) for r in c.execute("SELECT user_id,name,pref_genres,pref_authors FROM users ORDER BY user_id")]
        return pd.DataFrame(rows, columns=["user_id", "name", "pref_genres", "pref_authors"])

    def get_user(self, user_id):
        with self.db.connect() as c:
            r = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not r:
            return None
        return {"user_id": r["user_id"], "name": r["name"],
                "pref_genres": _split(r["pref_genres"]), "pref_authors": _split(r["pref_authors"])}

    def update_preferences(self, user_id, pref_genres, pref_authors):
        with self.db.connect() as c:
            c.execute("UPDATE users SET pref_genres=?, pref_authors=? WHERE user_id=?",
                      ("|".join(pref_genres), "|".join(pref_authors), user_id))

    def set_status(self, user_id, book_id, status, rating=None, liked=None, read_date=None):
        if status not in config.HISTORY_STATUSES:
            raise ValueError(f"status must be one of {config.HISTORY_STATUSES}")
        rating = None if is_blank(rating) else int(rating)
        if rating is not None and not 1 <= rating <= 5:
            raise ValueError("rating must be between 1 and 5")
        if liked not in (None, -1, 0, 1):
            raise ValueError("liked must be 1, 0, -1 or None")
        with self.db.connect() as c:
            if not c.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone():
                raise KeyError(f"Unknown user {user_id}")
            if not c.execute("SELECT 1 FROM books WHERE book_id=?", (book_id,)).fetchone():
                raise KeyError(f"Unknown book {book_id}")
            c.execute(
                "INSERT INTO reading_history (user_id,book_id,status,rating,liked,read_date) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(user_id,book_id) DO UPDATE SET status=excluded.status, rating=excluded.rating, "
                "liked=excluded.liked, read_date=excluded.read_date",
                (user_id, book_id, status, rating, liked, None if is_blank(read_date) else str(read_date)))

    def remove_history(self, user_id, book_id):
        with self.db.connect() as c:
            c.execute("DELETE FROM reading_history WHERE user_id=? AND book_id=?", (user_id, book_id))

    def get_history(self, user_id=None) -> pd.DataFrame:
        sql = ("SELECT h.user_id,h.book_id,b.title,b.author,b.genre,h.status,h.rating,h.liked,h.read_date "
               "FROM reading_history h JOIN books b ON b.book_id=h.book_id")
        args = ()
        if user_id is not None:
            sql, args = sql + " WHERE h.user_id=?", (user_id,)
        with self.db.connect() as c:
            rows = [dict(r) for r in c.execute(sql + " ORDER BY h.id", args)]
        return pd.DataFrame(rows, columns=HISTORY_COLUMNS)

    def genre_frequency(self, user_id) -> pd.Series:
        h = self.get_history(user_id)
        h = h[h["status"] == "read"]
        return h["genre"].value_counts() if len(h) else pd.Series(dtype=int)

    # ---------- recommendation log ----------
    def log_recommendations(self, user_id, recs: pd.DataFrame):
        with self.db.connect() as c:
            for _, r in recs.iterrows():
                c.execute("INSERT INTO recommendation_log (user_id,book_id,score,explanation) VALUES (?,?,?,?)",
                          (user_id, r["book_id"], float(r["score"]), r["explanation"]))

    def mark_selected(self, user_id, book_id):
        with self.db.connect() as c:
            c.execute("UPDATE recommendation_log SET selected=1 WHERE user_id=? AND book_id=?", (user_id, book_id))

    def recommendation_log(self, user_id=None) -> pd.DataFrame:
        sql = ("SELECT l.user_id,l.book_id,b.title,l.score,l.explanation,l.created_at,l.selected "
               "FROM recommendation_log l JOIN books b ON b.book_id=l.book_id")
        args = ()
        if user_id is not None:
            sql, args = sql + " WHERE l.user_id=?", (user_id,)
        with self.db.connect() as c:
            rows = [dict(r) for r in c.execute(sql + " ORDER BY l.id DESC", args)]
        return pd.DataFrame(rows, columns=["user_id", "book_id", "title", "score", "explanation", "created_at", "selected"])
