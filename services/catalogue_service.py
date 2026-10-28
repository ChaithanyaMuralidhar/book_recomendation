"""Single source of truth for books: CRUD, duplicate detection, scan logs, catalogue matching."""
from dataclasses import dataclass, field

import pandas as pd

import config
from services.db import Database
from services.text_utils import (
    author_surname, clean_isbn, is_blank, is_valid_isbn, normalize_author, normalize_genre,
    normalize_title, similarity, to_confidence, to_year,
)

_TEXT = ["title", "author", "genre", "subgenre", "isbn", "description", "language", "source", "availability"]


@dataclass
class AddResult:
    status: str  # added | duplicate | merged
    book_id: str
    changed_fields: list = field(default_factory=list)


def _empty(field_name, value) -> bool:
    return is_blank(value) or (field_name in ("genre", "language", "availability") and str(value) == "Unknown") \
        or (field_name == "source" and str(value) == "manual")


class CatalogueService:
    def __init__(self, db: Database):
        self.db = db

    # ---------- cleaning ----------
    @staticmethod
    def clean(rec: dict) -> dict:
        title = "" if is_blank(rec.get("title")) else " ".join(str(rec["title"]).split())
        if not title:
            raise ValueError("A book must have a title.")
        isbn = clean_isbn(rec.get("isbn"))
        return {
            "title": title,
            "author": normalize_author(rec.get("author")),
            "genre": normalize_genre(rec.get("genre")),
            "subgenre": "" if is_blank(rec.get("subgenre")) else str(rec["subgenre"]).strip(),
            "isbn": isbn if is_valid_isbn(isbn) else "",
            "description": "" if is_blank(rec.get("description")) else str(rec["description"]).strip(),
            "publication_year": to_year(rec.get("publication_year")),
            "language": "Unknown" if is_blank(rec.get("language")) else str(rec["language"]).strip(),
            "source": "manual" if is_blank(rec.get("source")) else str(rec["source"]).strip(),
            "scan_confidence": to_confidence(rec.get("scan_confidence")),
            "availability": "Unknown" if is_blank(rec.get("availability")) else str(rec["availability"]).strip(),
        }

    def _next_id(self, conn) -> str:
        nums = [int(r[0][1:]) for r in conn.execute("SELECT book_id FROM books") if r[0][1:].isdigit()]
        return f"B{(max(nums) if nums else 0) + 1:03d}"

    # ---------- queries ----------
    def list_books(self) -> pd.DataFrame:
        with self.db.connect() as c:
            rows = [dict(r) for r in c.execute(
                f"SELECT {', '.join(config.BOOK_COLUMNS)} FROM books ORDER BY book_id")]
        df = pd.DataFrame(rows, columns=config.BOOK_COLUMNS)
        df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").astype("Int64")
        return df

    def get_book(self, book_id):
        with self.db.connect() as c:
            r = c.execute(f"SELECT {', '.join(config.BOOK_COLUMNS)} FROM books WHERE book_id=?", (book_id,)).fetchone()
        return dict(r) if r else None

    def count(self) -> int:
        with self.db.connect() as c:
            return c.execute("SELECT COUNT(*) FROM books").fetchone()[0]

    def find_duplicate(self, title, author="", isbn="", exclude_id=None):
        """book_id of an existing entry with the same ISBN, or same normalised title and compatible author."""
        isbn = clean_isbn(isbn)
        nt, sn = normalize_title(title), author_surname(author)
        with self.db.connect() as c:
            if is_valid_isbn(isbn):
                r = c.execute("SELECT book_id FROM books WHERE isbn=? AND book_id IS NOT ?", (isbn, exclude_id)).fetchone()
                if r:
                    return r[0]
            if not nt:
                return None
            for r in c.execute("SELECT book_id, norm_surname FROM books WHERE norm_title=? AND book_id IS NOT ?",
                               (nt, exclude_id)):
                if not sn or not r["norm_surname"] or sn == r["norm_surname"]:
                    return r["book_id"]
        return None

    # ---------- writes ----------
    def add_book(self, record: dict, on_duplicate: str = "merge_empty") -> AddResult:
        """Insert a book. Duplicates are never overwritten: 'skip' ignores, 'merge_empty' only fills blank fields."""
        rec = self.clean(record)
        dup = self.find_duplicate(rec["title"], rec["author"], rec["isbn"])
        if dup:
            changed = []
            if on_duplicate == "merge_empty":
                existing = self.get_book(dup)
                upd = {f: rec[f] for f in _TEXT + ["publication_year"]
                       if _empty(f, existing[f]) and not _empty(f, rec[f]) and f != "title"}
                if upd:
                    self._write_update(dup, {**existing, **upd})
                    changed = sorted(upd)
            return AddResult("merged" if changed else "duplicate", dup, changed)
        with self.db.connect() as c:
            wanted = record.get("book_id")
            taken = wanted and c.execute("SELECT 1 FROM books WHERE book_id=?", (wanted,)).fetchone()
            bid = wanted if (not is_blank(wanted) and not taken) else self._next_id(c)
            c.execute(
                "INSERT INTO books (book_id,title,author,genre,subgenre,isbn,description,publication_year,language,"
                "source,scan_confidence,availability,norm_title,norm_surname) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (bid, rec["title"], rec["author"], rec["genre"], rec["subgenre"], rec["isbn"], rec["description"],
                 rec["publication_year"], rec["language"], rec["source"], rec["scan_confidence"], rec["availability"],
                 normalize_title(rec["title"]), author_surname(rec["author"])))
        return AddResult("added", bid)

    def add_many(self, records, on_duplicate="merge_empty") -> dict:
        counts = {"added": 0, "merged": 0, "duplicate": 0, "invalid": 0}
        for r in records:
            try:
                counts[self.add_book(r, on_duplicate).status] += 1
            except ValueError:
                counts["invalid"] += 1
        return counts

    def _write_update(self, book_id, rec):
        rec = self.clean(rec)
        with self.db.connect() as c:
            c.execute(
                "UPDATE books SET title=?,author=?,genre=?,subgenre=?,isbn=?,description=?,publication_year=?,language=?,"
                "source=?,scan_confidence=?,availability=?,norm_title=?,norm_surname=?,updated_at=CURRENT_TIMESTAMP "
                "WHERE book_id=?",
                (rec["title"], rec["author"], rec["genre"], rec["subgenre"], rec["isbn"], rec["description"],
                 rec["publication_year"], rec["language"], rec["source"], rec["scan_confidence"], rec["availability"],
                 normalize_title(rec["title"]), author_surname(rec["author"]), book_id))

    def update_book(self, book_id: str, fields: dict):
        """Explicit user edit (may overwrite). Refuses edits that would duplicate another book."""
        existing = self.get_book(book_id)
        if not existing:
            raise KeyError(f"Unknown book_id {book_id}")
        merged = {**existing, **fields}
        rec = self.clean(merged)
        if self.find_duplicate(rec["title"], rec["author"], rec["isbn"], exclude_id=book_id):
            raise ValueError("This edit would duplicate another book already in the catalogue.")
        self._write_update(book_id, merged)

    def delete_book(self, book_id: str):
        with self.db.connect() as c:
            c.execute("DELETE FROM books WHERE book_id=?", (book_id,))

    # ---------- scan logs ----------
    def log_scan(self, input_type, filename, found, saved, notes=""):
        with self.db.connect() as c:
            c.execute("INSERT INTO scan_logs (input_type,filename,items_found,items_saved,notes) VALUES (?,?,?,?,?)",
                      (input_type, filename, int(found), int(saved), notes))

    def list_scan_logs(self) -> pd.DataFrame:
        with self.db.connect() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT timestamp,input_type,filename,items_found,items_saved,notes FROM scan_logs ORDER BY id DESC")]
        return pd.DataFrame(rows, columns=["timestamp", "input_type", "filename", "items_found", "items_saved", "notes"])

    # ---------- matching detected books to the catalogue ----------
    def match_records(self, records, threshold: float = config.MATCH_THRESHOLD, catalogue_df=None) -> list:
        """Annotate detected records with matched_book_id / match_score / match_status
        (matched | uncertain | unmatched). ISBN equality is an exact match; otherwise fuzzy title (+author)."""
        cat = self.list_books() if catalogue_df is None else catalogue_df
        out = []
        for rec in records:
            best_id, best = None, 0.0
            isbn = clean_isbn(rec.get("isbn"))
            title, author = rec.get("title"), rec.get("author")
            for _, b in cat.iterrows():
                if is_valid_isbn(isbn) and isbn == b["isbn"]:
                    best_id, best = b["book_id"], 1.0
                    break
                s = similarity(title, b["title"])
                nt, nb = normalize_title(title), normalize_title(b["title"])
                if min(len(nt), len(nb)) >= 6 and (nt in nb or nb in nt):
                    s = max(s, 0.9)
                if not is_blank(author) and not is_blank(b["author"]):
                    s = 0.8 * s + 0.2 * similarity(author, b["author"])
                if s > best:
                    best_id, best = b["book_id"], s
            status = "matched" if best >= threshold else "uncertain" if best >= 0.65 else "unmatched"
            out.append({**rec, "matched_book_id": best_id if status != "unmatched" else None,
                        "match_score": round(best, 3), "match_status": status})
        return out
