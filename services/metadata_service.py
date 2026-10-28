"""Book metadata lookup: local catalogue + bundled sample data first (offline), Open Library second (optional).

Online data is marked source='openlibrary'; nothing is ever fabricated - if no match is found, the result is
found=False and fields stay blank/'Unknown'. After repeated network failures the service stays offline.
"""
import re
from dataclasses import dataclass, field

import pandas as pd
import requests

import config
from services.text_utils import (
    author_surname, clean_isbn, is_blank, is_valid_isbn, normalize_text, normalize_title, similarity,
)

_GENRE_KEYWORDS = [  # priority order: first keyword found in Open Library subjects wins
    ("science fiction", "Science Fiction"), ("fantasy", "Fantasy"), ("mystery", "Mystery"), ("detective", "Mystery"),
    ("thriller", "Thriller"), ("horror", "Horror"), ("romance", "Romance"), ("biograph", "Biography"),
    ("autobiograph", "Biography"), ("self-help", "Self-Help"), ("self help", "Self-Help"), ("history", "History"),
    ("computer", "Technology"), ("programming", "Technology"), ("artificial intelligence", "Technology"),
    ("physics", "Science"), ("astronomy", "Science"), ("biology", "Science"), ("science", "Science"),
    ("classic", "Classic Literature"), ("literature", "Classic Literature"),
]


@dataclass
class MetadataResult:
    found: bool = False
    record: dict = field(default_factory=dict)
    source: str = "none"      # local | openlibrary | none
    score: float = 0.0


def genre_from_subjects(subjects) -> str:
    text = " ".join(subjects or []).lower()
    for kw, genre in _GENRE_KEYWORDS:
        if kw in text:
            return genre
    return "Unknown"


class MetadataService:
    def __init__(self, catalogue_provider=None, sample_csv=config.SAMPLE_BOOKS_CSV, online=True, session=None):
        self.catalogue_provider = catalogue_provider
        self.sample_csv = sample_csv
        self.online = online
        self.session = session or requests
        self.failures = 0
        self._sample = None

    # ---------- local ----------
    def _local_df(self) -> pd.DataFrame:
        if self._sample is None:
            try:
                self._sample = pd.read_csv(self.sample_csv, dtype=str, keep_default_na=False)
            except (OSError, ValueError):
                self._sample = pd.DataFrame(columns=config.BOOK_COLUMNS)
        frames = [self._sample]
        if self.catalogue_provider:
            try:
                frames.insert(0, self.catalogue_provider().astype(str).replace({"<NA>": ""}))
            except Exception:
                pass
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates(subset=["title", "author"]) if len(df) else df

    @staticmethod
    def _rec(row) -> dict:
        return {k: ("" if is_blank(row.get(k)) else row.get(k)) for k in
                ["title", "author", "genre", "subgenre", "isbn", "description", "publication_year", "language"]}

    def match_text(self, raw_text: str, threshold: float = 0.85):
        """Find a known book whose title tokens (and optionally author) appear in noisy OCR text."""
        toks = normalize_text(raw_text).split()
        if len(toks) < 2:
            return None, 0.0
        tokset = set(toks)
        best, best_s = None, 0.0
        for _, b in self._local_df().iterrows():
            tt = normalize_title(b["title"]).split()
            if not tt:
                continue
            hit = sum(1 for t in tt if t in tokset or any(similarity(t, o) >= 0.8 for o in toks if abs(len(o) - len(t)) <= 2))
            s = hit / len(tt)
            sn = author_surname(b["author"])
            author_hit = bool(sn) and (sn in tokset or any(similarity(sn, o) >= 0.8 for o in toks))
            if len(tt) < 2:  # single-word titles need the author to avoid false positives
                s = s * (1.0 if author_hit else 0.5)
            elif author_hit:
                s = min(1.0, s + 0.1)
            if s > best_s:
                best, best_s = b, s
        return (best, best_s) if best is not None and best_s >= threshold else (None, best_s)

    def lookup_local(self, title="", author="", isbn="", threshold=0.8) -> MetadataResult:
        df = self._local_df()
        if not len(df):
            return MetadataResult()
        isbn = clean_isbn(isbn)
        if is_valid_isbn(isbn):
            hit = df[df["isbn"].map(clean_isbn) == isbn]
            if len(hit):
                return MetadataResult(True, self._rec(hit.iloc[0]), "local", 1.0)
        if is_blank(title):
            return MetadataResult()
        best, best_s = None, 0.0
        for _, b in df.iterrows():
            s = similarity(title, b["title"])
            if not is_blank(author) and not is_blank(b["author"]):
                s = 0.8 * s + 0.2 * similarity(author, b["author"])
            if s > best_s:
                best, best_s = b, s
        if best is not None and best_s >= threshold:
            return MetadataResult(True, self._rec(best), "local", best_s)
        return MetadataResult()

    # ---------- online ----------
    def _get(self, url, params=None):
        if not self.online or self.failures >= config.ONLINE_MAX_FAILURES:
            return None
        try:
            r = self.session.get(url, params=params, timeout=config.ONLINE_TIMEOUT_S,
                                 headers={"User-Agent": "BookRecStudentProject/1.0"})
            r.raise_for_status()
            self.failures = 0
            return r.json()
        except (requests.RequestException, ValueError):
            self.failures += 1
            return None

    def lookup_online(self, title="", author="", isbn="") -> MetadataResult:
        params = {"limit": 3, "fields": "key,title,author_name,first_publish_year,subject,isbn,language"}
        isbn = clean_isbn(isbn)
        if is_valid_isbn(isbn):
            params["q"] = f"isbn:{isbn}"
        elif not is_blank(title):
            params["title"] = str(title)
            if not is_blank(author):
                params["author"] = str(author)
        else:
            return MetadataResult()
        data = self._get(config.OPENLIBRARY_SEARCH_URL, params)
        best, best_s = None, 0.0
        for d in (data or {}).get("docs") or []:
            s = 1.0 if "q" in params else similarity(title, d.get("title", ""))
            if s > best_s:
                best, best_s = d, s
        if best is None or best_s < 0.6:
            return MetadataResult()
        rec = {"title": best.get("title", ""), "author": ", ".join((best.get("author_name") or [])[:2]),
               "genre": genre_from_subjects(best.get("subject")), "subgenre": "", "isbn": isbn,
               "description": self._description(best.get("key")),
               "publication_year": best.get("first_publish_year") or "",
               "language": (best.get("language") or [""])[0]}
        return MetadataResult(True, rec, "openlibrary", best_s)

    def _description(self, work_key) -> str:
        if not work_key:
            return ""
        data = self._get(f"{config.OPENLIBRARY_BASE}{work_key}.json")
        d = (data or {}).get("description", "")
        d = d.get("value", "") if isinstance(d, dict) else d
        return re.sub(r"\s+", " ", str(d)).strip()[:600]

    # ---------- combined ----------
    def identify(self, title="", author="", isbn="", raw_text="") -> MetadataResult:
        res = self.lookup_local(title, author, isbn)
        if res.found:
            return res
        if raw_text:
            row, s = self.match_text(raw_text)
            if row is not None:
                return MetadataResult(True, self._rec(row), "local", s)
        return self.lookup_online(title, author, isbn)
