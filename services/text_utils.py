"""Shared text helpers: normalisation, fuzzy similarity, ISBN validation, safe type coercion."""
import re
import string
import unicodedata
from difflib import SequenceMatcher

_ARTICLES = ("the ", "a ", "an ")
_GENRE_MAP = {
    "sci fi": "Science Fiction", "scifi": "Science Fiction", "sf": "Science Fiction",
    "science fiction": "Science Fiction", "classic": "Classic Literature", "classics": "Classic Literature",
    "literature": "Classic Literature", "self help": "Self-Help", "selfhelp": "Self-Help",
    "tech": "Technology", "computers": "Technology", "programming": "Technology",
    "bio": "Biography", "memoir": "Biography", "autobiography": "Biography",
    "crime": "Mystery", "detective": "Mystery", "suspense": "Thriller", "sci": "Science",
}


def is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    return str(v).strip().lower() in ("", "nan", "none", "nat", "<na>")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_text(s) -> str:
    if is_blank(s):
        return ""
    s = strip_accents(str(s)).lower().replace("&", " and ")
    s = s.replace("'", "").replace("\u2019", "").replace(".", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_title(s) -> str:
    n = normalize_text(s)
    for art in _ARTICLES:
        if n.startswith(art):
            return n[len(art):]
    return n


def author_surname(author) -> str:
    """Surname of the first author ('J. K. Rowling' / 'Rowling, J.K.' -> 'rowling')."""
    if is_blank(author):
        return ""
    first = re.split(r";| and | & ", str(author))[0].strip()
    if "," in first:
        first = first.split(",")[0]
    toks = normalize_text(first).split()
    return toks[-1] if toks else ""


def normalize_author(author) -> str:
    return "" if is_blank(author) else re.sub(r"\s+", " ", str(author)).strip()


def similarity(a, b) -> float:
    """Fuzzy similarity in [0,1] (order-insensitive tokens + character ratio)."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    r1 = SequenceMatcher(None, na, nb).ratio()
    r2 = SequenceMatcher(None, " ".join(sorted(na.split())), " ".join(sorted(nb.split()))).ratio()
    return max(r1, r2)


def normalize_genre(g) -> str:
    if is_blank(g):
        return "Unknown"
    s = re.sub(r"\s+", " ", str(g).strip())
    key = re.sub(r"[-_]", " ", s.lower())
    if key in _GENRE_MAP:
        return _GENRE_MAP[key]
    if s.lower() == "unknown":
        return "Unknown"
    return string.capwords(s) if s.islower() or s.isupper() else s


def clean_isbn(s) -> str:
    return re.sub(r"[^0-9Xx]", "", "" if is_blank(s) else str(s)).upper()


def is_valid_isbn(s) -> bool:
    c = clean_isbn(s)
    if len(c) == 10 and c[:9].isdigit():
        total = sum((10 if ch == "X" else int(ch)) * (10 - i) for i, ch in enumerate(c))
        return total % 11 == 0
    if len(c) == 13 and c.isdigit():
        return sum(int(ch) * (1 if i % 2 == 0 else 3) for i, ch in enumerate(c)) % 10 == 0
    return False


_ISBN_RE = re.compile(r"(?<![\w-])(?:97[89][- ]?)?\d(?:[- ]?\d){8}[- ]?[\dXx](?![\w])")


def extract_isbns(text: str) -> list:
    """Return checksum-valid ISBNs found in free text (e.g. OCR output)."""
    found = []
    for m in _ISBN_RE.finditer(text or ""):
        c = clean_isbn(m.group(0))
        if is_valid_isbn(c) and c not in found:
            found.append(c)
    return found


def to_year(v):
    try:
        y = int(float(str(v).strip()[:4])) if not is_blank(v) else None
    except ValueError:
        return None
    return y if y is not None and 1000 <= y <= 2100 else None


def to_confidence(v):
    try:
        return None if is_blank(v) else min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return None


def confidence_label(c) -> str:
    c = to_confidence(c)
    if c is None:
        return "n/a"
    return "High" if c >= 0.75 else "Medium" if c >= 0.5 else "Low (verify)"
