"""Central configuration. All paths are relative to the project root (no absolute paths)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VR_INBOX_DIR = DATA_DIR / "vr_inbox"
EXPORT_XLSX = DATA_DIR / "book_catalogue.xlsx"
DB_PATH = Path(os.getenv("BOOKREC_DB", str(DATA_DIR / "bookrec.db")))
SAMPLE_BOOKS_CSV = DATA_DIR / "sample_books.csv"
SAMPLE_USERS_CSV = DATA_DIR / "sample_users.csv"
SAMPLE_HISTORY_CSV = DATA_DIR / "sample_history.csv"

# OCR (Tesseract). Set TESSERACT_CMD to the full path of tesseract(.exe) if it is not on PATH.
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

# Metadata (Open Library is free and needs no key; everything works offline without it)
OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
OPENLIBRARY_BASE = "https://openlibrary.org"
ONLINE_TIMEOUT_S = 5
ONLINE_MAX_FAILURES = 2  # after this many network failures, stay offline for the session

BOOK_COLUMNS = [
    "book_id", "title", "author", "genre", "subgenre", "isbn", "description",
    "publication_year", "language", "source", "scan_confidence", "availability",
]
HISTORY_STATUSES = ["read", "reading", "interested"]

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_TABLE_EXT = {".xlsx", ".xlsm", ".csv"}

KNOWN_GENRES = [
    "Biography", "Classic Literature", "Fantasy", "History", "Horror", "Mystery",
    "Romance", "Science", "Science Fiction", "Self-Help", "Technology", "Thriller",
]

# Recommendation weights (normalised at run time; collaborative is dropped if data is insufficient)
DEFAULT_WEIGHTS = {"content": 0.35, "genre": 0.30, "cluster": 0.10, "author": 0.10, "collab": 0.15}
MIN_CF_USERS = 5            # users with enough signals required before collaborative filtering is used
MIN_CF_SIGNALS_PER_USER = 3
MATCH_THRESHOLD = 0.80      # catalogue matching for library photos
