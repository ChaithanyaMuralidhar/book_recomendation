"""Excel/CSV import (validated, never auto-overwriting) and export (4-sheet workbook, template)."""
import io
from pathlib import Path

import pandas as pd

import config
from services.text_utils import (author_surname, clean_isbn, is_blank, is_valid_isbn, normalize_genre,
                                 normalize_title)

SHEETS = ["Book Catalogue", "Reading History", "Recommendations", "Scan Logs"]
_ALIASES = {
    "book_id": ["bookid", "id", "book id"],
    "title": ["book title", "name", "book name", "book"],
    "author": ["authors", "writer", "author name"],
    "genre": ["category", "genres", "type"],
    "subgenre": ["sub genre", "sub_genre", "subcategory"],
    "isbn": ["isbn13", "isbn_13", "isbn-13", "isbn10", "isbn 13"],
    "description": ["summary", "synopsis", "about"],
    "publication_year": ["year", "pub_year", "published", "publication year", "year published"],
    "language": ["lang"],
    "source": [], "scan_confidence": ["confidence"], "availability": ["available", "status"],
}
HISTORY_ALIASES = {"user_id": ["user", "userid", "user id"], "book_id": ["bookid", "book id"],
                   "title": ["book title", "book"], "status": [], "rating": ["stars"], "liked": [],
                   "read_date": ["date", "date_read", "read date"], "user_name": ["name"]}


class ExcelFormatError(ValueError):
    pass


def _canon(col: str, aliases: dict):
    c = str(col).strip().lower().replace("-", "_")
    for k, al in aliases.items():
        if c == k or c.replace("_", " ") == k.replace("_", " ") or c in al or c.replace("_", " ") in al:
            return k
    return None


def _read_table(source, name: str = None, prefer_sheet: str = None) -> pd.DataFrame:
    name = name or getattr(source, "name", None) or (str(source) if isinstance(source, (str, Path)) else "")
    ext = Path(name).suffix.lower()
    if ext not in config.SUPPORTED_TABLE_EXT:
        raise ExcelFormatError(f"Unsupported file type '{ext or 'unknown'}'. Use .xlsx or .csv.")
    try:
        if hasattr(source, "read"):
            data = source.read()
            source = io.BytesIO(data if isinstance(data, bytes) else bytes(data))
        if ext == ".csv":
            return pd.read_csv(source, dtype=str, keep_default_na=False)
        xl = pd.ExcelFile(source, engine="openpyxl")
        sheet = prefer_sheet if prefer_sheet in xl.sheet_names else xl.sheet_names[0]
        return xl.parse(sheet, dtype=str, keep_default_na=False)
    except ExcelFormatError:
        raise
    except Exception as exc:
        raise ExcelFormatError(f"The file could not be read as a valid spreadsheet ({type(exc).__name__}).") from exc


def read_books_file(source, name: str = None):
    """-> (clean DataFrame with BOOK_COLUMNS, list of human-readable issues)."""
    raw = _read_table(source, name, "Book Catalogue")
    issues = []
    mapping = {c: _canon(c, _ALIASES) for c in raw.columns}
    df = raw.rename(columns={c: m for c, m in mapping.items() if m}).copy()
    df = df.loc[:, ~df.columns.duplicated()]
    unknown = [c for c, m in mapping.items() if not m]
    if unknown:
        issues.append(f"Ignored unrecognised column(s): {', '.join(map(str, unknown))}.")
    if "title" not in df.columns:
        raise ExcelFormatError("Required column 'title' was not found. Download the template for the expected layout.")
    for col in config.BOOK_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            if col not in ("book_id", "scan_confidence", "availability", "source"):
                issues.append(f"Column '{col}' is missing; it will be left blank.")
    df = df[config.BOOK_COLUMNS].map(lambda v: "" if is_blank(v) else str(v).strip())
    blank = df["title"] == ""
    if blank.any():
        issues.append(f"Dropped {int(blank.sum())} row(s) with no title.")
    df = df[~blank].copy()
    bad_isbn = (df["isbn"] != "") & ~df["isbn"].map(is_valid_isbn)
    if bad_isbn.any():
        issues.append(f"{int(bad_isbn.sum())} row(s) have an invalid ISBN; the ISBN will be ignored.")
        df.loc[bad_isbn, "isbn"] = ""
    df["isbn"] = df["isbn"].map(lambda v: clean_isbn(v) if v else "")
    df["genre"] = df["genre"].map(normalize_genre)
    key = df["title"].map(normalize_title) + "|" + df["author"].map(author_surname)
    dup = key.duplicated()
    if dup.any():
        issues.append(f"Dropped {int(dup.sum())} duplicate row(s) inside the file.")
    df = df[~dup]
    df["source"] = df["source"].replace("", "excel_import")
    return df.reset_index(drop=True), issues


def read_history_file(source, name: str = None):
    raw = _read_table(source, name, "Reading History")
    df = raw.rename(columns={c: _canon(c, HISTORY_ALIASES) or c for c in raw.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    if "user_id" not in df.columns or not ({"book_id", "title"} & set(df.columns)):
        raise ExcelFormatError("Reading history needs 'user_id' and either 'book_id' or 'title' columns.")
    for c in ["book_id", "title", "status", "rating", "liked", "read_date"]:
        if c not in df.columns:
            df[c] = ""
    df = df.map(lambda v: "" if is_blank(v) else str(v).strip())
    df["status"] = df["status"].str.lower().replace("", "read")
    return df[(df["user_id"] != "") & ((df["book_id"] != "") | (df["title"] != ""))].reset_index(drop=True), []


def build_template() -> bytes:
    example = {"book_id": "", "title": "Example Title", "author": "Example Author", "genre": "Mystery",
               "subgenre": "Detective", "isbn": "", "description": "One or two sentences about the book.",
               "publication_year": 2020, "language": "English", "source": "manual", "scan_confidence": "",
               "availability": "Unknown"}
    return export_workbook(pd.DataFrame([example], columns=config.BOOK_COLUMNS), None, None, None)


def export_workbook(catalogue, history, recommendations, scan_logs) -> bytes:
    """4-sheet workbook. Missing inputs become empty sheets with headers so the layout is always the same."""
    from services.user_service import HISTORY_COLUMNS
    empties = {
        "Book Catalogue": pd.DataFrame(columns=config.BOOK_COLUMNS),
        "Reading History": pd.DataFrame(columns=HISTORY_COLUMNS),
        "Recommendations": pd.DataFrame(columns=["rank", "book_id", "title", "author", "genre", "score", "explanation"]),
        "Scan Logs": pd.DataFrame(columns=["timestamp", "input_type", "filename", "items_found", "items_saved", "notes"]),
    }
    frames = dict(zip(SHEETS, [catalogue, history, recommendations, scan_logs]))
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name in SHEETS:
            df = frames[name] if frames[name] is not None else empties[name]
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for i, col in enumerate(df.columns, 1):
                width = min(60, max(len(str(col)), *(len(str(v)) for v in df[col].head(50))) + 2) if len(df) else len(str(col)) + 2
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
    return buf.getvalue()
