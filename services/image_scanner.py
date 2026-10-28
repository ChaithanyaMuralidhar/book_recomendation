"""Image pipeline: load -> preprocess -> locate book regions -> OCR -> parse title/author/ISBN -> enrich.

Region detection is classical computer vision (edges/contours, vertical-edge projection for shelf spines).
It is a heuristic, not a trained detector: overlapping books, glare, curved spines and tiny text will fail,
and every result is therefore shown for human review with a confidence value.
"""
import io
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from services import ocr_service
from services.ocr_service import OCRError, OCRUnavailableError
from services.text_utils import extract_isbns

MAX_REGIONS = 25
_NAME_RE = re.compile(r"^[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*){1,3}$")


class InvalidImageError(ValueError):
    pass


@dataclass
class ScanReport:
    records: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    annotated: np.ndarray = None   # RGB image with numbered boxes
    unidentified: int = 0
    ocr_available: bool = True


# ------------------------------------------------------------------ loading / preprocessing
def load_image(source) -> np.ndarray:
    """bytes | path | file-like -> BGR ndarray. Raises InvalidImageError for corrupt/unsupported input."""
    try:
        if isinstance(source, (str, Path)):
            im = Image.open(source)
        else:
            data = source.read() if hasattr(source, "read") else bytes(source)
            im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception as exc:
        raise InvalidImageError(f"Could not read the file as an image ({type(exc).__name__}).") from exc
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def preprocess(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def crop_fraction(img, left=0.0, right=1.0, top=0.0, bottom=1.0):
    h, w = img.shape[:2]
    x0, x1, y0, y1 = int(left * w), int(right * w), int(top * h), int(bottom * h)
    return img[y0:y1, x0:x1] if (x1 - x0 > 20 and y1 - y0 > 20) else img


# ------------------------------------------------------------------ region detection
def _spine_segments(blur: np.ndarray) -> list:
    h, w = blur.shape
    sob = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    edge = (sob > sob.mean() + 1.5 * sob.std()).astype(np.uint8) * 255
    vert = cv2.morphologyEx(edge, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 4))))
    prof = vert.sum(axis=0) / 255.0 / h
    win = max(3, w // 150)
    prof = np.convolve(prof, np.ones(win) / win, mode="same")
    if prof.max() <= 0:
        return []
    min_gap = max(8, int(0.03 * w))
    thr = max(0.2, 0.4 * prof.max())
    cuts = []
    for x in np.argsort(-prof):
        if prof[x] < thr:
            break
        if all(abs(x - c) >= min_gap for c in cuts):
            cuts.append(int(x))
    cuts = sorted(set([0] + cuts + [w]))
    return [(a, 0, b - a, h) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_gap]


def _contour_boxes(blur: np.ndarray) -> list:
    h, w = blur.shape
    edges = cv2.dilate(cv2.Canny(blur, 40, 120), np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if 0.05 <= bw * bh / (w * h) <= 0.97 and 0.25 <= bh / max(bw, 1) <= 4.0:
            boxes.append((x, y, bw, bh))
    kept = []
    for b in sorted(boxes, key=lambda b: -b[2] * b[3]):  # non-maximum suppression
        if all(_overlap(b, k) < 0.6 for k in kept):
            kept.append(b)
    return sorted(kept, key=lambda b: (b[0], b[1]))


def _overlap(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, min(a[2] * a[3], b[2] * b[3]))


def detect_book_regions(img: np.ndarray, mode: str = "auto") -> list:
    """mode: 'single' (whole image), 'shelf' (vertical spines), 'auto' (shelf if wide & >=3 spines, else contours)."""
    h, w = img.shape[:2]
    full = [(0, 0, w, h)]
    if mode == "single":
        return full
    blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if mode in ("auto", "shelf"):
        segs = _spine_segments(blur)
        if len(segs) >= 3 and (mode == "shelf" or w / h >= 1.2):
            return segs[:MAX_REGIONS]
    if mode == "shelf":
        return full
    return (_contour_boxes(blur) or full)[:MAX_REGIONS]


# ------------------------------------------------------------------ text parsing
def _usable(l) -> bool:
    alnum = re.sub(r"[^A-Za-z0-9]", "", l.text)
    return len(alnum) >= 3 and len(alnum) / max(len(l.text.replace(" ", "")), 1) >= 0.6 and l.conf >= 0.35


def _tidy(text: str) -> str:
    text = re.sub(r"[^\w\s:'’,&.\-!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" :;,.-")
    return string.capwords(text.lower()) if text.isupper() else text


def parse_lines(lines) -> dict:
    """Heuristic: title = biggest-font line group; author = 'by X' line or a name-shaped line."""
    raw = " | ".join(l.text for l in lines)
    out = {"title": "", "author": "", "isbn": (extract_isbns(raw) or [""])[0], "raw_text": raw, "ocr_conf": 0.0}
    good = [l for l in lines if _usable(l)]
    if not good:
        return out
    author_line = None
    for l in good:
        m = re.match(r"^(?:written\s+)?by\s+(.+)$", l.text, re.I)
        if m:
            out["author"], author_line = _tidy(m.group(1)), l
    pool = [l for l in good if l is not author_line]
    if not pool:
        return out
    max_h = max(l.height for l in pool)
    tall = sorted([l for l in pool if l.height >= 0.7 * max_h], key=lambda l: l.top)
    chain, last = [tall[0]], tall[0]
    for l in tall[1:]:  # contiguous block around the tallest line
        if l.top - last.top <= 3.0 * max_h:
            chain.append(l)
            last = l
    out["title"] = _tidy(" ".join(l.text for l in chain))
    w = [max(len(l.text), 1) for l in chain]
    out["ocr_conf"] = sum(l.conf * x for l, x in zip(chain, w)) / sum(w)
    if not out["author"]:
        for l in pool:
            if l not in chain and _NAME_RE.match(_tidy(l.text)) and len(l.text) <= 40:
                out["author"] = _tidy(l.text)
                break
    return out


# ------------------------------------------------------------------ scanner
class ImageScanner:
    def __init__(self, metadata=None, ocr=ocr_service):
        self.metadata = metadata
        self.ocr = ocr

    def build_record(self, parsed: dict, region_index: int = 0) -> dict:
        title, author, isbn = parsed["title"], parsed["author"], parsed["isbn"]
        res = self.metadata.identify(title, author, isbn, parsed["raw_text"]) if self.metadata else None
        ocr_c = float(parsed.get("ocr_conf") or 0.0)
        if res is not None and res.found:
            r = res.record
            rec = {"title": r.get("title") or title, "author": r.get("author") or author,
                   "genre": r.get("genre") or "Unknown", "subgenre": r.get("subgenre", ""),
                   "isbn": r.get("isbn") or isbn, "description": r.get("description", ""),
                   "publication_year": r.get("publication_year", ""), "language": r.get("language", ""),
                   "source": f"image_scan+{res.source}"}
            conf = 0.4 * ocr_c + 0.6 * res.score
            if isbn:
                conf = max(conf, 0.9)
        else:
            rec = {"title": title, "author": author, "genre": "Unknown", "subgenre": "", "isbn": isbn,
                   "description": "", "publication_year": "", "language": "", "source": "image_scan"}
            conf = 0.6 * ocr_c
        rec.update({"scan_confidence": round(min(conf, 1.0), 3), "raw_text": parsed["raw_text"][:300],
                    "region": region_index + 1})
        return rec

    def scan(self, img: np.ndarray, mode: str = "auto", crop=None) -> ScanReport:
        rep = ScanReport()
        img = preprocess(crop_fraction(img, *crop) if crop else img)
        regions = detect_book_regions(img, mode)
        rep.regions = regions
        annotated = img.copy()
        for i, (x, y, w, h) in enumerate(regions):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), max(2, img.shape[1] // 400))
            cv2.putText(annotated, str(i + 1), (x + 6, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        rep.annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        for i, (x, y, w, h) in enumerate(regions):
            try:
                lines = self.ocr.read_best(img[y:y + h, x:x + w], tall=h > 1.8 * w)
            except OCRUnavailableError as exc:
                rep.ocr_available = False
                rep.warnings.append(str(exc))
                return rep
            except OCRError as exc:
                rep.warnings.append(f"Region {i + 1}: {exc}")
                continue
            parsed = parse_lines(lines)
            if not parsed["title"] and not parsed["isbn"]:
                rep.unidentified += 1
                continue
            rep.records.append(self.build_record(parsed, i))
        if rep.unidentified:
            rep.warnings.append(f"{rep.unidentified} region(s) had no readable text (blur, glare, tiny or stylised text).")
        if not rep.records and rep.ocr_available:
            rep.warnings.append("No book text could be read. Try a sharper, well-lit, closer photo, a different scan "
                                "mode, or add the book manually.")
        return rep
