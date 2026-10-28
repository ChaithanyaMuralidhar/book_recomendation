"""OCR wrapper around Tesseract (via pytesseract) with orientation search and clear failure modes."""
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # handled at call time
    pytesseract = None


class OCRError(RuntimeError):
    pass


class OCRUnavailableError(OCRError):
    pass


@dataclass
class OCRLine:
    text: str
    conf: float     # 0..1
    height: float   # pixel height of the tallest word (proxy for font size)
    top: float
    left: float


def _configure():
    if pytesseract is None:
        raise OCRUnavailableError("The 'pytesseract' package is not installed (pip install pytesseract).")
    cmd = config.TESSERACT_CMD or shutil.which("tesseract")
    if not cmd:
        for p in config.WINDOWS_TESSERACT_PATHS:
            if Path(p).exists():
                cmd = p
                break
    if not cmd:
        raise OCRUnavailableError(
            "Tesseract is not installed or not on PATH. Install it (Windows: UB-Mannheim installer) "
            "or set the TESSERACT_CMD environment variable.")
    pytesseract.pytesseract.tesseract_cmd = cmd


def is_available() -> bool:
    try:
        _configure()
        return True
    except OCRUnavailableError:
        return False


def _prepare(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    if min(h, w) < 400:  # upscale small crops: Tesseract needs ~20+ px text height
        f = min(400 / min(h, w), 4)
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.GaussianBlur(gray, (3, 3), 0) if gray.size > 1_500_000 else cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)


def read_lines(gray: np.ndarray, psm: int = 11) -> list:
    _configure()
    try:
        d = pytesseract.image_to_data(gray, config=f"--psm {psm}", output_type=Output.DICT)
    except Exception as exc:  # pytesseract.TesseractError / OSError
        raise OCRError(f"Tesseract failed: {exc}") from exc
    groups = {}
    for i, txt in enumerate(d["text"]):
        txt = (txt or "").strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if not txt or conf < 0:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        groups.setdefault(key, []).append((d["left"][i], txt, conf / 100.0, d["height"][i], d["top"][i]))
    lines = []
    for words in groups.values():
        words.sort()
        text = " ".join(w[1] for w in words)
        weights = [max(len(w[1]), 1) for w in words]
        conf = sum(w[2] * wt for w, wt in zip(words, weights)) / sum(weights)
        lines.append(OCRLine(text, conf, float(max(w[3] for w in words)), float(min(w[4] for w in words)),
                             float(words[0][0])))
    return sorted(lines, key=lambda l: (l.top, l.left))


def _score(lines) -> float:
    return sum(l.conf * len(l.text) for l in lines if l.conf >= 0.3)


def read_best(img: np.ndarray, tall: bool = False) -> list:
    """OCR an image region, trying rotations for tall (spine-like) regions and inversion if the result is weak."""
    if img is None or img.size == 0:
        raise OCRError("Empty image region.")
    gray = _prepare(img)
    rots = [None] + ([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE] if tall else [])
    best, best_s = [], -1.0
    for inverted in (False, True):
        for rot in rots:
            g = cv2.bitwise_not(gray) if inverted else gray
            if rot is not None:
                g = cv2.rotate(g, rot)
            lines = read_lines(g)
            s = _score(lines)
            if s > best_s:
                best, best_s = lines, s
        if best_s >= 40:  # good enough: skip the inverted pass
            break
    return best
