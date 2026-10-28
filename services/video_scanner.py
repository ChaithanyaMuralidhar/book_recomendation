"""Video / webcam-frame scanning: sample frames, skip near-identical ones, OCR, deduplicate across frames."""
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import config
from services.image_scanner import ImageScanner
from services.text_utils import clean_isbn, is_blank, similarity


class InvalidVideoError(ValueError):
    pass


@dataclass
class VideoReport:
    records: list = field(default_factory=list)
    frames_read: int = 0
    frames_processed: int = 0
    frames_skipped_similar: int = 0
    warnings: list = field(default_factory=list)


def check_video_path(path) -> Path:
    p = Path(path)
    if p.suffix.lower() not in config.SUPPORTED_VIDEO_EXT:
        raise InvalidVideoError(f"Unsupported video format '{p.suffix}'. Use: {sorted(config.SUPPORTED_VIDEO_EXT)}")
    return p


def probe_video(path) -> dict:
    p = check_video_path(path)
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise InvalidVideoError("The video could not be opened (corrupt file or missing codec).")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"fps": fps, "frames": n, "duration": (n / fps) if fps else 0.0}
    finally:
        cap.release()


def get_frame_at(path, t_seconds: float):
    """Single frame at a time offset (fallback when automatic scanning fails)."""
    p = check_video_path(path)
    cap = cv2.VideoCapture(str(p))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_seconds) * 1000)
        ok, frame = cap.read()
        if not ok:
            raise InvalidVideoError("Could not read a frame at that time.")
        return frame
    finally:
        cap.release()


def _thumb(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def is_similar(a, b, threshold: float = 6.0) -> bool:
    """Mean absolute difference of 32x32 grayscale thumbnails (0-255 scale)."""
    return a is not None and float(np.mean(np.abs(a - b))) < threshold


def sample_frames(path, interval_s: float = 1.0, max_frames: int = 40):
    """Yield (time_s, frame) every interval_s seconds (at most max_frames)."""
    p = check_video_path(path)
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise InvalidVideoError("The video could not be opened (corrupt file or missing codec).")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps * max(interval_s, 0.1))))
        idx = n = 0
        while n < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            yield idx / fps, frame
            n += 1
            idx += step
    finally:
        cap.release()


def dedupe_records(records, threshold: float = 0.85) -> list:
    """Merge records describing the same book across frames; keep the most confident, count sightings."""
    groups = []
    for r in records:
        if is_blank(r.get("title")) and is_blank(r.get("isbn")):
            continue
        for g in groups:
            same_isbn = clean_isbn(r.get("isbn")) and clean_isbn(r.get("isbn")) == clean_isbn(g["isbn"])
            if same_isbn or similarity(r.get("title"), g["title"]) >= threshold:
                g["sightings"] += 1
                if (r.get("scan_confidence") or 0) > (g.get("scan_confidence") or 0):
                    sight = g["sightings"]
                    g.clear()
                    g.update({**r, "sightings": sight})
                break
        else:
            groups.append({**r, "sightings": 1})
    return sorted(groups, key=lambda g: -(g.get("scan_confidence") or 0))


def scan_frames(frames, scanner: ImageScanner, mode="auto", diff_threshold=6.0, progress=None) -> VideoReport:
    rep = VideoReport()
    prev, raw = None, []
    frames = list(frames)
    for i, (t, frame) in enumerate(frames):
        rep.frames_read += 1
        th = _thumb(frame)
        if is_similar(prev, th, diff_threshold):
            rep.frames_skipped_similar += 1
        else:
            prev = th
            res = scanner.scan(frame, mode=mode)
            rep.frames_processed += 1
            if not res.ocr_available:
                rep.warnings.extend(res.warnings)
                break
            for r in res.records:
                r["frame_time_s"] = round(t, 2)
                raw.append(r)
        if progress:
            progress((i + 1) / len(frames))
    rep.records = dedupe_records(raw)
    if not rep.records:
        rep.warnings.append("No books were recognised in the sampled frames. Use the frame picker to scan a clear frame.")
    return rep


def scan_video(path, scanner: ImageScanner, interval_s=1.0, max_frames=30, mode="auto", diff_threshold=6.0,
               progress=None) -> VideoReport:
    return scan_frames(sample_frames(path, interval_s, max_frames), scanner, mode, diff_threshold, progress)


def save_upload_to_temp(uploaded_name: str, data: bytes) -> str:
    suffix = Path(uploaded_name).suffix.lower()
    if suffix not in config.SUPPORTED_VIDEO_EXT:
        raise InvalidVideoError(f"Unsupported video format '{suffix}'.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
    return f.name
