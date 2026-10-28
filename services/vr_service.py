"""VR/AR input bridge (optional). A headset client (or the simulator in the UI) POSTs captures to a small
HTTP bridge; captures are stored as JSON files in an inbox that the Streamlit app processes through the
same scanning/catalogue pipeline. No direct headset access is attempted from Python."""
import base64
import json
import time
import uuid
from pathlib import Path

import config
from services.image_scanner import load_image
from services.text_utils import is_blank

MAX_PAYLOAD_BYTES = 12 * 1024 * 1024


class VRPayloadError(ValueError):
    pass


def _inbox(inbox) -> Path:
    p = Path(inbox or config.VR_INBOX_DIR)
    (p / "processed").mkdir(parents=True, exist_ok=True)
    return p


def submit(payload: dict, inbox=None) -> str:
    """Validate and store one capture. Accepts an image (image_b64) and/or already-recognised fields."""
    if not isinstance(payload, dict):
        raise VRPayloadError("Payload must be a JSON object.")
    has_img = not is_blank(payload.get("image_b64"))
    has_txt = any(not is_blank(payload.get(k)) for k in ("title", "isbn"))
    if not (has_img or has_txt):
        raise VRPayloadError("Payload needs 'image_b64' or at least 'title'/'isbn'.")
    if has_img:
        try:
            raw = base64.b64decode(str(payload["image_b64"]).split(",")[-1], validate=False)
        except Exception as exc:
            raise VRPayloadError("image_b64 is not valid base64.") from exc
        if len(raw) > MAX_PAYLOAD_BYTES:
            raise VRPayloadError("Image too large.")
        try:
            load_image(raw)
        except ValueError as exc:
            raise VRPayloadError(str(exc)) from exc
    item = {"id": f"{int(time.time())}_{uuid.uuid4().hex[:6]}", "received_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(payload.get("device", "unknown"))[:60],
            **{k: str(payload.get(k, ""))[:300] for k in ("title", "author", "isbn")},
            "image_b64": payload.get("image_b64", "") if has_img else ""}
    d = _inbox(inbox)
    tmp = d / f"{item['id']}.tmp"
    tmp.write_text(json.dumps(item), encoding="utf-8")
    tmp.rename(d / f"{item['id']}.json")
    return item["id"]


def list_pending(inbox=None) -> list:
    out = []
    for f in sorted(_inbox(inbox).glob("*.json")):
        try:
            it = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({k: it.get(k, "") for k in ("id", "received_at", "device", "title", "author", "isbn")}
                   | {"has_image": bool(it.get("image_b64"))})
    return out


def load_item(item_id: str, inbox=None) -> dict:
    f = _inbox(inbox) / f"{Path(item_id).name}.json"
    if not f.exists():
        raise KeyError(f"No pending VR item {item_id}")
    return json.loads(f.read_text(encoding="utf-8"))


def image_of(item: dict):
    return load_image(base64.b64decode(str(item["image_b64"]).split(",")[-1])) if item.get("image_b64") else None


def mark_processed(item_id: str, inbox=None):
    d = _inbox(inbox)
    f = d / f"{Path(item_id).name}.json"
    if f.exists():
        f.rename(d / "processed" / f.name)
