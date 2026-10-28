"""Tiny stdlib HTTP bridge for VR/AR clients.  Run:  python -m vr.bridge_server --port 8765
POST /api/vr/submit  (JSON: {device, image_b64?, title?, author?, isbn?})   GET /health   GET /client
Optional shared secret: set VR_BRIDGE_TOKEN and send it as header X-Bridge-Token."""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from services import vr_service


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Bridge-Token")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path in ("/", "/client"):
            return self._send(200, (Path(__file__).parent / "webxr_client.html").read_bytes(), "text/html; charset=utf-8")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/vr/submit":
            return self._send(404, {"error": "not found"})
        token = os.getenv("VR_BRIDGE_TOKEN")
        if token and self.headers.get("X-Bridge-Token") != token:
            return self._send(401, {"error": "bad token"})
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > vr_service.MAX_PAYLOAD_BYTES * 1.4:
            return self._send(413, {"error": "payload missing or too large"})
        try:
            item_id = vr_service.submit(json.loads(self.rfile.read(n)))
        except (ValueError, vr_service.VRPayloadError) as exc:
            return self._send(400, {"error": str(exc)})
        self._send(201, {"id": item_id})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"VR bridge listening on http://{a.host}:{a.port}  (inbox: {vr_service._inbox(None)})")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
