"""Minimal Flask webserver for the NIDS test target.

Serves static HTML/JS on a couple of routes, plus lightweight API
endpoints so there is realistic baseline traffic to blend attacks with.
Listens on port 8080 inside the container.
"""

from __future__ import annotations

import hashlib
import random
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
import signal

RUNNING = True


def _handle_signal(signum, frame):
    global RUNNING
    RUNNING = False
    print("[target] Received shutdown signal, exiting.", flush=True)


class TargetHandler(BaseHTTPRequestHandler):
    """Simple handler for normal and API traffic."""

    def log_message(self, format, *args):
        pass  # Silence default stderr logging

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        routes = {
            "/": self._index,
            "/index.html": self._index,
            "/app.js": self._app_js,
            "/api/status": self._api_status,
            "/api/users": self._api_users,
            "/api/data": self._api_data,
        }
        handler = routes.get(self.path, self._not_found)
        handler()

    def do_POST(self) -> None:
        if self.path == "/api/login":
            self._api_login()
        else:
            self._not_found()

    def _index(self) -> None:
        body = b"<html><head><title>Netwatron Test Target</title></head>"
        body += b"<body><h1>NIDS Test Server</h1>"
        body += b'<script src="/app.js"></script></body></html>'
        self._send(200, "text/html", body)

    def _app_js(self) -> None:
        body = b"console.log('netwatron target loaded');"
        self._send(200, "application/javascript", body)

    def _api_status(self) -> None:
        data = json.dumps(
            {
                "status": "ok",
                "uptime": time.time(),
                "hostname": "target",
            }
        )
        self._send(200, "application/json", data.encode())

    def _api_users(self) -> None:
        users = [{"id": i, "name": f"user_{i}"} for i in range(1, 11)]
        data = json.dumps({"users": users})
        self._send(200, "application/json", data.encode())

    def _api_data(self) -> None:
        payload = {
            "values": [random.random() for _ in range(20)],
            "checksum": hashlib.md5(str(time.time()).encode()).hexdigest(),
        }
        data = json.dumps(payload)
        self._send(200, "application/json", data.encode())

    def _api_login(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        _body = self.rfile.read(content_length) if content_length else b""
        success = random.random() > 0.7
        data = json.dumps({"success": success})
        code = 200 if success else 401
        self._send(code, "application/json", data.encode())

    def _not_found(self) -> None:
        self._send(404, "text/plain", b"404 Not Found")


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), TargetHandler)
    print(f"[target] Listening on port {port}", flush=True)
    while RUNNING:
        server.handle_request()
    server.server_close()
    print("[target] Shut down cleanly.", flush=True)


if __name__ == "__main__":
    main()
