"""Minimal HTTP command server for driving the fork's ``remote`` policy.

The Isaac Sim fork runs ``--policy_type remote`` (``RemotePolicy``), a thin
HTTP client that POSTs commands and applies whatever the server returns. This
module is the server side used by the local tools that drive a single,
non-RL sim session (action replay, DAgger collection). The autonomous RL
rollout proxy in ``eval_worker.py`` implements the same protocol inline.

Protocol (all JSON; observation images are base64-encoded):

    POST /ping               -> {"status": "ok"}
    POST /next_task          -> episode task | {"shutdown": true}
    POST /reset    {...}     -> {}            (episode is set up; body carries
                                garment_info / augmentation_info / restore_snapshot)
    POST /infer    {obs...}  -> {"actions": [12], "stop"?, "save_state"?, "n_steps"?}
    POST /snapshot {state}   -> {}
    POST /update_check_status-> {}

Subclass :class:`RemoteCommandServer` and implement ``handle(endpoint, body,
session_id)``; it runs on the HTTP worker thread. Every request carries a
``session_id`` so one server can drive several concurrent simulations.

Mirrors the plumbing of the challenge's ``dummy_docker_policy/server.py``.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np


def decode_observation(body: dict) -> dict:
    """Decode a posted observation: base64 images -> arrays, lists -> arrays."""
    obs: dict = {}
    for k, v in body.items():
        if isinstance(v, dict) and "base64" in v:
            buf = base64.b64decode(v["base64"])
            obs[k] = np.frombuffer(buf, dtype=np.dtype(v["dtype"])).reshape(v["shape"])
        elif isinstance(v, list):
            obs[k] = np.array(v)
        else:
            obs[k] = v
    return obs


class RemoteCommandServer:
    """HTTP server that answers the fork's remote-policy commands.

    Subclasses implement ``handle(endpoint, body, session_id) -> dict``. ``/ping``
    is answered automatically. Start with ``.start()`` (binds a port, serves in a
    background thread), read ``.url`` for the address to hand the sim, and call
    ``.stop()`` when done.
    """

    def __init__(self, host: str = "localhost", port: int = 0):
        self.host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- override -----------------------------------------------------------

    def handle(self, endpoint: str, body: dict, session_id: str) -> dict | None:
        raise NotImplementedError

    # -- lifecycle ----------------------------------------------------------

    @property
    def url(self) -> str:
        assert self._server is not None, "server not started"
        return f"http://{self.host}:{self._server.server_address[1]}"

    def start(self) -> "RemoteCommandServer":
        owner = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    body = json.loads(raw)
                except Exception:
                    body = {}
                session_id = body.get("session_id", "default")
                if self.path == "/ping":
                    resp: Any = {"status": "ok"}
                else:
                    try:
                        resp = owner.handle(self.path, body, session_id)
                    except Exception as e:  # surface errors to the sim
                        resp = {"error": str(e)}
                if resp is None:
                    resp = {}
                out = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer((self.host, self._port), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="remote-cmd-server"
        )
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
