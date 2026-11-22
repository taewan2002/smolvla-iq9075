"""Serve inference over localhost so a separate machine can render LIBERO."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

MAX_REQUEST_BYTES = 2 * 1024 * 1024


class RemotePolicy:
    def __init__(self, url):
        self.url = url.rstrip("/")
        with urlopen(self.url + "/manifest", timeout=30) as response:
            self.manifest = json.load(response)
        self.action_shape = tuple(self.manifest["action_shape"])

    def __call__(self, rgb, state, task, noise):
        buffer = BytesIO()
        np.savez_compressed(buffer, rgb=np.stack(rgb), state=state, task=task, noise=noise)
        request = Request(self.url + "/predict", data=buffer.getvalue(),
                          headers={"Content-Type": "application/octet-stream"})
        with urlopen(request, timeout=60) as response:
            return np.load(BytesIO(response.read()), allow_pickle=False)


def serve(policy, port):
    manifest = {
        "action_shape": list(policy.action_shape),
        "checkpoint_sha256": policy.manifest["checkpoint_sha256"],
        "backend": policy.backend,
    }
    if "denoising_steps" in policy.manifest:
        manifest["denoising_steps"] = policy.manifest["denoising_steps"]

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, payload, media):
            self.send_response(200)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/manifest":
                self.send_error(404)
                return
            self.send_bytes(json.dumps(manifest).encode(), "application/json")

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    self.send_error(413)
                    return
                with np.load(BytesIO(self.rfile.read(size)), allow_pickle=False) as batch:
                    result = policy(batch["rgb"], batch["state"], str(batch["task"].item()), batch["noise"])
                payload = BytesIO()
                np.save(payload, result)
            except (KeyError, ValueError, TypeError) as error:
                self.send_error(400, type(error).__name__)
                return
            except (FloatingPointError, RuntimeError) as error:
                self.send_error(500, type(error).__name__)
                return
            self.send_bytes(payload.getvalue(), "application/octet-stream")

        def log_request(self, code="-", size="-"):
            if code != 200:
                super().log_request(code, size)

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        nodes = getattr(policy, "devices", [])
        print(f"{policy.backend} inference ready on 127.0.0.1:{port}; HTP nodes {nodes}", flush=True)
        server.serve_forever()


def main():
    from .runtime import EdgePolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--engine", choices=["cpu", "htp", "dual"], default="dual")
    parser.add_argument("--port", type=int, default=18075)
    args = parser.parse_args()
    policy = EdgePolicy(args.bundle, args.engine)
    try:
        serve(policy, args.port)
    finally:
        policy.close()


if __name__ == "__main__":
    main()
