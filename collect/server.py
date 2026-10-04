"""Serves the recorder page and writes posted sessions to data/<source>/.

    python collect/server.py                      # then open http://127.0.0.1:8765
    python collect/server.py --tasks drag,scroll  # a session of only drags and scrolls
"""
import argparse
import json
import re
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
MAX_BODY = 50 * 1024 * 1024


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_BODY:
            self.send_error(413)
            return
        try:
            session = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return

        source = re.sub(r"[^a-z0-9_-]", "", str(session.get("source", "human")).lower()) or "human"
        out_dir = DATA / ("raw" if source == "human" else source)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{source}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}.json"
        (out_dir / name).write_text(json.dumps(session), encoding="utf-8")

        body = json.dumps({"saved": name, "trials": len(session.get("trials", []))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if self.command == "POST":
            super().log_message(fmt, *args)


def make_server(port=8765):
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT)))


def serve_in_background(port=8765):
    server = make_server(port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tasks", help="comma-separated subset of click,drag,scroll for the link it prints")
    args = parser.parse_args()
    server = make_server(args.port)
    url = f"http://127.0.0.1:{args.port}/" + (f"?tasks={args.tasks}" if args.tasks else "")
    print(f"Recorder at {url}  (Ctrl+C to stop; press S in the page to save)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
