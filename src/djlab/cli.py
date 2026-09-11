"""Entry point: ``djlab`` launches the local testing UI.

Binds to 127.0.0.1 only -- this is a local development tool for one person
testing the engine on their own machine, never a hosted/multi-user service.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from djlab.server import LabState, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="djlab", description="Local testing UI for the AI DJ engine")
    parser.add_argument("--port", type=int, default=8787, help="Local port to serve on (default: 8787)")
    parser.add_argument("--library", type=Path, default=None, help="Music folder to load on startup")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open a browser tab")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    state = LabState()
    if args.library is not None:
        root = args.library.expanduser()
        if not root.is_dir():
            parser.error(f"--library is not a directory: {root}")
        state.set_library(root)

    app = create_app(state)
    url = f"http://127.0.0.1:{args.port}"

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"AI DJ Lab running at {url} (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
