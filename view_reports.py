#!/usr/bin/env python3
"""Open the saved reports in a browser instead of the terminal.

    python view_reports.py            # or: ./view_reports.py

Serves ``./reports`` — the tree ``analyze.py`` and the interactive CLI write —
as a local page: a concise tape of every run (call, levels, horizon, the
portfolio manager's summary) and a reader that shows any run's full report with
section navigation, in-report search, and print/markdown export.

The terminal dashboard has to squeeze a 25,000-word report into scrollback;
this does not. Nothing leaves the machine: it is a read-only server bound to
localhost that renders the same markdown files already on disk.

    python view_reports.py --reports-dir ~/other/reports --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cli.report_server import serve

DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_PORT = 8765


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="view_reports.py",
        description="Browse the saved TradingAgents report tree in a local web viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR,
        help=f"Directory holding the run folders (default: ./{DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port to bind; the next free one is used if taken (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1, i.e. this machine only)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Print the URL instead of opening a browser",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log every HTTP request",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # A missing directory is not fatal: the page shows an empty state, and the
    # banner names the path it is watching, so a typo is obvious at a glance.
    serve(
        args.reports_dir.expanduser(),
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        quiet=not args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
