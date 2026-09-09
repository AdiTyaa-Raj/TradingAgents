"""Local web viewer for the saved report tree.

The terminal dashboard has to fit a run's reports into a scrollback buffer; a
25,000-word report does not survive that. This serves the same report tree
(``./reports/<TICKER>_<timestamp>/``) as a browsable single page: a concise
tape of every run's decision, and a reader that renders each run's markdown in
full with section navigation and search.

It is a read-only ``http.server`` bound to localhost, with no dependency
outside the standard library and ``markdown_it`` (already required by rich), so
``python view_reports.py`` works offline with nothing to install.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import webbrowser
from dataclasses import asdict
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from markdown_it import MarkdownIt

from tradingagents.report_index import (
    complete_report_markdown,
    list_runs,
    load_run,
)

STATIC_DIR = Path(__file__).parent / "static"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
}

# Reports are LLM-authored text: render markdown, never raw HTML embedded in it.
# Tables and strikethrough are off in the commonmark preset but heavily used by
# the analysts, so enable them explicitly.
_md = MarkdownIt("commonmark", {"html": False, "typographer": True})
_md.enable(["table", "strikethrough"])

_SLUG_STRIP = re.compile(r"[^a-z0-9\s-]")


def _slug(text: str) -> str:
    base = _SLUG_STRIP.sub("", text.lower()).strip()
    return re.sub(r"[\s-]+", "-", base) or "section"


def _inline_text(token) -> str:
    """Plain text of an inline token, for heading titles and anchor slugs."""
    if not token or not token.children:
        return (token.content if token else "") or ""
    return "".join(
        child.content for child in token.children if child.type in ("text", "code_inline")
    )


def render_markdown(text: str, prefix: str = "") -> tuple[str, list[dict]]:
    """Render markdown to HTML, returning it with a table of contents.

    Every heading gets a stable ``id`` (namespaced by ``prefix`` so several
    sections can share one page without colliding), and h2/h3 headings are
    collected for the reader's outline.
    """
    tokens = _md.parse(text or "")
    toc: list[dict] = []
    seen: dict[str, int] = {}
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        title = _inline_text(tokens[index + 1] if index + 1 < len(tokens) else None).strip()
        level = int(token.tag[1:])
        slug = f"{prefix}-{_slug(title)}" if prefix else _slug(title)
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"
        token.attrSet("id", slug)
        if 2 <= level <= 3 and title:
            toc.append({"id": slug, "title": title, "level": level})
    html = _md.renderer.render(tokens, _md.options, {})
    # Reports cite sources; keep the viewer put when one is opened.
    html = html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    return html, toc


def build_run_payload(root: Path, run_id: str) -> dict:
    """A run's digest plus every section rendered to HTML, for one fetch."""
    detail = load_run(root, run_id)
    sections = []
    for meta, text in detail.sections:
        html, toc = render_markdown(text, prefix=meta.id)
        sections.append({**asdict(meta), "html": html, "toc": toc})
    return {"summary": detail.summary.as_dict(), "sections": sections}


class ReportViewerHandler(BaseHTTPRequestHandler):
    """Read-only routes: the page, its assets, and the report JSON API."""

    server_version = "TradingAgentsReportViewer"
    protocol_version = "HTTP/1.1"
    # Keep-alive is worth having for the asset fetches, but an idle browser
    # connection must not hold its thread for the life of the process.
    timeout = 30

    def __init__(self, *args, root: Path, quiet: bool = True, **kwargs):
        self.root = root
        self.quiet = quiet
        super().__init__(*args, **kwargs)

    # BaseHTTPRequestHandler logs every hit to stderr; the page pulls assets and
    # JSON on each navigation, which would bury the startup banner.
    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)

    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                return self._send_static("viewer.html")
            if path.startswith("/static/"):
                return self._send_static(path[len("/static/"):])
            if path == "/api/runs":
                runs = [run.as_dict() for run in list_runs(self.root)]
                return self._send_json({"root": str(self.root.resolve()), "runs": runs})
            match = re.fullmatch(r"/api/runs/([^/]+)", path)
            if match:
                return self._send_json(build_run_payload(self.root, match.group(1)))
            match = re.fullmatch(r"/api/runs/([^/]+)/markdown", path)
            if match:
                run_id = match.group(1)
                body = complete_report_markdown(self.root, run_id).encode("utf-8")
                return self._send(
                    200, body, "text/markdown; charset=utf-8",
                    extra={"Content-Disposition": f'attachment; filename="{run_id}.md"'},
                )
            return self._send_json({"error": "not found"}, status=404)
        except KeyError:
            return self._send_json({"error": "unknown run"}, status=404)
        except FileNotFoundError:
            return self._send_json({"error": "not found"}, status=404)
        except BrokenPipeError:
            return None  # browser navigated away mid-response
        except Exception as exc:  # a bad report file must not kill the server
            return self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _send_static(self, name: str):
        target = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in target.parents or not target.is_file():
            return self._send_json({"error": "not found"}, status=404)
        ctype = CONTENT_TYPES.get(target.suffix)
        if ctype is None:
            return self._send_json({"error": "not found"}, status=404)
        return self._send(200, target.read_bytes(), ctype)

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Always re-read from disk: a run that finishes while the page is open
        # should appear on the next refresh.
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)


def make_server(root: Path | str, host: str = "127.0.0.1", port: int = 8765,
                quiet: bool = True) -> ThreadingHTTPServer:
    """Bind a viewer server, walking forward from ``port`` if it is taken."""
    root = Path(root).expanduser()
    handler = partial(ReportViewerHandler, root=root, quiet=quiet)
    last_error: OSError | None = None
    for candidate in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            last_error = exc
    raise SystemExit(f"No free port in {port}-{port + 19}: {last_error}")


def serve(root: Path | str, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True, quiet: bool = True) -> None:
    """Run the viewer until interrupted."""
    root = Path(root).expanduser()
    httpd = make_server(root, host=host, port=port, quiet=quiet)
    bound_host, bound_port = httpd.server_address[0], httpd.server_address[1]
    if isinstance(bound_host, bytes):  # pragma: no cover - platform quirk
        bound_host = bound_host.decode()
    url = f"http://{bound_host}:{bound_port}/"

    runs = list_runs(root)
    newest = runs[0] if runs else None
    banner = [
        "",
        f"  Report viewer : {url}",
        f"  Reports root  : {root.resolve()}",
        f"  Runs found    : {len(runs)}  (newest: {newest.ticker} {newest.run_date or ''})"
        if newest
        else "  Runs found    : 0 — run `python analyze.py NVDA` first",
        "",
        "  Ctrl-C to stop.",
        "",
    ]
    # flush=True: the URL is the whole point of the banner, and stdout is block
    # buffered whenever this is piped or redirected.
    print("\n".join(banner), flush=True)

    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.shutdown()
        httpd.server_close()


def find_free_port(host: str = "127.0.0.1") -> int:  # pragma: no cover - test helper
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]
