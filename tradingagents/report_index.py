"""Read side of the saved report tree.

``tradingagents.reporting`` writes a finished run as a markdown tree; this
module reads it back. It discovers runs under a reports root, pulls out the
structured decision fields the Trader and Portfolio Manager emit (the
``**Label**: value`` shape produced by ``agents.schemas``), and returns both a
concise per-run digest and the full section texts.

Nothing here touches HTTP or HTML — ``cli.report_server`` layers the viewer on
top, and the digest is equally usable from a notebook or a script.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# A run id is a directory name; keep it to characters a report writer can
# actually produce so it can never escape the reports root (see resolve_run).
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=+-]{0,120}$")

# <TICKER>_<YYYYMMDD>_<HHMMSS>, the layout cli.main's default save path uses.
RUN_DIR_RE = re.compile(r"^(?P<ticker>.+)_(?P<date>\d{8})_(?P<time>\d{6})$")

WORDS_PER_MINUTE = 220


@dataclass(frozen=True)
class GroupSpec:
    """One numbered stage of the pipeline, as written on disk."""

    key: str
    label: str
    numeral: str
    dirname: str
    files: tuple[tuple[str, str], ...]


# Mirrors the tree written by tradingagents.reporting.write_report_tree.
GROUP_SPECS: tuple[GroupSpec, ...] = (
    GroupSpec("analysts", "Analyst Team", "I", "1_analysts", (
        ("market.md", "Market Analyst"),
        ("sentiment.md", "Sentiment Analyst"),
        ("news.md", "News Analyst"),
        ("fundamentals.md", "Fundamentals Analyst"),
    )),
    GroupSpec("research", "Research Team", "II", "2_research", (
        ("bull.md", "Bull Researcher"),
        ("bear.md", "Bear Researcher"),
        ("manager.md", "Research Manager"),
    )),
    GroupSpec("trading", "Trading Team", "III", "3_trading", (
        ("trader.md", "Trader"),
    )),
    GroupSpec("risk", "Risk Management", "IV", "4_risk", (
        ("aggressive.md", "Aggressive Analyst"),
        ("conservative.md", "Conservative Analyst"),
        ("neutral.md", "Neutral Analyst"),
    )),
    GroupSpec("portfolio", "Portfolio Manager", "V", "5_portfolio", (
        ("decision.md", "Portfolio Manager"),
    )),
)

GROUPS_BY_DIR = {g.dirname: g for g in GROUP_SPECS}

_BULLISH = {"buy", "overweight", "long", "accumulate"}
_BEARISH = {"sell", "underweight", "short", "reduce", "trim"}


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

# Matches both shapes the agents render: ``**Label**: value`` (trader, PM) and
# ``**Label:** value`` (sentiment header).
_LABEL_RE = re.compile(r"^\*\*(?P<label>[^*\n]{1,80}?)\*\*[ \t]*:?[ \t]*", re.MULTILINE)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_PROPOSAL_RE = re.compile(r"FINAL TRANSACTION PROPOSAL:\s*\**\s*([A-Za-z]+)")


def parse_labeled_fields(text: str) -> dict[str, str]:
    """Split ``**Label**: value`` blocks into a ``{snake_label: value}`` dict.

    A value runs until the next label at the start of a line, so multi-paragraph
    fields (the PM's investment thesis) survive intact. First occurrence wins.
    """
    matches = list(_LABEL_RE.finditer(text))
    fields: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        label = match.group("label").strip().rstrip(":").strip().lower()
        key = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
        if key:
            fields.setdefault(key, text[match.end():end].strip())
    return fields


def _plain(value: str | None, limit: int | None = None) -> str | None:
    """First paragraph of a value, stripped of inline emphasis, on one line.

    Values run to the next ``**Label**:`` marker, so a short field followed by
    unlabeled prose (the sentiment header's ``**Confidence:**`` sitting above
    the narrative) would otherwise swallow the whole report. Every field read
    into a digest is a single paragraph by schema, so cutting at the first blank
    line is both safe and the right summary.
    """
    if not value:
        return None
    cleaned = re.sub(r"[*`_]{1,3}", "", value.split("\n\n", 1)[0])
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if limit and len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _number(value: str | None) -> float | None:
    """First number in a value, tolerating currency symbols and thousands commas."""
    if not value:
        return None
    match = _NUMBER_RE.search(value)
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def _direction(action: str | None, rating: str | None) -> str:
    """Bull / bear / flat, for colouring. The trader's action outranks the rating."""
    for token in (action, rating):
        key = (token or "").strip().lower()
        if key in _BULLISH:
            return "bull"
        if key in _BEARISH:
            return "bear"
        if key:
            return "flat"
    return "none"


def _pct(from_price: float | None, to_price: float | None) -> float | None:
    if not from_price or to_price is None:
        return None
    return round((to_price - from_price) / from_price * 100, 2)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SectionMeta:
    """One markdown file inside a run, with its place in the pipeline."""

    id: str
    title: str
    group: str
    group_label: str
    numeral: str
    file: str
    words: int


@dataclass
class RunSummary:
    """The concise digest of a run: who decided what, at which levels."""

    run_id: str
    path: str
    ticker: str
    run_date: str | None = None
    run_time: str | None = None
    generated_at: str | None = None
    sort_key: str = ""
    action: str | None = None
    rating: str | None = None
    direction: str = "none"
    entry_price: float | None = None
    stop_loss: float | None = None
    price_target: float | None = None
    position_sizing: str | None = None
    time_horizon: str | None = None
    executive_summary: str | None = None
    reasoning: str | None = None
    sentiment: str | None = None
    sentiment_score: float | None = None
    sentiment_confidence: str | None = None
    target_pct: float | None = None
    stop_pct: float | None = None
    words: int = 0
    minutes: int = 0
    has_complete_report: bool = False
    sections: list[SectionMeta] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunDetail:
    """A run's digest plus every section's raw markdown."""

    summary: RunSummary
    sections: list[tuple[SectionMeta, str]]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def looks_like_run(path: Path) -> bool:
    """True when a directory holds a report tree (rather than a bag of runs)."""
    if not path.is_dir():
        return False
    if (path / "complete_report.md").is_file():
        return True
    return any((path / spec.dirname).is_dir() for spec in GROUP_SPECS)


def discover_run_dirs(root: Path | str) -> list[Path]:
    """Run directories under ``root``, newest first.

    A ``--save-path`` pointed straight at a single run is handled too: if no
    child looks like a run but ``root`` itself does, ``root`` is the only run.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    children = [p for p in root.iterdir() if looks_like_run(p)]
    if not children and looks_like_run(root):
        return [root]
    return sorted(children, key=lambda p: (_dir_sort_key(p), p.name), reverse=True)


def _dir_sort_key(path: Path) -> str:
    match = RUN_DIR_RE.match(path.name)
    if match:
        return f"{match.group('date')}{match.group('time')}"
    # Undated directory (custom --save-path): fall back to its mtime so it still
    # lands in chronological order next to the timestamped ones.
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d%H%M%S")


def resolve_run_dir(root: Path | str, run_id: str) -> Path:
    """Map a run id back to its directory, refusing anything outside ``root``."""
    root = Path(root).expanduser()
    if not RUN_ID_RE.match(run_id or ""):
        raise KeyError(run_id)
    if run_id == root.name and looks_like_run(root):
        return root
    candidate = root / run_id
    resolved = candidate.resolve()
    if root.resolve() not in resolved.parents or not looks_like_run(resolved):
        raise KeyError(run_id)
    return resolved


def iter_section_files(run_dir: Path):
    """Yield ``(GroupSpec, path, title)`` for every section file, in pipeline order.

    Unrecognised ``*.md`` files inside a known group directory are included with
    a title derived from the filename, so a newly added agent shows up in the
    viewer before this table learns its name.
    """
    for spec in GROUP_SPECS:
        group_dir = run_dir / spec.dirname
        if not group_dir.is_dir():
            continue
        titles = dict(spec.files)
        ordered = [name for name, _ in spec.files if (group_dir / name).is_file()]
        extras = sorted(p.name for p in group_dir.glob("*.md") if p.name not in titles)
        for name in ordered + extras:
            title = titles.get(name) or name.removesuffix(".md").replace("_", " ").replace("-", " ").title()
            yield spec, group_dir / name, title


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _generated_at(run_dir: Path) -> str | None:
    """The ``Generated:`` stamp from the head of complete_report.md, if present."""
    complete = run_dir / "complete_report.md"
    if not complete.is_file():
        return None
    with complete.open(encoding="utf-8", errors="replace") as handle:
        for _ in range(10):
            line = handle.readline()
            if not line:
                break
            if line.startswith("Generated:"):
                return line.split(":", 1)[1].strip()
    return None


def _ticker_from(run_dir: Path, match: re.Match | None) -> str:
    if match:
        return match.group("ticker")
    complete = run_dir / "complete_report.md"
    if complete.is_file():
        first = _read(complete).partition("\n")[0]
        if ":" in first:
            return first.split(":", 1)[1].strip() or run_dir.name
    return run_dir.name


def summarize_run(run_dir: Path, texts: dict[str, str] | None = None) -> RunSummary:
    """Build the concise digest for one run directory.

    ``texts`` lets a caller that already read the files (``load_run``) avoid a
    second pass; otherwise every section is read once to count words.
    """
    match = RUN_DIR_RE.match(run_dir.name)
    summary = RunSummary(
        run_id=run_dir.name,
        path=str(run_dir),
        ticker=_ticker_from(run_dir, match),
        generated_at=_generated_at(run_dir),
        has_complete_report=(run_dir / "complete_report.md").is_file(),
        sort_key=_dir_sort_key(run_dir),
    )
    if match:
        date, time = match.group("date"), match.group("time")
        summary.run_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        summary.run_time = f"{time[:2]}:{time[2:4]}"

    by_file: dict[str, str] = {}
    for spec, path, title in iter_section_files(run_dir):
        text = texts[path.name] if texts and path.name in texts else _read(path)
        by_file[f"{spec.dirname}/{path.name}"] = text
        words = len(text.split())
        summary.words += words
        summary.sections.append(SectionMeta(
            id=f"{spec.key}-{path.name.removesuffix('.md')}",
            title=title,
            group=spec.key,
            group_label=spec.label,
            numeral=spec.numeral,
            file=f"{spec.dirname}/{path.name}",
            words=words,
        ))
    summary.minutes = max(1, round(summary.words / WORDS_PER_MINUTE)) if summary.words else 0

    _apply_trader(summary, by_file.get("3_trading/trader.md", ""))
    _apply_portfolio(summary, by_file.get("5_portfolio/decision.md", ""))
    _apply_sentiment(summary, by_file.get("1_analysts/sentiment.md", ""))

    summary.direction = _direction(summary.action, summary.rating)
    # Signed distance from entry, not a reward/risk ratio: a run can pair a
    # short-side target with a stop on a retained long core (see the trader's
    # position_sizing), and dividing the two would invent a ratio the report
    # never claimed.
    summary.target_pct = _pct(summary.entry_price, summary.price_target)
    summary.stop_pct = _pct(summary.entry_price, summary.stop_loss)
    return summary


def _apply_trader(summary: RunSummary, text: str) -> None:
    if not text:
        return
    fields = parse_labeled_fields(text)
    summary.action = _plain(fields.get("action"))
    if not summary.action:
        proposal = _PROPOSAL_RE.search(text)
        if proposal:
            summary.action = proposal.group(1).capitalize()
    summary.reasoning = _plain(fields.get("reasoning"))
    summary.entry_price = _number(fields.get("entry_price"))
    summary.stop_loss = _number(fields.get("stop_loss"))
    summary.position_sizing = _plain(fields.get("position_sizing"))


def _apply_portfolio(summary: RunSummary, text: str) -> None:
    if not text:
        return
    fields = parse_labeled_fields(text)
    summary.rating = _plain(fields.get("rating"))
    summary.executive_summary = _plain(fields.get("executive_summary"))
    summary.price_target = _number(fields.get("price_target"))
    summary.time_horizon = _plain(fields.get("time_horizon"))


def _apply_sentiment(summary: RunSummary, text: str) -> None:
    if not text:
        return
    fields = parse_labeled_fields(text)
    raw = fields.get("overall_sentiment") or ""
    summary.sentiment = _plain(raw.split("(")[0])
    score = re.search(r"([-+]?\d+(?:\.\d+)?)\s*/\s*10", raw)
    if score:
        summary.sentiment_score = float(score.group(1))
    summary.sentiment_confidence = _plain(fields.get("confidence"))


def list_runs(root: Path | str) -> list[RunSummary]:
    """Digests for every run under ``root``, newest first."""
    return [summarize_run(path) for path in discover_run_dirs(root)]


def load_run(root: Path | str, run_id: str) -> RunDetail:
    """Full detail — digest plus each section's markdown — for one run."""
    run_dir = resolve_run_dir(root, run_id)
    sections: list[tuple[SectionMeta, str]] = []
    texts: dict[str, str] = {}
    for spec, path, _title in iter_section_files(run_dir):
        texts[path.name] = _read(path)
    summary = summarize_run(run_dir, texts)
    for meta in summary.sections:
        sections.append((meta, texts.get(Path(meta.file).name, "")))
    return RunDetail(summary=summary, sections=sections)


def complete_report_markdown(root: Path | str, run_id: str) -> str:
    """The run's consolidated markdown, rebuilt from sections if it is missing."""
    run_dir = resolve_run_dir(root, run_id)
    complete = run_dir / "complete_report.md"
    if complete.is_file():
        return _read(complete)
    detail = load_run(root, run_id)
    parts = [f"# Trading Analysis Report: {detail.summary.ticker}", ""]
    current = None
    for meta, text in detail.sections:
        if meta.group != current:
            current = meta.group
            parts.append(f"## {meta.numeral}. {meta.group_label}\n")
        parts.append(f"### {meta.title}\n{text}\n")
    return "\n".join(parts)
