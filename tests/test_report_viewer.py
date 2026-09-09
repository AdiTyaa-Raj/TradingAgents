"""Tests for the report-tree reader and the local viewer server.

Covers what the viewer depends on: discovering runs under a reports root,
extracting the Trader / Portfolio Manager / Sentiment fields out of the saved
markdown, and the HTTP surface (including that a run id cannot walk out of the
reports root).
"""

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

from cli.report_server import STATIC_DIR, make_server, render_markdown
from tradingagents.report_index import (
    complete_report_markdown,
    discover_run_dirs,
    list_runs,
    load_run,
    parse_labeled_fields,
    resolve_run_dir,
    summarize_run,
)

TRADER = """**Action**: Sell

**Reasoning**: The breakout is unconfirmed and cash conversion is deteriorating.

**Entry Price**: 397.95

**Stop Loss**: 365.5

**Position Sizing**: Trim 40-50% of the existing holding into strength.

FINAL TRANSACTION PROPOSAL: **SELL**"""

DECISION = """**Rating**: Underweight

**Executive Summary**: Trim into strength and keep a core below benchmark weight.

**Investment Thesis**: A good business at a bad entry.

Cash conversion is the deciding evidence.

**Price Target**: 375.0

**Time Horizon**: 3-6 months"""

# The narrative runs on after **Confidence:** with no further label, which is
# exactly the case that must not swallow the whole report into one field.
SENTIMENT = """**Overall Sentiment:** **Neutral** (Score: 5.0/10)
**Confidence:** Low

## Cross-Source Analysis

Two of three sources came back empty, so there is no directional evidence."""


def write_run(root: Path, name: str, *, trader=TRADER, decision=DECISION,
              sentiment=SENTIMENT, extra=None, complete=True) -> Path:
    run = root / name
    for group, files in (
        ("1_analysts", {"market.md": "# Market\n\n## Trend\nStacked bullish.", "sentiment.md": sentiment}),
        ("2_research", {"bull.md": "Bull case.", "bear.md": "Bear case."}),
        ("3_trading", {"trader.md": trader}),
        ("4_risk", {"neutral.md": "Neutral view."}),
        ("5_portfolio", {"decision.md": decision}),
    ):
        (run / group).mkdir(parents=True, exist_ok=True)
        for filename, body in files.items():
            (run / group / filename).write_text(body, encoding="utf-8")
    if extra:
        for relative, body in extra.items():
            (run / relative).write_text(body, encoding="utf-8")
    if complete:
        (run / "complete_report.md").write_text(
            f"# Trading Analysis Report: {name.split('_')[0]}\n\nGenerated: 2026-09-09 11:07:35\n",
            encoding="utf-8",
        )
    return run


@pytest.fixture()
def reports(tmp_path):
    root = tmp_path / "reports"
    write_run(root, "SMSPHARMA.NS_20260909_100652")
    write_run(root, "NVDA_20260908_143012",
              trader=TRADER.replace("Sell", "Buy").replace("SELL", "BUY"),
              decision=DECISION.replace("Underweight", "Overweight"))
    return root


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_labeled_fields_keep_multi_paragraph_values():
    fields = parse_labeled_fields(DECISION)
    assert fields["rating"] == "Underweight"
    assert fields["price_target"] == "375.0"
    assert "deciding evidence" in fields["investment_thesis"]


@pytest.mark.unit
def test_summary_reads_trader_and_portfolio_fields(reports):
    summary = summarize_run(reports / "SMSPHARMA.NS_20260909_100652")
    assert (summary.ticker, summary.run_date, summary.run_time) == ("SMSPHARMA.NS", "2026-09-09", "10:06")
    assert (summary.action, summary.rating, summary.direction) == ("Sell", "Underweight", "bear")
    assert (summary.entry_price, summary.stop_loss, summary.price_target) == (397.95, 365.5, 375.0)
    assert summary.time_horizon == "3-6 months"
    assert summary.executive_summary.startswith("Trim into strength")
    assert summary.generated_at == "2026-09-09 11:07:35"


@pytest.mark.unit
def test_short_fields_stop_at_the_next_paragraph(reports):
    """A one-line field followed by unlabeled prose must not absorb it."""
    summary = summarize_run(reports / "SMSPHARMA.NS_20260909_100652")
    assert summary.sentiment == "Neutral"
    assert summary.sentiment_score == 5.0
    assert summary.sentiment_confidence == "Low"
    assert summary.position_sizing.endswith("into strength.")


@pytest.mark.unit
def test_direction_follows_the_traders_action(reports):
    assert summarize_run(reports / "NVDA_20260908_143012").direction == "bull"


@pytest.mark.unit
@pytest.mark.parametrize("action,rating,expected", [
    ("Buy", "Overweight", "bull"),
    ("Sell", "Underweight", "bear"),
    ("Hold", "Hold", "flat"),
])
def test_direction_mapping(tmp_path, action, rating, expected):
    root = tmp_path / "reports"
    run = write_run(root, "X_20260101_000000",
                    trader=f"**Action**: {action}\n\n**Reasoning**: r.",
                    decision=f"**Rating**: {rating}\n\n**Executive Summary**: s.")
    assert summarize_run(run).direction == expected


@pytest.mark.unit
def test_level_percentages_are_signed_distances_from_entry(reports):
    summary = summarize_run(reports / "SMSPHARMA.NS_20260909_100652")
    assert summary.target_pct == pytest.approx(-5.77, abs=0.01)   # 375.0 vs 397.95
    assert summary.stop_pct == pytest.approx(-8.15, abs=0.01)     # 365.5 vs 397.95


@pytest.mark.unit
def test_missing_levels_stay_none(tmp_path):
    run = write_run(tmp_path / "reports", "HOLD_20260101_000000",
                    trader="**Action**: Hold\n\n**Reasoning**: Nothing to do.",
                    decision="**Rating**: Hold\n\n**Executive Summary**: Sit still.")
    summary = summarize_run(run)
    assert (summary.entry_price, summary.stop_loss, summary.price_target) == (None, None, None)
    assert (summary.target_pct, summary.stop_pct) == (None, None)


@pytest.mark.unit
def test_prices_survive_currency_symbols_and_commas(tmp_path):
    run = write_run(tmp_path / "reports", "BRK_20260101_000000",
                    trader="**Action**: Buy\n\n**Reasoning**: r.\n\n**Entry Price**: $1,234.50")
    assert summarize_run(run).entry_price == 1234.50


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_runs_are_listed_newest_first(reports):
    assert [run.ticker for run in list_runs(reports)] == ["SMSPHARMA.NS", "NVDA"]


@pytest.mark.unit
def test_sections_follow_pipeline_order_and_count_words(reports):
    detail = load_run(reports, "SMSPHARMA.NS_20260909_100652")
    assert [meta.id for meta, _ in detail.sections] == [
        "analysts-market", "analysts-sentiment",
        "research-bull", "research-bear",
        "trading-trader", "risk-neutral", "portfolio-decision",
    ]
    assert detail.summary.words == sum(meta.words for meta in detail.summary.sections)
    assert detail.summary.minutes >= 1


@pytest.mark.unit
def test_unknown_section_file_is_still_offered(reports):
    write_run(reports, "EXTRA_20260101_000000", extra={"1_analysts/macro_desk.md": "Macro view."})
    detail = load_run(reports, "EXTRA_20260101_000000")
    titles = {meta.title for meta, _ in detail.sections}
    assert "Macro Desk" in titles


@pytest.mark.unit
def test_a_root_pointed_straight_at_one_run_still_works(tmp_path):
    run = write_run(tmp_path, "NVDA_20260908_143012")
    assert discover_run_dirs(run) == [run]
    assert list_runs(run)[0].ticker == "NVDA"
    assert resolve_run_dir(run, run.name) == run


@pytest.mark.unit
def test_missing_root_lists_nothing(tmp_path):
    assert list_runs(tmp_path / "nope") == []


@pytest.mark.unit
@pytest.mark.parametrize("run_id", ["../../etc", "..", "/etc/passwd", "", "a/b", "nope"])
def test_run_ids_cannot_escape_the_reports_root(reports, run_id):
    with pytest.raises(KeyError):
        resolve_run_dir(reports, run_id)


@pytest.mark.unit
def test_complete_report_is_rebuilt_when_absent(tmp_path):
    root = tmp_path / "reports"
    write_run(root, "NVDA_20260908_143012", complete=False)
    text = complete_report_markdown(root, "NVDA_20260908_143012")
    assert text.startswith("# Trading Analysis Report: NVDA")
    assert "### Trader" in text and "### Portfolio Manager" in text


# ---------------------------------------------------------------------------
# Rendering and HTTP
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_markdown_renders_tables_and_anchors_headings():
    html, toc = render_markdown("## Trend\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Trend\n", prefix="analysts-market")
    assert "<table>" in html
    assert 'id="analysts-market-trend"' in html
    assert [item["id"] for item in toc] == ["analysts-market-trend", "analysts-market-trend-2"]


@pytest.mark.unit
def test_embedded_html_in_a_report_is_not_executed():
    html, _ = render_markdown("Tool said <script>alert(1)</script> and failed.")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


@pytest.fixture()
def viewer(reports):
    server = make_server(reports, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address
    server.shutdown()
    server.server_close()


def get(address, path):
    connection = HTTPConnection(*address, timeout=10)
    connection.request("GET", path)
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, response.getheader("Content-Type"), body


@pytest.mark.unit
def test_page_and_assets_are_served(viewer):
    status, ctype, body = get(viewer, "/")
    assert status == 200 and "text/html" in ctype and b"Report Room" in body
    for asset, expected in (("/static/viewer.css", "text/css"), ("/static/viewer.js", "text/javascript")):
        status, ctype, body = get(viewer, asset)
        assert status == 200 and expected in ctype and body


@pytest.mark.unit
def test_runs_api_returns_the_digest(viewer):
    status, ctype, body = get(viewer, "/api/runs")
    assert status == 200 and "application/json" in ctype
    payload = json.loads(body)
    assert [run["ticker"] for run in payload["runs"]] == ["SMSPHARMA.NS", "NVDA"]
    assert payload["runs"][0]["action"] == "Sell"


@pytest.mark.unit
def test_run_api_returns_rendered_sections(viewer):
    status, _, body = get(viewer, "/api/runs/SMSPHARMA.NS_20260909_100652")
    assert status == 200
    payload = json.loads(body)
    assert payload["summary"]["rating"] == "Underweight"
    market = next(s for s in payload["sections"] if s["id"] == "analysts-market")
    assert "<h1" in market["html"] and market["toc"][0]["title"] == "Trend"


@pytest.mark.unit
def test_markdown_download_is_attached(viewer):
    connection = HTTPConnection(*viewer, timeout=10)
    connection.request("GET", "/api/runs/NVDA_20260908_143012/markdown")
    response = connection.getresponse()
    body = response.read()
    assert response.status == 200
    assert "attachment" in response.getheader("Content-Disposition")
    assert body.startswith(b"# Trading Analysis Report: NVDA")
    connection.close()


@pytest.mark.unit
@pytest.mark.parametrize("path", [
    "/api/runs/..%2F..%2Fetc",
    "/api/runs/unknown_run",
    "/static/../report_server.py",
    "/nonsense",
])
def test_bad_paths_are_refused(viewer, path):
    status, _, _ = get(viewer, path)
    assert status == 404


@pytest.mark.unit
def test_static_serving_is_limited_to_known_types(viewer):
    """welcome.txt exists in cli/static, but .txt has no CONTENT_TYPES entry."""
    assert (STATIC_DIR / "welcome.txt").is_file()
    status, _, _ = get(viewer, "/static/welcome.txt")
    assert status == 404
