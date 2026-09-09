"""News and insider-transaction data from Financial Modeling Prep."""

from __future__ import annotations

import contextlib
import logging
from datetime import datetime
from typing import Annotated

from dateutil.relativedelta import relativedelta

from .config import get_config
from .date_window import in_window
from .fmp_common import fmp_get
from .fmp_symbols import normalize_fmp_symbol

logger = logging.getLogger(__name__)

# FMP timestamps are "YYYY-MM-DD HH:MM:SS" in UTC, sometimes date-only.
_TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")


def _parse_published(value) -> datetime | None:
    """Parse an FMP publish timestamp, or None when absent/unparseable.

    None means "undated", which ``in_window`` keeps only for a live run — in a
    backtest an undated item cannot be proven not to be future (#1126).
    """
    if not value:
        return None
    text = str(value).strip()
    for fmt in _TIMESTAMP_FORMATS:
        with contextlib.suppress(ValueError):
            return datetime.strptime(text, fmt)
    with contextlib.suppress(ValueError):
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    return None


def _render_articles(articles: list[dict], start_dt: datetime, end_dt: datetime) -> tuple[str, int]:
    """Format articles that fall inside the window; returns (text, kept count)."""
    body = ""
    kept = 0
    for article in articles:
        if not isinstance(article, dict):
            continue
        if not in_window(_parse_published(article.get("publishedDate")), start_dt, end_dt):
            continue

        title = article.get("title") or "No title"
        publisher = article.get("publisher") or article.get("site") or "Unknown"
        body += f"### {title} (source: {publisher})\n"
        if article.get("publishedDate"):
            body += f"Published: {article['publishedDate']}\n"
        if article.get("text"):
            body += f"{article['text']}\n"
        if article.get("url"):
            body += f"Link: {article['url']}\n"
        body += "\n"
        kept += 1
    return body, kept


def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Ticker news published between ``start_date`` and ``end_date`` (inclusive).

    FMP filters by date server-side; the shared ``in_window`` guard is applied
    anyway so every news source in the codebase honors exactly one window rule.
    """
    article_limit = get_config()["news_article_limit"]
    canonical = normalize_fmp_symbol(ticker)
    resolved = "" if canonical == ticker.upper() else f" (resolved to {canonical})"

    try:
        payload = fmp_get(
            "news/stock",
            {
                "symbols": canonical,
                "from": start_date,
                "to": end_date,
                "limit": article_limit,
            },
        )
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"

    articles = payload if isinstance(payload, list) else []
    if not articles:
        return f"No news found for {ticker}{resolved}"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    body, kept = _render_articles(articles[:article_limit], start_dt, end_dt)

    if kept == 0:
        return (
            f"No news found for {ticker}{resolved} between {start_date} and {end_date}"
        )

    return f"## {ticker}{resolved} News, from {start_date} to {end_date}:\n\n{body}"


def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int | None, "Days to look back"] = None,
    limit: Annotated[int | None, "Max articles to return"] = None,
) -> str:
    """Global/macro market news for the window ending at ``curr_date``.

    Uses FMP's curated general-market feed, which accepts a real ``from``/``to``
    range — so a historical run gets the news of *that* window rather than
    today's headlines filtered down to nothing.

    The ``global_news_queries`` config does not apply to this vendor: FMP serves
    one editorial macro feed rather than free-text search, so topic coverage is
    the provider's, not query-driven.
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    try:
        payload = fmp_get(
            "news/general-latest",
            {"from": start_date, "to": curr_date, "page": 0, "limit": limit},
        )
    except Exception as e:
        return f"Error fetching global news: {str(e)}"

    articles = payload if isinstance(payload, list) else []
    if not articles:
        return f"No global news found for {curr_date}"

    # Deduplicate by title: the feed can carry the same story from syndicated
    # outlets, which would otherwise eat the article budget.
    unique: list[dict] = []
    seen: set[str] = set()
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = (article.get("title") or "").strip()
        if title and title in seen:
            continue
        seen.add(title)
        unique.append(article)

    body, kept = _render_articles(unique[:limit], start_dt, curr_dt)

    if kept == 0:
        return f"No global news found between {start_date} and {curr_date}"

    return f"## Global Market News, from {start_date} to {curr_date}:\n\n{body}"


# Insider-transaction columns, in filing order.
_INSIDER_FIELDS = (
    ("filingDate", "Filing Date"),
    ("transactionDate", "Transaction Date"),
    ("reportingName", "Insider"),
    ("typeOfOwner", "Role"),
    ("transactionType", "Transaction"),
    ("acquisitionOrDisposition", "Acquired/Disposed"),
    ("securitiesTransacted", "Shares"),
    ("price", "Price"),
    ("securitiesOwned", "Shares Owned After"),
    ("securityName", "Security"),
    ("formType", "Form"),
)

INSIDER_LIMIT = 50


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Recent insider transactions for ``ticker`` as CSV.

    An empty result is normal (many valid symbols have no Form 4 filings), so it
    is reported plainly rather than as a no-data error that would mark the
    symbol unavailable.

    ``curr_date`` filters to filings already public by that date when supplied.
    The tool wrapper does not pass one today — matching the other vendors — but
    accepting it keeps the point-in-time guard available without a signature
    change.
    """
    canonical = normalize_fmp_symbol(ticker)

    try:
        payload = fmp_get(
            "insider-trading/search",
            {"symbol": canonical, "page": 0, "limit": INSIDER_LIMIT},
        )
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"

    records = [r for r in payload if isinstance(r, dict)] if isinstance(payload, list) else []

    if curr_date:
        cutoff = str(curr_date)[:10]
        records = [r for r in records if str(r.get("filingDate", ""))[:10] <= cutoff]

    if not records:
        return f"No insider transactions reported for symbol '{canonical}'"

    def _cell(value) -> str:
        if value is None:
            return ""
        text = str(value)
        return f'"{text}"' if ("," in text or '"' in text) else text

    rows = [",".join(label for _, label in _INSIDER_FIELDS)]
    for record in records:
        rows.append(",".join(_cell(record.get(key)) for key, _ in _INSIDER_FIELDS))

    header = f"# Insider Transactions data for {canonical}\n"
    header += f"# Total records: {len(records)}\n"
    header += "# Source: Financial Modeling Prep (SEC Form 3/4/5 filings)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + "\n".join(rows) + "\n"
