"""Company fundamentals and financial statements from Financial Modeling Prep.

Statements are rendered as CSV with line items as rows and fiscal periods as
columns — the shape the yfinance vendor produced and the analyst prompts were
written against — so switching vendors does not change how the reports read.

Point-in-time handling is stricter here than on the other vendors, because FMP
supplies the field that makes it possible. yfinance and Alpha Vantage only
expose the fiscal period end, so they can only filter on that, which leaks: a
quarter ending 2026-06-27 was not public until its 2026-07-31 filing, yet a
period-end filter would serve it to a run dated 2026-07-01. FMP returns
``filingDate``/``acceptedDate``, so a period is withheld until the date it was
actually filed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

from .date_window import withhold_live_profile
from .errors import NoMarketDataError
from .fmp_common import fmp_get
from .fmp_stock import get_profile, get_quote
from .fmp_symbols import normalize_fmp_symbol

logger = logging.getLogger(__name__)

# Periods requested per statement. Enough history for trend and YoY comparison
# without flooding the agent's context.
STATEMENT_LIMIT = 8

_STATEMENT_ENDPOINTS = {
    "balance_sheet": ("balance-sheet-statement", "Balance Sheet"),
    "cash_flow": ("cash-flow-statement", "Cash Flow"),
    "income_statement": ("income-statement", "Income Statement"),
}

# Record keys that describe the filing rather than the business. Kept out of the
# line-item rows and used for the column headers / point-in-time filter instead.
_METADATA_KEYS = (
    "date", "symbol", "reportedCurrency", "cik", "filingDate", "acceptedDate",
    "fiscalYear", "period", "link", "finalLink",
)


def _fmp_period(freq: str) -> str:
    """Map the pipeline's ``annual``/``quarterly`` onto FMP's period values."""
    return "annual" if str(freq).lower().startswith("a") else "quarter"


def _published_on(record: dict) -> str:
    """The date a filing became public: its filing date, else accepted date.

    Falls back to the fiscal period end when neither is present, which is the
    best the record allows — never more permissive than the other vendors.
    """
    for key in ("filingDate", "acceptedDate"):
        value = record.get(key)
        if value:
            return str(value)[:10]
    return str(record.get("date", ""))[:10]


def _filter_to_published(records: list[dict], curr_date: str | None) -> list[dict]:
    """Drop periods that had not been filed by ``curr_date`` (look-ahead guard)."""
    if not curr_date:
        return records
    cutoff = str(curr_date)[:10]
    return [r for r in records if _published_on(r) <= cutoff]


def _statement_csv(records: list[dict], canonical: str, kind: str, freq: str) -> str:
    """Render statement records as CSV: line items as rows, periods as columns."""
    # Newest period first, matching the yfinance statement layout.
    ordered = sorted(records, key=lambda r: str(r.get("date", "")), reverse=True)

    columns = [str(r.get("date", ""))[:10] for r in ordered]

    # Union of line items, preserving first-seen order so the statement reads in
    # its natural top-to-bottom order rather than alphabetically.
    line_items: list[str] = []
    for record in ordered:
        for key in record:
            if key not in _METADATA_KEYS and key not in line_items:
                line_items.append(key)

    def _cell(value) -> str:
        if value is None:
            return ""
        text = str(value)
        # Quote anything that would break the CSV row.
        return f'"{text}"' if ("," in text or '"' in text) else text

    rows = [",".join(["", *columns])]
    for item in line_items:
        rows.append(",".join([item, *(_cell(r.get(item)) for r in ordered)]))

    currency = ordered[0].get("reportedCurrency") or "unknown"
    periods = ", ".join(
        f"{r.get('period', '?')} {r.get('fiscalYear', '?')} (filed {_published_on(r)})"
        for r in ordered
    )

    header = f"# {kind} data for {canonical} ({freq})\n"
    header += f"# Reported currency: {currency}\n"
    header += f"# Periods: {periods}\n"
    header += "# Source: Financial Modeling Prep\n"
    header += "# Only periods already filed as of the analysis date are included.\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + "\n".join(rows) + "\n"


def _get_statement(
    kind: str,
    ticker: str,
    freq: str,
    curr_date: str | None,
) -> str:
    endpoint, label = _STATEMENT_ENDPOINTS[kind]
    canonical = normalize_fmp_symbol(ticker)

    payload = fmp_get(
        endpoint,
        {"symbol": canonical, "period": _fmp_period(freq), "limit": STATEMENT_LIMIT},
    )
    records = [r for r in payload if isinstance(r, dict)] if isinstance(payload, list) else []

    if not records:
        raise NoMarketDataError(ticker, canonical, f"no {label.lower()} data")

    published = _filter_to_published(records, curr_date)
    if not published:
        raise NoMarketDataError(
            ticker,
            canonical,
            f"no {label.lower()} period had been filed as of {curr_date}",
        )

    return _statement_csv(published, canonical, label, _fmp_period(freq))


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Balance sheet statements filed on or before ``curr_date``."""
    return _get_statement("balance_sheet", ticker, freq, curr_date)


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Cash flow statements filed on or before ``curr_date``."""
    return _get_statement("cash_flow", ticker, freq, curr_date)


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Income statements filed on or before ``curr_date``."""
    return _get_statement("income_statement", ticker, freq, curr_date)


# Profile/quote/ratio fields shown in the fundamentals overview, as
# (label, source, key) where source picks which payload the value comes from.
_OVERVIEW_FIELDS = (
    ("Name", "profile", "companyName"),
    ("Sector", "profile", "sector"),
    ("Industry", "profile", "industry"),
    ("Exchange", "profile", "exchangeFullName"),
    ("Currency", "profile", "currency"),
    ("Market Cap", "profile", "marketCap"),
    ("Price", "profile", "price"),
    ("Beta", "profile", "beta"),
    ("52 Week Range", "profile", "range"),
    ("Average Volume", "profile", "averageVolume"),
    ("50 Day Average", "quote", "priceAvg50"),
    ("200 Day Average", "quote", "priceAvg200"),
    ("52 Week High", "quote", "yearHigh"),
    ("52 Week Low", "quote", "yearLow"),
    ("PE Ratio (TTM)", "ratios", "priceToEarningsRatio"),
    ("Price to Book", "ratios", "priceToBookRatio"),
    ("Price to Sales", "ratios", "priceToSalesRatio"),
    ("Dividend Yield", "ratios", "dividendYield"),
    ("Profit Margin", "ratios", "netProfitMargin"),
    ("Operating Margin", "ratios", "operatingProfitMargin"),
    ("Gross Margin", "ratios", "grossProfitMargin"),
    ("Current Ratio", "ratios", "currentRatio"),
    ("Quick Ratio", "ratios", "quickRatio"),
    ("Debt to Equity", "ratios", "debtToEquityRatio"),
    ("Return on Equity", "metrics", "returnOnEquity"),
    ("Return on Assets", "metrics", "returnOnAssets"),
    ("Return on Invested Capital", "metrics", "returnOnInvestedCapital"),
    ("Enterprise Value", "metrics", "enterpriseValue"),
    ("EV / EBITDA", "metrics", "evToEBITDA"),
    ("EV / Free Cash Flow", "metrics", "evToFreeCashFlow"),
    ("Free Cash Flow Yield", "metrics", "freeCashFlowYield"),
    ("Net Debt to EBITDA", "metrics", "netDebtToEBITDA"),
    ("Working Capital", "metrics", "workingCapital"),
)


def _latest_record(endpoint: str, canonical: str) -> dict:
    """Most recent record from a periodic endpoint, or ``{}`` when unavailable.

    Best-effort: the overview is an aggregate of several endpoints, and a plan
    restriction or gap in one of them should thin the report, not fail it.
    """
    try:
        payload = fmp_get(endpoint, {"symbol": canonical, "period": "quarter", "limit": 1})
    except Exception as exc:  # noqa: BLE001 — one thin section beats no overview
        logger.debug("FMP %s unavailable for %s: %s", endpoint, canonical, exc)
        return {}
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return {}


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "analysis date in YYYY-MM-DD format"] = None,
) -> str:
    """Company fundamentals overview: profile, quote levels, ratios, key metrics.

    Like every other fundamentals vendor, this is withheld for a past
    ``curr_date``: the profile, quote and TTM ratios are present-day values with
    no historical vintage, so serving them into a backtest would leak
    post-decision information (#1300). Point-in-time figures come from the
    statement tools, which filter by filing date.
    """
    canonical = normalize_fmp_symbol(ticker)

    # Guard before the request: the response would only be discarded, and the
    # answer does not depend on it.
    withheld = withhold_live_profile(curr_date, canonical)
    if withheld:
        return withheld

    profile = get_profile(canonical)
    if not profile:
        raise NoMarketDataError(ticker, canonical, "no company profile returned")

    quote = get_quote(canonical)

    sources = {
        "profile": profile,
        "quote": quote,
        "ratios": _latest_record("ratios", canonical),
        "metrics": _latest_record("key-metrics", canonical),
    }

    lines = []
    for label, source, key in _OVERVIEW_FIELDS:
        value = sources[source].get(key)
        # Skip absent values and FMP's zero-filled placeholders for fields it
        # does not carry for this instrument, so the agent doesn't read a
        # missing ratio as a real 0.
        if value in (None, "", 0):
            continue
        lines.append(f"{label}: {value}")

    if not lines:
        raise NoMarketDataError(ticker, canonical, "no fundamental fields returned")

    header = f"# Company Fundamentals for {canonical}\n"
    header += "# Source: Financial Modeling Prep (present-day profile, quote and TTM ratios)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + "\n".join(lines)
