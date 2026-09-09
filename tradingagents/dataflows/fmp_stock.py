"""OHLCV price data from Financial Modeling Prep.

``historical-price-eod/full`` serves split-adjusted daily bars for every asset
class the pipeline handles — equities, indices, forex, crypto, commodities — so
one endpoint replaces the per-asset-class handling Yahoo needed.

Adjustment note: these bars are split-adjusted but *not* dividend-adjusted,
where yfinance's ``auto_adjust=True`` applied both. Splits are the adjustment
that matters for continuity (an unadjusted 10:1 split puts a 90% cliff in the
series and wrecks every moving average); dividends shift a long window by a few
percent at most. Using one series everywhere is the more important property:
``get_stock_data``, the technical indicators, and the verified snapshot the
market analyst is told to treat as ground truth (#830) must agree to the cent,
which they cannot do if the price report and the indicators use differently
adjusted closes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import pandas as pd

from .errors import NoMarketDataError
from .fmp_common import fmp_get
from .fmp_symbols import normalize_fmp_symbol

# The frame shape the rest of the pipeline expects (yfinance's capitalized
# columns), so FMP frames drop straight into the shared stockstats helpers.
OHLCV_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")

_FIELD_MAP = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def fetch_ohlcv_frame(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    canonical: str | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV bars as a date-ascending DataFrame.

    Returns the standard ``OHLCV_COLUMNS`` frame. ``canonical`` lets a caller
    that has already normalized the symbol avoid re-resolving it.

    Raises:
        NoMarketDataError: the symbol is unknown, delisted, or not covered —
            FMP answers those with an empty list and HTTP 200.
    """
    resolved = canonical or normalize_fmp_symbol(symbol)

    payload = fmp_get(
        "historical-price-eod/full",
        {"symbol": resolved, "from": start_date, "to": end_date},
    )

    if not isinstance(payload, list) or not payload:
        raise NoMarketDataError(
            symbol, resolved, f"no rows between {start_date} and {end_date}"
        )

    frame = pd.DataFrame(payload).rename(columns=_FIELD_MAP)

    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise NoMarketDataError(
            symbol, resolved, f"response missing column(s) {', '.join(missing)}"
        )

    # FMP returns newest-first; every downstream consumer (stockstats windows,
    # the staleness guard, the CSV report) assumes ascending dates.
    frame = frame[list(OHLCV_COLUMNS)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if frame.empty:
        raise NoMarketDataError(symbol, resolved, "no rows with a parseable date")

    return frame


def get_stock(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Return daily OHLCV for ``symbol`` over an inclusive date range as CSV.

    Both bounds are inclusive — FMP's ``from``/``to`` are inclusive already, so
    unlike Yahoo there is no exclusive-end day to compensate for (#986/#987).
    """
    # Validate the inputs the same way the yfinance path does, so a malformed
    # date fails here rather than as an opaque vendor response.
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    canonical = normalize_fmp_symbol(symbol)
    data = fetch_ohlcv_frame(symbol, start_date, end_date, canonical=canonical)

    # Reject a stale frame (e.g. a long-delisted symbol still answering with
    # year-old bars) before it is formatted into the report (#1021). Imported
    # locally: stockstats_utils imports this module for its OHLCV source
    # registry, so a module-level import would be circular.
    from .stockstats_utils import _assert_ohlcv_not_stale

    _assert_ohlcv_not_stale(data, end_date, symbol, canonical)

    for col in ("Open", "High", "Low", "Close"):
        data[col] = pd.to_numeric(data[col], errors="coerce").round(2)

    csv_string = data.to_csv(index=False)

    # Note the resolved symbol when it differs so the agent (and user) can see
    # which instrument was actually priced.
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Stock data for {label} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += "# Source: Financial Modeling Prep (split-adjusted daily bars)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def _first_record(endpoint: str, canonical: str) -> dict:
    """First record from a single-symbol endpoint, or ``{}`` when absent."""
    payload = fmp_get(endpoint, {"symbol": canonical})
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return {}


def get_profile(symbol: str) -> dict:
    """Return FMP's company profile record for ``symbol``, or ``{}`` if absent.

    Shared by the fundamentals report and the deterministic instrument-identity
    lookup, which need the same company/sector/industry fields. Empty for
    non-equities: ``profile`` covers listed companies and funds only, so
    commodities, forex, crypto and indices resolve through :func:`get_quote`.
    """
    return _first_record("profile", normalize_fmp_symbol(symbol))


def get_quote(symbol: str) -> dict:
    """Return FMP's current quote record for ``symbol``, or ``{}`` if absent.

    Unlike ``profile`` this covers every asset class, and carries the
    instrument's display name and market (``Gold Futures``/``COMMODITY``,
    ``Bitcoin USD``/``CRYPTO``, ``S&P 500``/``INDEX``) — the only identity
    available for a non-equity.
    """
    return _first_record("quote", normalize_fmp_symbol(symbol))
