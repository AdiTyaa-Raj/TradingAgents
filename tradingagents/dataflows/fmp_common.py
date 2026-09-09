"""Shared HTTP/auth layer for the Financial Modeling Prep (FMP) vendor.

All FMP endpoints used here live under the ``/stable`` API surface and take the
key as an ``apikey`` query parameter. Responses are JSON: a list of records for
data endpoints (an empty list means "no coverage for this symbol", *not* an
error) and an ``{"Error Message": ...}`` object for failures.

Failures are classified into the shared vendor-error taxonomy (``errors.py``) so
the router reacts by behavior rather than by vendor: a throttle skips to the next
vendor, a bad/missing key marks the vendor unavailable.

Docs: https://site.financialmodelingprep.com/developer/docs
"""

from __future__ import annotations

import logging
import os

import requests

from .errors import VendorNotConfiguredError, VendorRateLimitError

logger = logging.getLogger(__name__)

API_BASE_URL = "https://financialmodelingprep.com/stable"

# Network timeout (seconds) so a stalled FMP request can't hang the CLI/agents
# indefinitely — same rationale as the Alpha Vantage vendor (#990).
REQUEST_TIMEOUT = 30


class FMPNotConfiguredError(VendorNotConfiguredError):
    """Raised when FMP is selected but no usable API key is configured.

    A ``VendorNotConfiguredError`` (and thus still a ``ValueError``), so the
    router's "vendor unavailable" handling and existing ValueError callers both
    keep working.
    """


class FMPRateLimitError(VendorRateLimitError):
    """Raised when FMP throttles the request; the router tries the next vendor."""


def get_api_key() -> str:
    """Return the configured FMP API key, or raise if it is missing."""
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise FMPNotConfiguredError(
            "FMP_API_KEY environment variable is not set. Get a key at "
            "https://site.financialmodelingprep.com/developer/docs"
        )
    return api_key


def _classify_error(message: str) -> Exception:
    """Map an FMP ``Error Message`` body onto the vendor-error taxonomy.

    Rate-limit phrasing is checked first: FMP's throttle notices also mention
    the plan/key ("Limit Reach . Please upgrade your plan"), so testing for key
    wording first would misreport a throttle as a bad key (the same trap the
    Alpha Vantage vendor documents for #991).
    """
    low = message.lower()
    if any(
        marker in low
        for marker in ("limit reach", "rate limit", "too many requests", "upgrade your plan")
    ):
        return FMPRateLimitError(f"FMP rate limit exceeded: {message}")
    if "api key" in low or "apikey" in low:
        return FMPNotConfiguredError(f"FMP API key invalid or missing: {message}")
    if "exclusive endpoint" in low or "not available under your" in low:
        # Plan does not include this endpoint. Treated as "this vendor cannot
        # serve the call" so the router moves on instead of aborting the run.
        return FMPNotConfiguredError(f"FMP endpoint not available on this plan: {message}")
    return RuntimeError(f"FMP request failed: {message}")


def fmp_get(endpoint: str, params: dict | None = None) -> list | dict:
    """GET a ``/stable`` endpoint and return the decoded JSON payload.

    Args:
        endpoint: Path under ``/stable``, e.g. ``"historical-price-eod/full"``.
        params: Query parameters; the API key is added here, never by callers.

    Returns:
        The decoded payload — normally a list of records, possibly empty. An
        empty list is a valid "no data" answer and is returned as-is for the
        caller to turn into a typed ``NoMarketDataError``.

    Raises:
        FMPNotConfiguredError: key missing/invalid, or endpoint not on the plan.
        FMPRateLimitError: request was throttled.
        RuntimeError: any other error reported in the response body.
        requests.HTTPError: unexpected transport-level failure.
    """
    query = dict(params or {})
    query["apikey"] = get_api_key()

    response = requests.get(
        f"{API_BASE_URL}/{endpoint.lstrip('/')}",
        params=query,
        timeout=REQUEST_TIMEOUT,
    )

    # Classify by status before parsing: an invalid key returns 401 and a
    # throttle returns 429, both with a JSON body raise_for_status would hide.
    if response.status_code in (401, 403):
        raise FMPNotConfiguredError(
            f"FMP rejected the API key (HTTP {response.status_code}): "
            f"{_error_message(response) or 'no detail returned'}"
        )
    if response.status_code == 429:
        raise FMPRateLimitError(
            f"FMP rate limit exceeded (HTTP 429): "
            f"{_error_message(response) or 'no detail returned'}"
        )

    response.raise_for_status()

    payload = response.json()

    # Data endpoints return a list; only error bodies are objects.
    if isinstance(payload, dict):
        message = payload.get("Error Message") or payload.get("error")
        if message:
            raise _classify_error(str(message))

    return payload


def _error_message(response: requests.Response) -> str:
    """Best-effort extraction of FMP's error text from a non-2xx response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:200]
    if isinstance(payload, dict):
        return str(payload.get("Error Message") or payload.get("error") or "").strip()
    return ""
