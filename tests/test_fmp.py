"""Financial Modeling Prep vendor: symbol conventions, error taxonomy, and the
look-ahead guards that differ from the other vendors.

FMP is the default market-data vendor, so these cover the properties the rest of
the pipeline depends on rather than re-testing indicator maths (shared with the
yfinance path and covered by its own tests).
"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd
import pytest

from tradingagents.dataflows import fmp_common, fmp_fundamentals, fmp_news, fmp_stock
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from tradingagents.dataflows.fmp_symbols import normalize_fmp_symbol
from tradingagents.dataflows.interface import VENDOR_METHODS

# --- symbol conventions ----------------------------------------------------
# FMP agrees with Yahoo on equities and indices but not on forex, crypto or
# commodities, and the pipeline holds symbols in *both* notations (users type
# broker symbols; benchmark_map and the README use Yahoo's).


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Equities and ETFs pass through, exchange suffix intact.
        ("AAPL", "AAPL"),
        ("aapl", "AAPL"),
        ("RELIANCE.NS", "RELIANCE.NS"),
        ("7203.T", "7203.T"),
        ("000001.SS", "000001.SS"),
        # Indices use the same ^ notation as Yahoo.
        ("^GSPC", "^GSPC"),
        ("^NSEI", "^NSEI"),
        # Index CFD names resolve to the index.
        ("US500", "^GSPC"),
        ("NAS100", "^NDX"),
        # Commodities: FMP quotes a USD pair where Yahoo uses a future.
        ("XAUUSD", "GCUSD"),
        ("XAUUSD+", "GCUSD"),
        ("GC=F", "GCUSD"),      # Yahoo spelling must also resolve
        ("CL=F", "CLUSD"),
        ("SILVER", "SIUSD"),
        # Crypto: no separator.
        ("BTC-USD", "BTCUSD"),
        ("BTCUSD", "BTCUSD"),
        ("BTC-USDT", "BTCUSD"),
        ("ETH-USD", "ETHUSD"),
        # Forex: no =X suffix.
        ("EURUSD", "EURUSD"),
        ("EURUSD=X", "EURUSD"),
        ("GBPJPY", "GBPJPY"),
    ],
)
def test_symbol_normalization(raw, expected):
    assert normalize_fmp_symbol(raw) == expected


@pytest.mark.unit
def test_symbol_normalization_leaves_blank_input_alone():
    assert normalize_fmp_symbol("") == ""
    assert normalize_fmp_symbol(None) is None


# --- error taxonomy --------------------------------------------------------
# The router reacts by behavior, so a throttle and a bad key must not be
# conflated: a throttle tries the next vendor, a bad key marks it unavailable.


def _response(status=200, payload=None, text=""):
    resp = mock.Mock()
    resp.status_code = status
    resp.text = text
    resp.json.return_value = payload if payload is not None else {}
    resp.raise_for_status.return_value = None
    return resp


@pytest.mark.unit
class FMPErrorTaxonomyTests(unittest.TestCase):
    def test_missing_key_raises_not_configured(self):
        with mock.patch.dict("os.environ", {"FMP_API_KEY": ""}, clear=False), \
                self.assertRaises(VendorNotConfiguredError):
            fmp_common.get_api_key()

    def test_http_401_is_not_configured(self):
        resp = _response(401, {"Error Message": "Invalid API KEY."})
        with mock.patch.object(fmp_common.requests, "get", return_value=resp), \
                self.assertRaises(VendorNotConfiguredError):
            fmp_common.fmp_get("profile", {"symbol": "AAPL"})

    def test_http_429_is_rate_limit(self):
        resp = _response(429, {"Error Message": "Limit Reach."})
        with mock.patch.object(fmp_common.requests, "get", return_value=resp), \
                self.assertRaises(VendorRateLimitError):
            fmp_common.fmp_get("profile", {"symbol": "AAPL"})

    def test_limit_message_in_200_body_is_rate_limit(self):
        # A throttle notice can arrive with a 200; it mentions the plan/key too,
        # so it must be classified as a limit rather than a bad key.
        resp = _response(200, {"Error Message": "Limit Reach . Please upgrade your plan"})
        with mock.patch.object(fmp_common.requests, "get", return_value=resp), \
                self.assertRaises(VendorRateLimitError):
            fmp_common.fmp_get("profile", {"symbol": "AAPL"})

    def test_api_key_message_in_200_body_is_not_configured(self):
        resp = _response(200, {"Error Message": "Invalid API KEY provided"})
        with mock.patch.object(fmp_common.requests, "get", return_value=resp), \
                self.assertRaises(VendorNotConfiguredError):
            fmp_common.fmp_get("profile", {"symbol": "AAPL"})

    def test_plan_restricted_endpoint_is_not_configured(self):
        # "Cannot serve this call" -> the router should move to the next vendor
        # rather than abort the run.
        resp = _response(200, {"Error Message": "This is an Exclusive Endpoint"})
        with mock.patch.object(fmp_common.requests, "get", return_value=resp), \
                self.assertRaises(VendorNotConfiguredError):
            fmp_common.fmp_get("key-metrics", {"symbol": "AAPL"})

    def test_list_payload_is_returned_unchanged(self):
        resp = _response(200, [{"symbol": "AAPL"}])
        with mock.patch.object(fmp_common.requests, "get", return_value=resp):
            assert fmp_common.fmp_get("profile", {"symbol": "AAPL"}) == [{"symbol": "AAPL"}]

    def test_api_key_comes_from_the_environment_not_the_caller(self):
        # A caller-supplied apikey must not be able to override the configured
        # one, so no call site can accidentally leak or pin a different key.
        resp = _response(200, [])
        with mock.patch.dict("os.environ", {"FMP_API_KEY": "env-key"}, clear=False), \
                mock.patch.object(fmp_common.requests, "get", return_value=resp) as get:
            fmp_common.fmp_get("profile", {"symbol": "AAPL", "apikey": "caller-key"})
        assert get.call_args.kwargs["params"]["apikey"] == "env-key"


# --- OHLCV -----------------------------------------------------------------

_BARS = [
    # FMP returns newest-first; everything downstream assumes ascending.
    {"date": "2026-09-03", "open": 3.0, "high": 3.0, "low": 3.0, "close": 3.0, "volume": 30},
    {"date": "2026-09-02", "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 20},
    {"date": "2026-09-01", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 10},
]


@pytest.mark.unit
class FMPStockTests(unittest.TestCase):
    def test_empty_result_raises_no_market_data(self):
        # FMP answers an unknown symbol with [] and HTTP 200. That must become a
        # typed error so the router emits its one "no data" sentinel instead of
        # letting the agent invent a price.
        with mock.patch.object(fmp_stock, "fmp_get", return_value=[]), \
                self.assertRaises(NoMarketDataError):
            fmp_stock.fetch_ohlcv_frame("ZZZZ", "2026-09-01", "2026-09-03")

    def test_frame_is_sorted_ascending_with_standard_columns(self):
        with mock.patch.object(fmp_stock, "fmp_get", return_value=_BARS):
            frame = fmp_stock.fetch_ohlcv_frame("AAPL", "2026-09-01", "2026-09-03")
        assert list(frame.columns) == list(fmp_stock.OHLCV_COLUMNS)
        assert frame["Date"].is_monotonic_increasing
        assert frame["Close"].tolist() == [1.0, 2.0, 3.0]

    def test_missing_column_raises_no_market_data(self):
        with mock.patch.object(
            fmp_stock, "fmp_get", return_value=[{"date": "2026-09-01", "close": 1.0}]
        ), self.assertRaises(NoMarketDataError):
            fmp_stock.fetch_ohlcv_frame("AAPL", "2026-09-01", "2026-09-03")

    def test_report_notes_the_resolved_symbol(self):
        with mock.patch.object(fmp_stock, "fmp_get", return_value=_BARS):
            out = fmp_stock.get_stock("XAUUSD", "2026-09-01", "2026-09-03")
        assert "GCUSD (from XAUUSD)" in out  # provenance is visible to the agent
        assert "Total records: 3" in out

    def test_requests_the_inclusive_range(self):
        # FMP's from/to are inclusive, so unlike Yahoo there is no +1 day to add.
        with mock.patch.object(fmp_stock, "fmp_get", return_value=_BARS) as get:
            fmp_stock.get_stock("AAPL", "2026-09-01", "2026-09-03")
        params = get.call_args[0][1]
        assert params["from"] == "2026-09-01"
        assert params["to"] == "2026-09-03"

    def test_stale_frame_is_rejected(self):
        # A year-old frame must not be formatted into a report as if current
        # (#1021) — same guard the yfinance path applies.
        old = [dict(b, date=b["date"].replace("2026", "2024")) for b in _BARS]
        with mock.patch.object(fmp_stock, "fmp_get", return_value=old), \
                self.assertRaises(NoMarketDataError):
            fmp_stock.get_stock("AAPL", "2026-08-25", "2026-09-03")


# --- statements: point-in-time by filing date ------------------------------


def _statement(period_end, filing_date, revenue):
    return {
        "date": period_end,
        "symbol": "AAPL",
        "reportedCurrency": "USD",
        "filingDate": filing_date,
        "acceptedDate": f"{filing_date} 06:01:02",
        "fiscalYear": period_end[:4],
        "period": "Q3",
        "revenue": revenue,
    }


@pytest.mark.unit
class FMPStatementLookaheadTests(unittest.TestCase):
    """A fiscal period is withheld until the date it was actually filed.

    The other vendors only expose the period end, so they can only filter on
    that — which leaks: a quarter ending 2026-06-27 was not public until its
    2026-07-31 filing. FMP returns filingDate, so this path can be correct.
    """

    def test_unfiled_period_is_withheld(self):
        records = [_statement("2026-06-27", "2026-07-31", 109)]
        with mock.patch.object(fmp_fundamentals, "fmp_get", return_value=records), \
                self.assertRaises(NoMarketDataError) as ctx:
            fmp_fundamentals.get_income_statement("AAPL", "quarterly", "2026-07-01")
        assert "filed" in str(ctx.exception)

    def test_filed_period_is_served(self):
        records = [_statement("2026-06-27", "2026-07-31", 109)]
        with mock.patch.object(fmp_fundamentals, "fmp_get", return_value=records):
            out = fmp_fundamentals.get_income_statement("AAPL", "quarterly", "2026-08-01")
        assert "revenue" in out
        assert "109" in out
        assert "filed 2026-07-31" in out

    def test_only_periods_filed_by_the_analysis_date_are_kept(self):
        records = [
            _statement("2026-06-27", "2026-07-31", 109),   # not yet filed
            _statement("2026-03-28", "2026-05-01", 95),    # filed
        ]
        with mock.patch.object(fmp_fundamentals, "fmp_get", return_value=records):
            out = fmp_fundamentals.get_income_statement("AAPL", "quarterly", "2026-06-30")
        assert "95" in out
        assert "109" not in out

    def test_no_curr_date_serves_everything(self):
        records = [_statement("2026-06-27", "2026-07-31", 109)]
        with mock.patch.object(fmp_fundamentals, "fmp_get", return_value=records):
            out = fmp_fundamentals.get_income_statement("AAPL", "quarterly", None)
        assert "109" in out

    def test_empty_response_raises_no_market_data(self):
        with mock.patch.object(fmp_fundamentals, "fmp_get", return_value=[]), \
                self.assertRaises(NoMarketDataError):
            fmp_fundamentals.get_balance_sheet("ZZZZ", "quarterly", None)

    def test_frequency_maps_to_fmp_period(self):
        records = [_statement("2025-09-27", "2025-10-31", 400)]
        with mock.patch.object(fmp_fundamentals, "fmp_get", return_value=records) as get:
            fmp_fundamentals.get_balance_sheet("AAPL", "annual", None)
        assert get.call_args[0][1]["period"] == "annual"

        with mock.patch.object(fmp_fundamentals, "fmp_get", return_value=records) as get:
            fmp_fundamentals.get_cashflow("AAPL", "quarterly", None)
        assert get.call_args[0][1]["period"] == "quarter"


# --- fundamentals overview -------------------------------------------------


@pytest.mark.unit
class FMPFundamentalsTests(unittest.TestCase):
    def test_past_date_withholds_the_live_profile(self):
        # Same rule as every other fundamentals vendor (#1300): the profile has
        # no historical vintage, so it must not reach a backtest.
        with mock.patch.object(fmp_fundamentals, "fmp_get") as get:
            out = fmp_fundamentals.get_fundamentals("AAPL", "2020-01-01")
        get.assert_not_called()  # withheld before any request is made
        assert "withheld" in out

    def test_current_date_returns_the_profile(self):
        def _fake_get(endpoint, params=None):
            if endpoint == "profile":
                return [{"companyName": "Apple Inc.", "sector": "Technology"}]
            if endpoint == "quote":
                return [{"priceAvg50": 315.15}]
            return [{"priceToEarningsRatio": 32.0, "returnOnEquity": 0.27}]

        with mock.patch.object(fmp_fundamentals, "fmp_get", side_effect=_fake_get), \
                mock.patch.object(fmp_stock, "fmp_get", side_effect=_fake_get):
            out = fmp_fundamentals.get_fundamentals("AAPL", None)
        assert "Name: Apple Inc." in out
        assert "Sector: Technology" in out
        assert "50 Day Average: 315.15" in out
        assert "PE Ratio (TTM): 32.0" in out

    def test_unknown_symbol_raises_no_market_data(self):
        with mock.patch.object(fmp_stock, "fmp_get", return_value=[]), \
                self.assertRaises(NoMarketDataError):
            fmp_fundamentals.get_fundamentals("ZZZZ", None)

    def test_a_thin_section_does_not_sink_the_overview(self):
        # A plan restriction on ratios/key-metrics should thin the report, not
        # fail it — the profile alone is still useful identity context.
        def _fake_get(endpoint, params=None):
            if endpoint == "profile":
                return [{"companyName": "Apple Inc."}]
            if endpoint == "quote":
                return []
            raise fmp_common.FMPNotConfiguredError("Exclusive Endpoint")

        with mock.patch.object(fmp_fundamentals, "fmp_get", side_effect=_fake_get), \
                mock.patch.object(fmp_stock, "fmp_get", side_effect=_fake_get):
            out = fmp_fundamentals.get_fundamentals("AAPL", None)
        assert "Name: Apple Inc." in out


# --- news and insider ------------------------------------------------------


@pytest.mark.unit
class FMPNewsTests(unittest.TestCase):
    def setUp(self):
        set_config({"news_article_limit": 20, "global_news_lookback_days": 7,
                    "global_news_article_limit": 10})

    def test_articles_outside_the_window_are_dropped(self):
        articles = [
            {"title": "In window", "publishedDate": "2026-09-02 10:00:00",
             "publisher": "P", "url": "u1", "text": "t1"},
            {"title": "After window", "publishedDate": "2026-09-20 10:00:00",
             "publisher": "P", "url": "u2", "text": "t2"},
        ]
        with mock.patch.object(fmp_news, "fmp_get", return_value=articles):
            out = fmp_news.get_news("AAPL", "2026-09-01", "2026-09-03")
        assert "In window" in out
        assert "After window" not in out

    def test_no_articles_reports_plainly(self):
        with mock.patch.object(fmp_news, "fmp_get", return_value=[]):
            out = fmp_news.get_news("AAPL", "2026-09-01", "2026-09-03")
        assert "No news found" in out

    def test_news_query_uses_the_resolved_symbol(self):
        with mock.patch.object(fmp_news, "fmp_get", return_value=[]) as get:
            fmp_news.get_news("BTC-USD", "2026-09-01", "2026-09-03")
        assert get.call_args[0][1]["symbols"] == "BTCUSD"

    def test_global_news_requests_the_historical_window(self):
        # The window must reach the API, not just filter its response: a
        # backtest needs the news of *that* week, not today's headlines.
        with mock.patch.object(fmp_news, "fmp_get", return_value=[]) as get:
            fmp_news.get_global_news("2026-06-05", look_back_days=7, limit=5)
        params = get.call_args[0][1]
        assert params["from"] == "2026-05-29"
        assert params["to"] == "2026-06-05"

    def test_global_news_deduplicates_by_title(self):
        articles = [
            {"title": "Same story", "publishedDate": "2026-06-04 10:00:00", "publisher": "A"},
            {"title": "Same story", "publishedDate": "2026-06-04 11:00:00", "publisher": "B"},
        ]
        with mock.patch.object(fmp_news, "fmp_get", return_value=articles):
            out = fmp_news.get_global_news("2026-06-05", look_back_days=7, limit=10)
        assert out.count("Same story") == 1

    def test_insider_empty_is_not_a_no_data_error(self):
        # Many valid symbols simply have no Form 4 filings; that must not mark
        # the symbol unavailable.
        with mock.patch.object(fmp_news, "fmp_get", return_value=[]):
            out = fmp_news.get_insider_transactions("AAPL")
        assert "No insider transactions" in out

    def test_insider_rows_are_rendered(self):
        records = [{
            "filingDate": "2026-09-03", "transactionDate": "2026-09-01",
            "reportingName": "Newstead Jennifer", "typeOfOwner": "officer: SVP, GC",
            "transactionType": "S-Sale", "securitiesTransacted": 1439, "price": 317.01,
        }]
        with mock.patch.object(fmp_news, "fmp_get", return_value=records):
            out = fmp_news.get_insider_transactions("AAPL")
        assert "Newstead Jennifer" in out
        assert '"officer: SVP, GC"' in out  # comma-bearing field is quoted
        assert "Total records: 1" in out

    def test_insider_filter_respects_curr_date(self):
        records = [
            {"filingDate": "2026-09-03", "reportingName": "Later"},
            {"filingDate": "2026-08-01", "reportingName": "Earlier"},
        ]
        with mock.patch.object(fmp_news, "fmp_get", return_value=records):
            out = fmp_news.get_insider_transactions("AAPL", "2026-08-15")
        assert "Earlier" in out
        assert "Later" not in out


# --- router wiring ---------------------------------------------------------


@pytest.mark.unit
def test_fmp_serves_every_market_data_method():
    """Every method the yfinance vendor implements must have an FMP impl.

    Without this the default config could route a category to fmp and get a
    "vendor not available" ValueError at run time.
    """
    yfinance_methods = {
        method for method, vendors in VENDOR_METHODS.items() if "yfinance" in vendors
    }
    missing = {m for m in yfinance_methods if "fmp" not in VENDOR_METHODS[m]}
    assert not missing, f"FMP is missing implementations for {sorted(missing)}"


@pytest.mark.unit
def test_ohlcv_cache_is_namespaced_per_vendor(tmp_path, monkeypatch):
    """Two vendors' bars must never be served for one another.

    They are differently adjusted and keyed by different symbol spellings, so a
    shared cache filename would silently mix them.
    """
    import tradingagents.dataflows.stockstats_utils as su

    monkeypatch.setattr(su, "get_config", lambda: {"data_cache_dir": str(tmp_path)})

    def _bars(canonical, start, end):
        return pd.DataFrame({
            "Date": pd.date_range(end=pd.Timestamp.today().normalize(), periods=3, freq="D"),
            "Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0],
            "Low": [1.0, 2.0, 3.0], "Close": [1.0, 2.0, 3.0], "Volume": [1, 2, 3],
        })

    monkeypatch.setitem(su.OHLCV_SOURCES["yfinance"], "download", _bars)
    monkeypatch.setitem(su.OHLCV_SOURCES["fmp"], "download", _bars)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    su.load_ohlcv("AAPL", today, "yfinance")
    su.load_ohlcv("AAPL", today, "fmp")

    names = sorted(p.name for p in tmp_path.iterdir())
    assert any("-YFin-data-" in n for n in names)
    assert any("-FMP-data-" in n for n in names)


@pytest.mark.unit
def test_unknown_ohlcv_vendor_is_rejected():
    import tradingagents.dataflows.stockstats_utils as su

    with pytest.raises(ValueError, match="Unknown OHLCV vendor"):
        su.load_ohlcv("AAPL", "2026-09-01", "not_a_vendor")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("fmp", "fmp"),
        ("yfinance", "yfinance"),
        ("fmp,yfinance", "fmp"),
        ("yfinance,fmp", "yfinance"),
        # alpha_vantage serves indicators from its own API, not local bars, so
        # it has no OHLCV source; the next usable vendor (or the default) wins.
        ("alpha_vantage,fmp", "fmp"),
        ("alpha_vantage", "yfinance"),
        ("default", "yfinance"),
        ("", "yfinance"),
        (None, "yfinance"),
    ],
)
def test_resolve_ohlcv_vendor(configured, expected):
    import tradingagents.dataflows.stockstats_utils as su

    assert su.resolve_ohlcv_vendor(configured) == expected
