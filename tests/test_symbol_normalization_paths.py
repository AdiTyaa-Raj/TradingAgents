"""Symbol normalization must apply on every vendor path, not just price fetch.

Regression tests for #983 (instrument identity), #984 (reflection returns), and
the news path: a broker symbol like XAUUSD must resolve to the same instrument
the price path uses, so identity, realized-return, and news lookups hit the
right instrument instead of failing/mismatching.

Each vendor has its own convention for the same instrument — Yahoo prices gold
as the COMEX future ``GC=F``, FMP as the USD pair ``GCUSD`` — so the assertion
is per vendor, and normalization must never be skipped on either.
"""
import pandas as pd

import tradingagents.agents.utils.agent_utils as au
import tradingagents.dataflows.stockstats_utils as su
import tradingagents.dataflows.yfinance_news as ynews
from tradingagents.dataflows.config import set_config
from tradingagents.graph.trading_graph import TradingAgentsGraph


def test_identity_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        @property
        def info(self):
            return {"longName": "Gold Futures", "quoteType": "FUTURE"}

    set_config({"data_vendors": {"fundamental_data": "yfinance"}})
    monkeypatch.setattr(au.yf, "Ticker", FakeTicker)
    au.resolve_instrument_identity.cache_clear()

    identity = au.resolve_instrument_identity("XAUUSD")

    assert seen["symbol"] == "GC=F"  # normalized, not the raw broker symbol
    assert identity.get("company_name") == "Gold Futures"


def test_identity_lookup_normalizes_symbol_for_fmp(monkeypatch):
    seen = {}

    def _fake_get(endpoint, params=None):
        seen["symbol"] = (params or {}).get("symbol")
        return [{"companyName": "Gold", "exchange": "COMMODITY"}]

    set_config({"data_vendors": {"fundamental_data": "fmp"}})
    monkeypatch.setattr("tradingagents.dataflows.fmp_stock.fmp_get", _fake_get)
    au.resolve_instrument_identity.cache_clear()

    identity = au.resolve_instrument_identity("XAUUSD")

    assert seen["symbol"] == "GCUSD"  # FMP's USD-pair convention, not GC=F
    assert identity.get("company_name") == "Gold"


def _fake_bars(*_args, **_kwargs):
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    return pd.DataFrame({
        "Date": pd.date_range(start="2025-01-02", periods=len(prices), freq="D"),
        "Open": prices, "High": prices, "Low": prices,
        "Close": prices, "Volume": [1] * len(prices),
    })


def test_fetch_returns_normalizes_symbol(monkeypatch):
    queried = []

    def _download(canonical, start, end):
        queried.append(canonical)
        return _fake_bars()

    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    monkeypatch.setitem(su.OHLCV_SOURCES["yfinance"], "download", _download)

    # _fetch_returns does not use ``self``; call unbound to avoid building the graph.
    raw, alpha, days, resolved = TradingAgentsGraph._fetch_returns(
        None, "XAUUSD", "2025-01-02", holding_days=5, benchmark="SPY"
    )

    assert queried[0] == "GC=F"  # stock symbol normalized (#984)
    assert queried[1] == "SPY"   # benchmark left as the canonical symbol
    assert raw is not None and days is not None
    assert resolved == "2025-01-07"  # resolution date recorded (#1251)


def test_fetch_returns_normalizes_symbol_for_fmp(monkeypatch):
    queried = []

    def _download(canonical, start, end):
        queried.append(canonical)
        return _fake_bars()

    set_config({"data_vendors": {"core_stock_apis": "fmp"}})
    monkeypatch.setitem(su.OHLCV_SOURCES["fmp"], "download", _download)

    raw, alpha, days, resolved = TradingAgentsGraph._fetch_returns(
        None, "XAUUSD", "2025-01-02", holding_days=5, benchmark="SPY"
    )

    assert queried[0] == "GCUSD"  # FMP's convention for gold, not GC=F (#984)
    assert queried[1] == "SPY"
    assert resolved == "2025-01-07"


def test_news_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def get_news(self, count):
            return []

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(ynews, "yf_retry", lambda fn: fn())

    out = ynews.get_news_yfinance("XAUUSD", "2025-01-01", "2025-01-10")

    assert seen["symbol"] == "GC=F"   # news queried with the canonical symbol
    assert "XAUUSD" in out            # the user's ticker stays in the report
    assert "GC=F" in out              # provenance noted
