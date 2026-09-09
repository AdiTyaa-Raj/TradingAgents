"""Symbol normalization for the Financial Modeling Prep vendor.

FMP agrees with Yahoo on equities (including exchange suffixes — ``RELIANCE.NS``,
``SHEL.L``, ``7203.T``, ``000001.SS``) and on index symbols (``^GSPC``,
``^NSEI``), but differs on every other asset class:

    instrument        Yahoo wants      FMP wants     rule
    ---------------   --------------   -----------   --------------------------
    spot forex        ``EURUSD=X``     ``EURUSD``    no suffix
    crypto            ``BTC-USD``      ``BTCUSD``    no separator
    gold / silver     ``GC=F``         ``GCUSD``     quoted as a USD pair,
    oil / gas         ``CL=F``         ``CLUSD``     not a dated future

Both conventions have to be accepted on input, not just broker-style symbols:
the rest of the pipeline is written in Yahoo notation (``benchmark_map`` in
``default_config`` holds ``^NSEI``/``000001.SS``, the README documents crypto as
``BTC-USD``, and saved memory-log entries carry whatever the user first typed).
So this resolves broker symbols *and* Yahoo symbols onto FMP notation, mirroring
:func:`tradingagents.dataflows.symbol_utils.normalize_symbol` for the Yahoo side.

Adding an instrument means appending a table row — no call-site changes.
"""

from __future__ import annotations

import logging
import re

from .symbol_utils import _FOREX_CURRENCIES, crypto_base

logger = logging.getLogger(__name__)

# Commodities FMP quotes as ``<ROOT>USD`` pairs. Keyed by every spelling the
# pipeline may hold: the broker/spot name, and the Yahoo front-month future.
_COMMODITIES = {
    # Precious metals
    "XAUUSD": "GCUSD", "XAU": "GCUSD", "GOLD": "GCUSD", "GC=F": "GCUSD",
    "XAGUSD": "SIUSD", "XAG": "SIUSD", "SILVER": "SIUSD", "SI=F": "SIUSD",
    "XPTUSD": "PLUSD", "PL=F": "PLUSD",
    "XPDUSD": "PAUSD", "PA=F": "PAUSD",
    # Energy
    "WTICOUSD": "CLUSD", "USOIL": "CLUSD", "WTI": "CLUSD", "CL=F": "CLUSD",
    "BCOUSD": "BZUSD", "UKOIL": "BZUSD", "BRENT": "BZUSD", "BZ=F": "BZUSD",
    "NATGAS": "NGUSD", "XNGUSD": "NGUSD", "NG=F": "NGUSD",
    "COPPER": "HGUSD", "XCUUSD": "HGUSD", "HG=F": "HGUSD",
}

# Index CFD names -> the index symbol FMP serves (identical to Yahoo's, so
# Yahoo-native ``^GSPC`` needs no row: it falls through unchanged).
_INDICES = {
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

_ALIASES = {**_COMMODITIES, **_INDICES}

# Characters FMP symbols actually use (letters, digits, and the structural
# ``.`` exchange suffix, ``^`` index marker, ``-`` and ``_`` separators).
_FMP_SAFE = re.compile(r"^[A-Za-z0-9.\-_^]+$")


def normalize_fmp_symbol(raw: str) -> str:
    """Map a user/broker/Yahoo symbol to its canonical FMP symbol.

    Resolution order (first match wins):
      1. Alias table — commodities and index CFDs, keyed by broker *and* Yahoo
         spelling (``XAUUSD`` and ``GC=F`` both give ``GCUSD``).
      2. Crypto rule: a known crypto base quoted in USD/USDT/USDC, with or
         without a separator -> ``BASEUSD`` (``BTC-USD`` -> ``BTCUSD``).
      3. Forex rule: six letters that are two ISO currency codes, with Yahoo's
         ``=X`` suffix stripped first -> ``EURUSD``.
      4. Otherwise the upper-cased symbol unchanged — equities, ETFs, and
         index symbols, whose FMP and Yahoo spellings agree.

    A trailing ``+`` (broker CFD marker, e.g. ``XAUUSD+``) is stripped before
    matching. Purely syntactic: no network calls, safe on every request.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper().rstrip("+")

    if s in _ALIASES:
        canonical = _ALIASES[s]
    else:
        crypto = crypto_base(s)
        # Strip Yahoo's spot-forex suffix so ``EURUSD=X`` reaches the pair rule.
        spot = s[:-2] if s.endswith("=X") else s
        if crypto is not None:
            canonical = f"{crypto}USD"
        elif (
            len(spot) == 6
            and spot[:3] in _FOREX_CURRENCIES
            and spot[3:] in _FOREX_CURRENCIES
        ):
            canonical = spot
        else:
            canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to FMP symbol %r", raw, canonical)
    return canonical


def is_fmp_safe(symbol: str) -> bool:
    """True when ``symbol`` only contains characters FMP symbols use."""
    return bool(symbol) and _FMP_SAFE.fullmatch(symbol) is not None
