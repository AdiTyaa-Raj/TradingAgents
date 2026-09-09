"""Technical-indicator windows computed locally from cached OHLCV bars.

The indicator maths is `stockstats` over a price frame, so it is identical for
every vendor that can supply daily bars — only the price source differs. This
module holds that one implementation and takes the source as an argument, so
adding a bar vendor means adding a row to ``stockstats_utils.OHLCV_SOURCES``
rather than copying ~150 lines of indicator descriptions per vendor.

(Alpha Vantage is the exception and does not use this path: it serves computed
indicator series from its own API endpoints.)
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from .errors import NoMarketDataError
from .stockstats_utils import (
    DEFAULT_OHLCV_VENDOR,
    StockstatsUtils,
    load_ohlcv,
)

logger = logging.getLogger(__name__)

# Supported indicators and the guidance shown alongside each value, so the
# analyst reads the number with its caveats rather than in isolation.
INDICATOR_DESCRIPTIONS = {
    # Moving Averages
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. "
        "Usage: Identify trend direction and serve as dynamic support/resistance. "
        "Tips: It lags price; combine with faster indicators for timely signals."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. "
        "Usage: Confirm overall market trend and identify golden/death cross setups. "
        "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. "
        "Usage: Capture quick shifts in momentum and potential entry points. "
        "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
    ),
    # MACD Related
    "macd": (
        "MACD: Computes momentum via differences of EMAs. "
        "Usage: Look for crossovers and divergence as signals of trend changes. "
        "Tips: Confirm with other indicators in low-volatility or sideways markets."
    ),
    "macds": (
        "MACD Signal: An EMA smoothing of the MACD line. "
        "Usage: Use crossovers with the MACD line to trigger trades. "
        "Tips: Should be part of a broader strategy to avoid false positives."
    ),
    "macdh": (
        "MACD Histogram: Shows the gap between the MACD line and its signal. "
        "Usage: Visualize momentum strength and spot divergence early. "
        "Tips: Can be volatile; complement with additional filters in fast-moving markets."
    ),
    # Momentum Indicators
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. "
        "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
        "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
    ),
    # Volatility Indicators
    "boll": (
        "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
        "Usage: Acts as a dynamic benchmark for price movement. "
        "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
    ),
    "boll_ub": (
        "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
        "Usage: Signals potential overbought conditions and breakout zones. "
        "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
    ),
    "boll_lb": (
        "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
        "Usage: Indicates potential oversold conditions. "
        "Tips: Use additional analysis to avoid false reversal signals."
    ),
    "atr": (
        "ATR: Averages true range to measure volatility. "
        "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
        "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
    ),
    # Volume-Based Indicators
    "vwma": (
        "VWMA: A moving average weighted by volume. "
        "Usage: Confirm trends by integrating price action with volume data. "
        "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
    ),
    "mfi": (
        "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
        "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
        "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
    ),
}


def indicator_series(
    symbol: str,
    indicator: str,
    curr_date: str,
    vendor: str = DEFAULT_OHLCV_VENDOR,
) -> dict:
    """Compute ``indicator`` for every cached bar up to ``curr_date``.

    One fetch plus one vectorized stockstats pass, returning a
    ``{"YYYY-MM-DD": value}`` map — far cheaper than recomputing per requested
    day.
    """
    from stockstats import wrap

    data = load_ohlcv(symbol, curr_date, vendor)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df[indicator]  # triggers stockstats to calculate the indicator

    result_dict = {}
    for _, row in df.iterrows():
        value = row[indicator]
        result_dict[row["Date"]] = "N/A" if pd.isna(value) else str(value)

    return result_dict


def indicator_value(
    symbol: str,
    indicator: str,
    curr_date: str,
    vendor: str = DEFAULT_OHLCV_VENDOR,
) -> str:
    """Single-day indicator value, used as the per-day fallback path."""
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")

    try:
        value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date_dt.strftime("%Y-%m-%d"),
            vendor,
        )
    except NoMarketDataError:
        raise  # Unknown/delisted symbol — let the router emit the sentinel
    except Exception as e:
        logger.warning("Stockstats indicator %s failed on %s: %s", indicator, curr_date, e)
        return ""

    return str(value)


def indicator_window_report(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    vendor: str = DEFAULT_OHLCV_VENDOR,
) -> str:
    """Render ``look_back_days`` of ``indicator`` values ending at ``curr_date``."""
    if indicator not in INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: "
            f"{list(INDICATOR_DESCRIPTIONS.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    try:
        indicator_data = indicator_series(symbol, indicator, curr_date, vendor)

        date_values = []
        current_dt = curr_date_dt
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            date_values.append((
                date_str,
                indicator_data.get(
                    date_str, "N/A: Not a trading day (weekend or holiday)"
                ),
            ))
            current_dt = current_dt - relativedelta(days=1)

        ind_string = "".join(f"{date_str}: {value}\n" for date_str, value in date_values)

    except NoMarketDataError:
        raise  # Unknown/delisted symbol — let the router emit the sentinel
    except Exception as e:
        logger.warning("Bulk stockstats fetch failed, falling back per-day: %s", e)
        # Fall back to the per-day path if the bulk calculation fails.
        ind_string = ""
        current_dt = curr_date_dt
        while current_dt >= before:
            value = indicator_value(
                symbol, indicator, current_dt.strftime("%Y-%m-%d"), vendor
            )
            ind_string += f"{current_dt.strftime('%Y-%m-%d')}: {value}\n"
            current_dt = current_dt - relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + INDICATOR_DESCRIPTIONS.get(indicator, "No description available.")
    )
