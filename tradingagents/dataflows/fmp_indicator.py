"""Technical indicators computed from Financial Modeling Prep bars.

FMP does sell precomputed indicator endpoints, but they are deliberately not
used: indicators are derived here from the same cached OHLCV frame that feeds
``get_stock_data`` and the verified snapshot, so a value in the indicator report
can never disagree with the prices shown next to it. It also keeps the supported
indicator set and its parameters identical across vendors, which is what makes
the vendor swap invisible to the agents.
"""

from __future__ import annotations

from typing import Annotated

from .indicators_common import indicator_window_report


def get_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Return ``look_back_days`` of ``indicator`` values ending at ``curr_date``."""
    return indicator_window_report(
        symbol, indicator, curr_date, look_back_days, vendor="fmp"
    )
