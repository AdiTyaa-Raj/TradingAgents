#!/usr/bin/env python3
"""One-command, non-interactive TradingAgents run.

    python analyze.py NVDA          # or: ./analyze.py NVDA

Everything the interactive CLI asks for is answered from defaults so the
ticker is the only thing you have to type (last argument, as with any other
flagged command — ``python analyze.py --depth 3 NVDA`` works too):

* analysis date  -> today (crypto/stock alike; ``--date`` to backtest a past day)
* analysts       -> all of them (fundamentals is dropped for crypto, which has none)
* output language-> English
* LLM provider   -> ``claude_code``: the locally signed-in Claude Code CLI, no API key
* models         -> quick=sonnet, deep=opus (the catalog defaults for the provider)

Precedence for every setting is flag > ``TRADINGAGENTS_*`` env var > default
here, matching the interactive CLI's own env-override rules, so a .env that
already pins a provider or model keeps working.

The run itself reuses ``cli.main.run_analysis`` — the same live dashboard,
per-section report files under ``results_dir``, and the saved report tree
under ``./reports/<TICKER>_<timestamp>/`` that the interactive CLI produces.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import cli.main as cli_main
from cli.models import AnalystType, AssetType
from cli.utils import (
    ANALYST_ORDER,
    confirm_claude_code_cli,
    detect_asset_type,
    filter_analysts_for_asset_type,
    normalize_ticker_symbol,
    resolve_backend_url,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import get_model_options

# No API key, no per-token billing: authenticated by whatever session
# `claude login` established for the Claude Code CLI.
DEFAULT_PROVIDER = "claude_code"
DEFAULT_LANGUAGE = "English"
DEFAULT_DEPTH = 3 # 1 shallow / 3 medium / 5 deep, same mapping as the CLI menu


def _catalog_default(provider: str, mode: str) -> str | None:
    """First real model the interactive menu would offer for this provider/mode."""
    try:
        options = get_model_options(provider, mode)
    except KeyError:
        return None
    return next((value for _, value in options if value != "custom"), None)


def _env_set(name: str) -> bool:
    return bool(os.environ.get(name))


def _valid_date(value: str) -> str:
    try:
        parsed = datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a date in YYYY-MM-DD format"
        ) from None
    if parsed > datetime.date.today():
        raise argparse.ArgumentTypeError("analysis date cannot be in the future")
    return value


def _parse_analysts(value: str, asset_type: AssetType) -> list[AnalystType]:
    """Resolve a comma-separated analyst list, keeping the canonical run order."""
    known = {a.value: a for _, a in ANALYST_ORDER}
    wanted = []
    for raw in value.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in known:
            raise SystemExit(
                f"Unknown analyst {raw.strip()!r}. Choose from: {', '.join(known)}"
            )
        wanted.append(known[name])
    if not wanted:
        raise SystemExit("--analysts needs at least one analyst")
    ordered = [a for _, a in ANALYST_ORDER if a in wanted]
    allowed = filter_analysts_for_asset_type(ordered, asset_type)
    dropped = [a.value for a in ordered if a not in allowed]
    if dropped:
        print(f"Skipping {', '.join(dropped)} — not applicable to {asset_type.value}.")
    if not allowed:
        raise SystemExit(f"No requested analyst applies to a {asset_type.value} ticker.")
    return allowed


def build_selections(args: argparse.Namespace) -> dict:
    """Assemble the same dict ``cli.main.get_user_selections`` returns, unattended."""
    ticker = normalize_ticker_symbol(args.ticker)
    asset_type = detect_asset_type(ticker)

    if args.analysts:
        analysts = _parse_analysts(args.analysts, asset_type)
    else:
        analysts = filter_analysts_for_asset_type(
            [a for _, a in ANALYST_ORDER], asset_type
        )

    if args.provider:
        provider = args.provider.lower()
    elif _env_set("TRADINGAGENTS_LLM_PROVIDER"):
        provider = DEFAULT_CONFIG["llm_provider"].lower()
    else:
        provider = DEFAULT_PROVIDER

    quick = args.quick_model or (
        DEFAULT_CONFIG["quick_think_llm"] if _env_set("TRADINGAGENTS_QUICK_THINK_LLM")
        else _catalog_default(provider, "quick")
    )
    deep = args.deep_model or (
        DEFAULT_CONFIG["deep_think_llm"] if _env_set("TRADINGAGENTS_DEEP_THINK_LLM")
        else _catalog_default(provider, "deep")
    )
    if not quick or not deep:
        raise SystemExit(
            f"No default model known for provider {provider!r}. "
            "Pass --quick-model and --deep-model explicitly."
        )

    if args.depth is not None:
        depth = args.depth
    elif _env_set("TRADINGAGENTS_MAX_DEBATE_ROUNDS") and _env_set("TRADINGAGENTS_MAX_RISK_ROUNDS"):
        depth = DEFAULT_CONFIG["max_debate_rounds"]
    else:
        depth = DEFAULT_DEPTH

    language = args.language or (
        DEFAULT_CONFIG["output_language"] if _env_set("TRADINGAGENTS_OUTPUT_LANGUAGE")
        else DEFAULT_LANGUAGE
    )

    # The provider's own default endpoint unless TRADINGAGENTS_LLM_BACKEND_URL
    # overrides it — identical to what the interactive menu resolves.
    backend_url = resolve_backend_url(provider, env_url=DEFAULT_CONFIG["backend_url"])

    return {
        "ticker": ticker,
        "asset_type": asset_type.value,
        "analysis_date": args.date,
        "analysts": analysts,
        "research_depth": depth,
        "llm_provider": provider,
        "backend_url": backend_url,
        "shallow_thinker": quick,
        "deep_thinker": deep,
        "google_thinking_level": DEFAULT_CONFIG["google_thinking_level"],
        "openai_reasoning_effort": DEFAULT_CONFIG["openai_reasoning_effort"],
        "anthropic_effort": args.effort or DEFAULT_CONFIG["anthropic_effort"],
        "output_language": language,
    }


def preflight(selections: dict) -> None:
    """Fail fast on a missing CLI / API key instead of dying at the first LLM call."""
    provider = selections["llm_provider"]
    if provider == "claude_code":
        confirm_claude_code_cli()  # prints the resolved binary, exits if absent
        return
    env_var = get_api_key_env(provider)
    if env_var and not os.environ.get(env_var):
        raise SystemExit(
            f"{env_var} is not set. Export it or add it to .env, then re-run "
            f"(or drop --provider to use the keyless {DEFAULT_PROVIDER} default)."
        )


def announce(selections: dict) -> None:
    print()
    print(f"  Ticker      : {selections['ticker']} ({selections['asset_type']})")
    print(f"  Date        : {selections['analysis_date']}")
    print(f"  Analysts    : {', '.join(a.value for a in selections['analysts'])}")
    print(f"  Language    : {selections['output_language']}")
    print(f"  Provider    : {selections['llm_provider']}")
    print(f"  Models      : quick={selections['shallow_thinker']} deep={selections['deep_thinker']}")
    print(f"  Depth       : {selections['research_depth']} debate/risk round(s)")
    print()


def install_auto_answers(save: bool, save_path: Path | None, display: bool):
    """Answer ``run_analysis``'s post-run prompts without a human at the keyboard.

    Returns the original ``typer.prompt`` so the caller can restore it. Unknown
    prompts fall through to their own default, so a new question added upstream
    degrades to the CLI's default answer rather than hanging on stdin.
    """
    original = cli_main.typer.prompt

    def auto_prompt(text="", *args, default=None, **kwargs):
        label = str(text).strip().lower()
        if "save report" in label:
            return "Y" if save else "N"
        if "save path" in label:
            return str(save_path) if save_path else (default or "")
        if "display full report" in label:
            return "Y" if display else "N"
        return default if default is not None else ""

    cli_main.typer.prompt = auto_prompt
    return original


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze.py",
        description="Run the full TradingAgents workflow for one ticker, unattended.",
        epilog="Example: python analyze.py NVDA   |   python analyze.py --depth 3 0700.HK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. NVDA, 0700.HK, BTC-USD")
    parser.add_argument(
        "--date", type=_valid_date,
        default=datetime.date.today().strftime("%Y-%m-%d"),
        help="Analysis date (YYYY-MM-DD); defaults to today",
    )
    parser.add_argument(
        "--analysts",
        help="Comma-separated subset of market,social,news,fundamentals (default: all)",
    )
    parser.add_argument(
        "--depth", type=int, choices=(1, 3, 5),
        help=f"Debate/risk rounds: 1 shallow, 3 medium, 5 deep (default: {DEFAULT_DEPTH})",
    )
    parser.add_argument("--language", help=f"Report language (default: {DEFAULT_LANGUAGE})")
    parser.add_argument(
        "--provider",
        help=f"LLM provider key, e.g. openai, anthropic (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument("--quick-model", help="Override the quick-thinking model")
    parser.add_argument("--deep-model", help="Override the deep-thinking model")
    parser.add_argument(
        "--effort", choices=("low", "medium", "high"),
        help="Claude/Claude Code effort level (default: the provider's own)",
    )
    parser.add_argument(
        "--save-path",
        help="Where to write the report tree (default: ./reports/<TICKER>_<timestamp>)",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing the report tree")
    parser.add_argument(
        "--no-display", action="store_true",
        help="Skip printing the full report to the terminal when the run ends",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument(
        "--checkpoint", dest="checkpoint", action="store_true", default=None,
        help="Save state after each node so a crashed run can resume",
    )
    checkpoint.add_argument(
        "--no-checkpoint", dest="checkpoint", action="store_false",
        help="Disable checkpointing regardless of TRADINGAGENTS_CHECKPOINT_ENABLED",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selections = build_selections(args)
    preflight(selections)
    announce(selections)

    # `run_analysis` owns the live dashboard, per-node logging and report
    # writing; the only interactive seams are the selection questionnaire and
    # the two post-run prompts, both of which are supplied here.
    original_get_selections = cli_main.get_user_selections
    original_prompt = install_auto_answers(
        save=not args.no_save,
        save_path=Path(args.save_path).expanduser() if args.save_path else None,
        display=not args.no_display,
    )
    cli_main.get_user_selections = lambda: selections
    try:
        cli_main.run_analysis(checkpoint=args.checkpoint)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    finally:
        cli_main.get_user_selections = original_get_selections
        cli_main.typer.prompt = original_prompt
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
