"""Claude Code CLI provider — subscription auth, no API key.

Covers the three things that make this provider different from every other
one: it shells out instead of calling an HTTP endpoint, it must never
authenticate with an API key, and its tool calling is a prompt protocol parsed
back into ``AIMessage.tool_calls`` rather than a native tool-use API.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import tradingagents.llm_clients.claude_code_client as cc
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.validators import validate_model


@tool
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Fetch OHLCV price history for a ticker over a date range."""
    return ""


def _envelope(result, **extra):
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": result,
        "session_id": "sess-1",
        "stop_reason": "end_turn",
        "total_cost_usd": 0.01,
        "num_turns": 1,
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 5,
            "output_tokens": 20,
        },
    }
    payload.update(extra)
    return payload


@pytest.fixture()
def fake_cli(monkeypatch):
    """Stub out the binary lookup and the subprocess call.

    Returns a recorder whose ``calls`` list holds the (argv, prompt, env) of
    every invocation and whose ``envelope`` is what the CLI "returns".
    """
    recorder = SimpleNamespace(calls=[], envelope=_envelope("hello"), returncode=0, stderr="")
    monkeypatch.setattr(cc, "find_cli", lambda cli_path=None: "/usr/local/bin/claude")

    def _run(argv, **kwargs):
        recorder.calls.append(
            SimpleNamespace(argv=argv, prompt=kwargs.get("input"), env=kwargs.get("env"),
                            cwd=kwargs.get("cwd"))
        )
        return subprocess.CompletedProcess(
            argv, recorder.returncode, json.dumps(recorder.envelope), recorder.stderr
        )

    monkeypatch.setattr(cc.subprocess, "run", _run)
    return recorder


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["claude_code", "claude-code", "Claude_Code"])
def test_factory_routes_claude_code(provider):
    client = create_llm_client(provider, "sonnet")
    assert type(client).__name__ == "ClaudeCodeClient"


@pytest.mark.unit
def test_no_api_key_env_and_any_model_accepted():
    # The whole point of the provider: there is no key for the CLI to prompt for.
    assert get_api_key_env("claude_code") is None
    assert validate_model("claude_code", "opus") is True
    assert validate_model("claude_code", "claude-fable-5") is True


@pytest.mark.unit
def test_provider_appears_in_cli_menu_and_catalog():
    from cli.utils import _llm_provider_table
    from tradingagents.llm_clients.model_catalog import get_model_options

    assert "claude_code" in {key for _, key, _ in _llm_provider_table()}
    assert "sonnet" in {value for _, value in get_model_options("claude_code", "quick")}
    assert "opus" in {value for _, value in get_model_options("claude_code", "deep")}


# ---------------------------------------------------------------------------
# CLI discovery
# ---------------------------------------------------------------------------


@pytest.fixture()
def offpath_home(monkeypatch, tmp_path):
    """A home directory with nothing on PATH.

    Reproduces how the CLI looks to a run started from a GUI app, an IDE run
    configuration or a launchd job: an installed binary that PATH says nothing
    about.
    """
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(empty_bin))
    monkeypatch.delenv("TRADINGAGENTS_CLAUDE_CODE_CLI", raising=False)
    # The machine running the tests may well have a real system-wide install;
    # keep only the locations that the patched HOME redirects.
    monkeypatch.setattr(
        cc, "_KNOWN_CLI_PATHS", tuple(p for p in cc._KNOWN_CLI_PATHS if p.startswith("~"))
    )

    def install(relative_path):
        binary = tmp_path / relative_path
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        return binary

    return SimpleNamespace(home=tmp_path, install=install)


@pytest.mark.unit
@pytest.mark.parametrize("location", [".local/bin/claude", ".claude/local/claude"])
def test_cli_off_path_is_found_in_its_install_location(offpath_home, location):
    binary = offpath_home.install(location)

    assert cc.find_cli() == str(binary)


@pytest.mark.unit
def test_cli_off_path_is_found_under_a_node_version_manager(offpath_home):
    offpath_home.install(".nvm/versions/node/v20.11.0/bin/claude")
    newest = offpath_home.install(".nvm/versions/node/v22.3.0/bin/claude")

    assert cc.find_cli() == str(newest)


@pytest.mark.unit
def test_missing_cli_is_reported_as_missing(offpath_home):
    assert cc.find_cli() is None


@pytest.mark.unit
def test_configured_cli_path_expands_home(offpath_home, monkeypatch):
    binary = offpath_home.install("opt/claude-code/claude")
    monkeypatch.setenv("TRADINGAGENTS_CLAUDE_CODE_CLI", "~/opt/claude-code/claude")

    assert cc.find_cli() == str(binary)


@pytest.mark.unit
def test_configured_cli_path_never_resolves_to_another_install(offpath_home, monkeypatch):
    offpath_home.install(".local/bin/claude")
    monkeypatch.setenv("TRADINGAGENTS_CLAUDE_CODE_CLI", str(offpath_home.home / "nope/claude"))

    assert cc.find_cli() is None


@pytest.mark.unit
def test_workspace_dir_expands_home(offpath_home, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_CLAUDE_CODE_WORKSPACE", "~/.tradingagents/ws")

    assert cc.default_workspace_dir() == str(offpath_home.home / ".tradingagents/ws")
    assert not (offpath_home.home / "~").exists()


# ---------------------------------------------------------------------------
# Invocation shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_argv_disables_builtin_tools_and_forwards_effort(fake_cli):
    llm = create_llm_client("claude_code", "opus", effort="high").get_llm()
    llm.invoke([HumanMessage(content="hi")])

    argv = fake_cli.calls[0].argv
    assert argv[0] == "/usr/local/bin/claude"
    assert "--print" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--effort") + 1] == "high"
    # Claude Code's own tools must be off: the graph runs the agents' tools.
    assert argv[argv.index("--tools") + 1] == ""
    for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
        assert flag in argv


@pytest.mark.unit
def test_api_key_is_stripped_from_the_child_environment(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-used")
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    llm.invoke([HumanMessage(content="hi")])

    assert "ANTHROPIC_API_KEY" not in fake_cli.calls[0].env


@pytest.mark.unit
def test_api_key_auth_can_be_opted_into(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-explicit")
    monkeypatch.setenv("TRADINGAGENTS_CLAUDE_CODE_API_KEY_AUTH", "true")
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    llm.invoke([HumanMessage(content="hi")])

    assert fake_cli.calls[0].env["ANTHROPIC_API_KEY"] == "sk-ant-explicit"


@pytest.mark.unit
def test_blank_api_key_never_reaches_the_cli(fake_cli, monkeypatch):
    # A key left empty in .env is worse than no key: the CLI treats it as a
    # configured credential and fails auth instead of using the session.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("TRADINGAGENTS_CLAUDE_CODE_API_KEY_AUTH", "true")
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    llm.invoke([HumanMessage(content="hi")])

    assert "ANTHROPIC_API_KEY" not in fake_cli.calls[0].env


@pytest.mark.unit
def test_max_tokens_forwarded_as_cli_env(fake_cli):
    llm = create_llm_client("claude_code", "sonnet", max_tokens=4096).get_llm()
    llm.invoke([HumanMessage(content="hi")])

    assert fake_cli.calls[0].env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"


@pytest.mark.unit
def test_runs_outside_the_callers_working_directory(fake_cli, monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_CLAUDE_CODE_WORKSPACE", str(tmp_path / "ws"))
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    llm.invoke([HumanMessage(content="hi")])

    assert fake_cli.calls[0].cwd == str(tmp_path / "ws")


@pytest.mark.unit
def test_missing_cli_raises_install_hint(monkeypatch):
    monkeypatch.setattr(cc, "find_cli", lambda cli_path=None: None)
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    with pytest.raises(cc.ClaudeCodeError, match="npm install -g @anthropic-ai/claude-code"):
        llm.invoke([HumanMessage(content="hi")])


@pytest.mark.unit
def test_cli_error_envelope_surfaces(fake_cli):
    fake_cli.envelope = _envelope("boom", is_error=True, subtype="error_during_execution")
    llm = create_llm_client("claude_code", "sonnet", max_retries=0).get_llm()
    with pytest.raises(cc.ClaudeCodeError, match="boom"):
        llm.invoke([HumanMessage(content="hi")])


@pytest.mark.unit
def test_transient_failure_is_retried(fake_cli, monkeypatch):
    calls = {"n": 0}
    real_run = cc.subprocess.run

    def _flaky(argv, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(argv, 1, "", "temporary upstream failure")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(cc.subprocess, "run", _flaky)
    llm = create_llm_client("claude_code", "sonnet", max_retries=1).get_llm()
    assert llm.invoke([HumanMessage(content="hi")]).content == "hello"
    assert calls["n"] == 2


@pytest.mark.unit
def test_usage_metadata_includes_cached_input_tokens(fake_cli):
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    message = llm.invoke([HumanMessage(content="hi")])

    assert message.usage_metadata == {
        "input_tokens": 115,   # 10 fresh + 100 cache read + 5 cache creation
        "output_tokens": 20,
        "total_tokens": 135,
    }


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_prompt_splits_system_from_transcript():
    system, prompt = cc.render_prompt([
        SystemMessage(content="You are an analyst."),
        HumanMessage(content="Analyze NVDA"),
        AIMessage(content="", tool_calls=[
            {"name": "get_stock_data", "args": {"symbol": "NVDA"}, "id": "c1", "type": "tool_call"}
        ]),
        ToolMessage(content="date,close", tool_call_id="c1", name="get_stock_data"),
    ])

    assert system == "You are an analyst."
    assert "## User\nAnalyze NVDA" in prompt
    assert '"name": "get_stock_data"' in prompt          # assistant turn replayed
    assert "## Tool result: get_stock_data\ndate,close" in prompt


@pytest.mark.unit
def test_empty_transcript_still_produces_a_prompt():
    _, prompt = cc.render_prompt([SystemMessage(content="Only a system prompt.")])
    assert prompt.strip()


@pytest.mark.unit
def test_system_prompt_overrides_the_harness_environment(fake_cli):
    # Claude Code reports the real wall-clock date even under --system-prompt;
    # without the override a past-dated run gets look-ahead date ranges.
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    llm.invoke([SystemMessage(content="Today's date is 2024-05-10."), HumanMessage(content="go")])

    argv = fake_cli.calls[0].argv
    system_prompt = argv[argv.index("--system-prompt") + 1]
    assert "Ignore every environment detail" in system_prompt
    assert system_prompt.endswith("Today's date is 2024-05-10.")


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bound_tools_are_described_in_the_system_prompt(fake_cli):
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    llm.bind_tools([get_stock_data]).invoke([HumanMessage(content="go")])

    argv = fake_cli.calls[0].argv
    system_prompt = argv[argv.index("--system-prompt") + 1]
    assert "# Tool calling protocol" in system_prompt
    assert "### get_stock_data" in system_prompt
    assert '"start_date"' in system_prompt


@pytest.mark.unit
def test_tool_call_block_becomes_tool_calls(fake_cli):
    fake_cli.envelope = _envelope(
        '```tool_calls\n[{"name": "get_stock_data", "arguments": '
        '{"symbol": "NVDA", "start_date": "2024-01-10", "end_date": "2024-05-10"}}]\n```'
    )
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    message = llm.bind_tools([get_stock_data]).invoke([HumanMessage(content="go")])

    assert [call["name"] for call in message.tool_calls] == ["get_stock_data"]
    assert message.tool_calls[0]["args"]["symbol"] == "NVDA"
    assert message.content == ""


@pytest.mark.unit
def test_final_answer_carries_no_tool_calls(fake_cli):
    fake_cli.envelope = _envelope("## Market report\nNVDA is trending up.")
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    message = llm.bind_tools([get_stock_data]).invoke([HumanMessage(content="go")])

    assert message.tool_calls == []
    assert message.content.startswith("## Market report")


@pytest.mark.unit
def test_tool_block_is_parsed_even_with_surrounding_prose():
    text, calls = cc.parse_tool_calls(
        'Let me pull the data.\n```tool_calls\n[{"name": "get_indicators", '
        '"arguments": {"indicator": "rsi"}}]\n```\n'
    )
    assert text == "Let me pull the data."
    assert calls[0]["name"] == "get_indicators"
    assert calls[0]["args"] == {"indicator": "rsi"}


@pytest.mark.unit
def test_bare_json_array_is_accepted_as_tool_calls():
    _, calls = cc.parse_tool_calls('[{"name": "get_news", "arguments": {"symbol": "NVDA"}}]')
    assert calls[0]["name"] == "get_news"


@pytest.mark.unit
def test_prose_containing_a_json_fence_is_not_a_tool_call():
    text = 'The API returns:\n```json\n{"price": 100}\n```\nThat is all.'
    parsed, calls = cc.parse_tool_calls(text)
    assert calls == []
    assert parsed == text


@pytest.mark.unit
def test_a_report_holding_a_named_json_array_is_not_a_tool_call(fake_cli):
    # A report can legitimately contain `[{"name": ...}]`; only names that
    # match a bound tool may be read as calls, or the report would be lost.
    report = 'Peers:\n```json\n[{"name": "AMD", "pe": 40}]\n```\nEnd of report.'
    fake_cli.envelope = _envelope(report)
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    message = llm.bind_tools([get_stock_data]).invoke([HumanMessage(content="go")])

    assert message.tool_calls == []
    assert message.content == report


@pytest.mark.unit
def test_tool_calls_are_only_parsed_when_tools_are_bound(fake_cli):
    fake_cli.envelope = _envelope('```tool_calls\n[{"name": "x", "arguments": {}}]\n```')
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    message = llm.invoke([HumanMessage(content="go")])

    assert message.tool_calls == []


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class _Decision(BaseModel):
    action: str = Field(description="Buy, Hold or Sell")
    confidence: float = Field(description="0-1")


@pytest.mark.unit
def test_structured_output_uses_the_cli_json_schema_flag(fake_cli):
    fake_cli.envelope = _envelope(
        '{"action": "Buy", "confidence": 0.8}',
        structured_output={"action": "Buy", "confidence": 0.8},
    )
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    result = llm.with_structured_output(_Decision).invoke([HumanMessage(content="go")])

    argv = fake_cli.calls[0].argv
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert schema["properties"]["action"]["type"] == "string"
    assert isinstance(result, _Decision)
    assert result.action == "Buy"


@pytest.mark.unit
def test_structured_output_falls_back_to_parsing_the_text(fake_cli):
    # Older CLI builds may omit structured_output even when the schema holds.
    fake_cli.envelope = _envelope('{"action": "Hold", "confidence": 0.5}')
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    result = llm.with_structured_output(_Decision).invoke([HumanMessage(content="go")])

    assert result.action == "Hold"


@pytest.mark.unit
def test_unparseable_structured_reply_raises_for_the_free_text_fallback(fake_cli):
    # bind_structured/invoke_structured_or_freetext catch this and retry as prose.
    fake_cli.envelope = _envelope("I could not comply.")
    llm = create_llm_client("claude_code", "sonnet").get_llm()
    with pytest.raises(ValueError, match="no structured output"):
        llm.with_structured_output(_Decision).invoke([HumanMessage(content="go")])
