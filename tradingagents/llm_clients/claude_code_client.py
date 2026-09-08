"""Claude Code CLI provider — run TradingAgents on a Claude subscription, no API key.

Every other provider in this package talks to an HTTP endpoint that needs a
key. This one shells out to the locally installed ``claude`` binary in
non-interactive print mode (``claude -p --output-format json``), so the model
calls are authenticated by whatever session ``claude login`` already
established. Nothing here ever reads ``ANTHROPIC_API_KEY``; by default the
child process is explicitly stripped of it so a stale key in ``.env`` cannot
silently redirect the run onto pay-per-token API billing (set
``TRADINGAGENTS_CLAUDE_CODE_API_KEY_AUTH=true`` to opt back in).

Two pieces of the Claude Code contract do the heavy lifting:

* ``--json-schema`` gives native structured output, which is what
  ``with_structured_output`` binds for the Research Manager / Trader /
  Portfolio Manager.
* ``--tools ""`` removes Claude Code's own built-in tools (Bash, Edit, Read,
  ...). The agents' LangChain tools are *not* handed to the CLI — the graph
  still executes them itself in ``ToolNode``. Instead the tool schemas are
  described in the system prompt and the model answers with a fenced
  ``tool_calls`` block, which is parsed back into ``AIMessage.tool_calls``.
  Keeping tool execution inside the LangGraph process is what preserves
  checkpoint/resume, the vendor routing layer, and the CLI's tool statistics.

The CLI is stateless here: ``--no-session-persistence`` is passed and the full
conversation is re-rendered into the prompt on every call, so a node can be
retried or resumed without a dangling session on disk.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)

# An alias resolves to the current model in that family, so a Claude Code
# upgrade picks up new models without a catalog edit here.
DEFAULT_MODEL = "sonnet"

# Env vars that would make the CLI authenticate as an API customer instead of
# using the logged-in subscription. Removed from the child environment unless
# the user opts in.
_API_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_API_KEY_AUTH_ENV = "TRADINGAGENTS_CLAUDE_CODE_API_KEY_AUTH"
_CLI_PATH_ENV = "TRADINGAGENTS_CLAUDE_CODE_CLI"
_WORKSPACE_ENV = "TRADINGAGENTS_CLAUDE_CODE_WORKSPACE"

# Claude Code describes its own runtime to the model — wall-clock date, working
# directory, machine — and it does so even when --system-prompt replaces the
# built-in prompt. For a run analysing a past trade date that real date is a
# look-ahead leak: the model will compute "the last 4 months" from today rather
# than from the analysis date. Every system prompt is therefore prefixed with
# this override, which reasserts the transcript as the only source of truth.
_ENVIRONMENT_OVERRIDE = """\
You are being called as a plain language model from inside another program.
Ignore every environment detail your harness reports to you — the wall-clock
date, the working directory, the machine, any repository or file state. Those
describe the process making the call, not the task, and none of them are
evidence about the subject you are analysing.

The conversation below is the only source of truth. When it states an analysis
or trade date, treat that date as "now" for every calculation, comparison, and
date range, even when it lies in the past. Never substitute today's real date,
and never use knowledge of what happened after the stated date.

"""

# Where the CLI puts itself. Probed in order when the PATH lookup comes up
# empty, which is what happens whenever the run is launched from a GUI app, an
# IDE run configuration or a launchd/cron job: those inherit a minimal PATH
# that holds neither the native installer's directory nor npm's global bin, so
# an installed CLI would otherwise look missing.
_KNOWN_CLI_PATHS = (
    "~/.local/bin/claude",       # native installer
    "~/.claude/local/claude",    # older local install
    "/usr/local/bin/claude",     # npm global: system node, Intel homebrew
    "/opt/homebrew/bin/claude",  # npm global: Apple-silicon homebrew
    "~/.npm-global/bin/claude",
    "~/.volta/bin/claude",
    "~/.bun/bin/claude",
)

# Version-managed node installs, newest first.
_KNOWN_CLI_GLOBS = ("~/.nvm/versions/node/*/bin/claude",)

_BOOL_TRUE = ("true", "1", "yes", "on")

INSTALL_HINT = (
    "The Claude Code CLI was not found. Install it with "
    "`npm install -g @anthropic-ai/claude-code` (or see "
    "https://code.claude.com/docs), then run `claude` once and sign in. "
    "Set TRADINGAGENTS_CLAUDE_CODE_CLI to point at the binary if it lives "
    "outside your PATH."
)

LOGIN_HINT = (
    "Claude Code rejected the request as unauthenticated. Run `claude` in a "
    "terminal and sign in with your Claude account, then retry."
)


class ClaudeCodeError(RuntimeError):
    """The ``claude`` CLI failed, or returned something we could not use."""


def _executable(path: str) -> str | None:
    """Expand ``~`` in `path` and return it if it's a runnable file."""
    expanded = os.path.expanduser(path)
    return expanded if os.path.isfile(expanded) and os.access(expanded, os.X_OK) else None


def find_cli(cli_path: str | None = None) -> str | None:
    """Resolve the ``claude`` executable, or None when it isn't installed.

    Accepts a bare command name (looked up on PATH) or a path, ``~`` included.
    A path given explicitly — argument or TRADINGAGENTS_CLAUDE_CODE_CLI — is
    honoured as written and nothing else is tried, so a deliberately chosen
    binary never quietly resolves to a different install. Only the default
    lookup falls back to the known install locations.
    """
    explicit = cli_path or os.environ.get(_CLI_PATH_ENV) or None
    candidate = os.path.expanduser(explicit or "claude")

    if resolved := shutil.which(candidate):
        return resolved
    if resolved := _executable(candidate):
        return resolved
    if explicit:
        return None

    for path in _KNOWN_CLI_PATHS:
        if resolved := _executable(path):
            return resolved
    for pattern in _KNOWN_CLI_GLOBS:
        for path in sorted(glob.glob(os.path.expanduser(pattern)), reverse=True):
            if resolved := _executable(path):
                return resolved
    return None


def default_workspace_dir() -> str:
    """A neutral, empty directory to run the CLI from.

    Claude Code reports its working directory to the model. Running from the
    user's repo would put unrelated project state in front of every analyst, so
    runs happen in a dedicated empty directory instead.
    """
    base = os.path.expanduser(
        os.environ.get(_WORKSPACE_ENV)
        or os.path.join("~", ".tradingagents", "claude_code_workspace")
    )
    os.makedirs(base, exist_ok=True)
    return base


def cli_version(cli_path: str | None = None) -> str | None:
    """Return the CLI's reported version, or None if it can't be run."""
    resolved = find_cli(cli_path)
    if resolved is None:
        return None
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


# --------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------

_TOOL_BLOCK_RE = re.compile(r"```(?:tool_calls|json)[ \t]*\r?\n(.*?)```", re.DOTALL)

_TOOL_PROTOCOL_HEADER = """

# Tool calling protocol

The tools below run outside this session. You do not execute them: you emit a
call, the caller runs it, and the output comes back on the next turn as a
"## Tool result" section.

To call one or more tools, reply with ONLY this fenced block and nothing else:

```tool_calls
[{"name": "<tool name>", "arguments": {<arguments object>}}]
```

Rules:
- Use only the tools listed below, spelled exactly as given, and make
  `arguments` match that tool's JSON schema exactly.
- Put several entries in the array to request several tools in one turn.
- A turn is either tool calls or an answer, never both. Do not add prose,
  explanations, or extra fences around the block.
- Once you have the information you need, reply with your final answer as
  ordinary text or markdown and no `tool_calls` block.

## Available tools
"""


def _message_text(content: Any) -> str:
    """Flatten LangChain message content (string or typed blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Render an assistant turn's tool calls back into the wire protocol."""
    payload = [
        {"name": call.get("name", ""), "arguments": call.get("args", {})}
        for call in tool_calls
    ]
    return "```tool_calls\n" + json.dumps(payload, default=str) + "\n```"


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    """Names of the bound tools, from their OpenAI tool dicts."""
    return {
        name
        for tool in tools
        if isinstance(name := tool.get("function", tool).get("name"), str)
    }


def _render_tool_catalog(tools: list[dict[str, Any]]) -> str:
    """Describe the bound tools (OpenAI tool dicts) for the system prompt."""
    lines = []
    for tool in tools:
        spec = tool.get("function", tool)
        name = spec.get("name", "")
        description = (spec.get("description") or "").strip()
        parameters = spec.get("parameters") or {"type": "object", "properties": {}}
        lines.append(f"\n### {name}\n{description}")
        lines.append(f"Arguments JSON schema: {json.dumps(parameters, default=str)}")
    return "\n".join(lines)


def render_prompt(messages: list[BaseMessage]) -> tuple[str, str]:
    """Split a message list into (system prompt, transcript sent on stdin).

    The CLI takes one system prompt and one prompt body, so the conversation is
    replayed as a labelled transcript. Labels match the ones the tool protocol
    tells the model to expect.
    """
    system_parts: list[str] = []
    turns: list[str] = []

    for message in messages:
        text = _message_text(message.content)
        if isinstance(message, SystemMessage):
            if text:
                system_parts.append(text)
        elif isinstance(message, ToolMessage):
            label = message.name or "tool"
            turns.append(f"## Tool result: {label}\n{text}")
        elif isinstance(message, AIMessage):
            body = text
            if message.tool_calls:
                rendered = _render_tool_calls(message.tool_calls)
                body = f"{body}\n{rendered}" if body else rendered
            turns.append(f"## Assistant\n{body}")
        elif isinstance(message, HumanMessage):
            turns.append(f"## User\n{text}")
        else:
            turns.append(f"## {message.type.capitalize()}\n{text}")

    transcript = "\n\n".join(turns).strip()
    if not transcript:
        # The CLI needs a non-empty prompt; a system-only invocation still has
        # to say something to trigger a turn.
        transcript = "## User\nProceed."
    return "\n\n".join(system_parts).strip(), transcript


def parse_tool_calls(
    text: str,
    known_names: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract tool calls from a model reply.

    Returns ``(remaining_text, tool_calls)``. Any prose outside the block is
    kept as the message content so a model that explains itself first still
    produces a usable call instead of a parse failure.

    ``known_names`` restricts what counts as a call. A report can legitimately
    contain a ``json`` fence holding an array of named objects; without the
    bound tool names to check against, that would be misread as a tool call and
    the report would be lost.
    """
    candidates: list[tuple[str, str]] = []  # (whole match, inner json)
    for match in _TOOL_BLOCK_RE.finditer(text):
        candidates.append((match.group(0), match.group(1)))

    stripped = text.strip()
    if not candidates and stripped.startswith("["):
        candidates.append((text, stripped))

    calls: list[dict[str, Any]] = []
    remaining = text
    for whole, inner in candidates:
        try:
            parsed = json.loads(inner)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            continue
        block_calls = [
            {
                "name": entry["name"],
                "args": entry.get("arguments") or entry.get("args") or {},
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "tool_call",
            }
            for entry in parsed
            if isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and (known_names is None or entry["name"] in known_names)
        ]
        if block_calls:
            calls.extend(block_calls)
            remaining = remaining.replace(whole, "", 1)

    return (remaining.strip() if calls else text), calls


# --------------------------------------------------------------------------
# Chat model
# --------------------------------------------------------------------------


class ChatClaudeCode(BaseChatModel):
    """LangChain chat model backed by the local ``claude`` CLI.

    Unsupported knobs are accepted and ignored rather than rejected, because
    ``TradingAgentsGraph`` forwards a common kwarg set to every provider:
    ``temperature`` and ``stop`` have no equivalent in Claude Code's print mode.
    """

    model: str = DEFAULT_MODEL
    cli_path: str | None = None
    effort: str | None = None
    max_tokens: int | None = None
    max_retries: int = 2
    timeout: float = 900.0
    temperature: float | None = None
    # Disables CLAUDE.md discovery, skills, plugins, hooks and MCP servers so a
    # user's personal Claude Code setup can't leak into an analysis run.
    safe_mode: bool = True
    # Working directory for the child process. None resolves to a dedicated
    # empty directory (see default_workspace_dir) so the caller's repo state
    # never reaches the model.
    workspace_dir: str | None = None
    max_budget_usd: float | None = None
    extra_cli_args: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort}

    # -- request construction ------------------------------------------------

    def _resolved_cli(self) -> str:
        resolved = find_cli(self.cli_path)
        if resolved is None:
            raise ClaudeCodeError(INSTALL_HINT)
        return resolved

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if os.environ.get(_API_KEY_AUTH_ENV, "").strip().lower() not in _BOOL_TRUE:
            for name in _API_KEY_ENV_VARS:
                env.pop(name, None)
        else:
            # An empty key is worse than no key: the CLI treats it as a
            # configured credential and fails auth instead of falling through.
            for name in _API_KEY_ENV_VARS:
                if not env.get(name, "").strip():
                    env.pop(name, None)
        if self.max_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.max_tokens)
        return env

    def _build_argv(self, system_prompt: str, response_schema: dict | None) -> list[str]:
        argv = [
            self._resolved_cli(),
            "--print",
            "--output-format", "json",
            "--model", self.model,
            # No Claude Code built-in tools: the agents' tools run in-process.
            "--tools", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--permission-prompts", "none",
            "--system-prompt", system_prompt,
        ]
        if self.safe_mode:
            argv.append("--safe-mode")
        if self.effort:
            argv += ["--effort", self.effort]
        if self.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(self.max_budget_usd)]
        if response_schema is not None:
            argv += ["--json-schema", json.dumps(response_schema, default=str)]
        argv += list(self.extra_cli_args)
        return argv

    def _run_cli(self, argv: list[str], prompt: str) -> dict[str, Any]:
        """Invoke the CLI once and return its parsed JSON result envelope."""
        try:
            completed = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.workspace_dir or default_workspace_dir(),
                env=self._child_env(),
                check=False,
            )
        except FileNotFoundError as exc:
            raise ClaudeCodeError(INSTALL_HINT) from exc
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeError(
                f"Claude Code did not respond within {self.timeout:.0f}s. Raise the "
                f"timeout, or lower the effort level, for very long analyst turns."
            ) from exc

        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0:
            raise ClaudeCodeError(_describe_failure(completed.returncode, stderr))

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCodeError(
                "Could not parse the Claude Code JSON envelope. "
                f"stdout: {completed.stdout[:500]!r} stderr: {stderr[:500]!r}"
            ) from exc

        if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
            raise ClaudeCodeError(
                f"Claude Code reported an error ({payload.get('subtype')}): "
                f"{payload.get('result') or payload.get('api_error_status') or stderr}"
            )
        return payload

    def _call_with_retries(self, argv: list[str], prompt: str) -> dict[str, Any]:
        attempts = max(1, self.max_retries + 1)
        last: ClaudeCodeError | None = None
        for attempt in range(attempts):
            try:
                return self._run_cli(argv, prompt)
            except ClaudeCodeError as exc:
                message = str(exc)
                if INSTALL_HINT in message or LOGIN_HINT in message:
                    raise  # Not transient: retrying can only waste time.
                last = exc
                logger.warning(
                    "Claude Code call failed (attempt %d/%d): %s",
                    attempt + 1, attempts, exc,
                )
        raise last  # type: ignore[misc]

    # -- LangChain interface -------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools: list[dict[str, Any]] = kwargs.get("tools") or []
        response_schema: dict | None = kwargs.get("response_schema")

        system_prompt, prompt = render_prompt(messages)
        system_prompt = _ENVIRONMENT_OVERRIDE + system_prompt
        if tools:
            system_prompt += _TOOL_PROTOCOL_HEADER + _render_tool_catalog(tools)

        payload = self._call_with_retries(
            self._build_argv(system_prompt, response_schema), prompt
        )
        text = payload.get("result") or ""

        additional_kwargs: dict[str, Any] = {}
        tool_calls: list[dict[str, Any]] = []
        if response_schema is not None:
            additional_kwargs["structured_output"] = payload.get("structured_output")
        elif tools:
            text, tool_calls = parse_tool_calls(text, _tool_names(tools))

        message = AIMessage(
            content=text,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            usage_metadata=_usage_metadata(payload.get("usage") or {}),
            response_metadata={
                "model": self.model,
                "session_id": payload.get("session_id"),
                "stop_reason": payload.get("stop_reason"),
                "total_cost_usd": payload.get("total_cost_usd"),
                "num_turns": payload.get("num_turns"),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any):
        """Bind tools by describing them in the system prompt.

        ``tool_choice`` has no equivalent in the CLI protocol and is ignored;
        the prompt already instructs the model to answer with tool calls until
        it has what it needs.
        """
        if tool_choice is not None:
            logger.debug("claude_code: tool_choice=%r is not supported; ignoring", tool_choice)
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, **kwargs)

    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Use the CLI's native ``--json-schema`` validation for typed output."""
        json_schema, parse = _schema_and_parser(schema)
        bound = self.bind(response_schema=json_schema)
        if include_raw:
            return bound | RunnableLambda(
                lambda message: {
                    "raw": message,
                    "parsed": parse(message),
                    "parsing_error": None,
                }
            )
        return bound | RunnableLambda(parse)


def _describe_failure(returncode: int, stderr: str) -> str:
    lowered = stderr.lower()
    if any(
        marker in lowered
        for marker in ("not logged in", "authentication", "unauthorized", "please run /login")
    ):
        return f"{LOGIN_HINT} (exit {returncode}: {stderr[:300]})"
    return f"Claude Code exited with status {returncode}: {stderr[:800] or '<no stderr>'}"


def _usage_metadata(usage: dict[str, Any]) -> dict[str, int]:
    """Map the CLI's usage block onto LangChain's usage_metadata shape.

    Cached tokens are folded into the input count so the CLI's token counter
    reflects everything that was actually sent.
    """
    input_tokens = (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _schema_and_parser(schema: Any):
    """Return (JSON schema for --json-schema, parser for the reply)."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()

        def parse(message: AIMessage):
            return schema.model_validate(_structured_payload(message))

        return json_schema, parse

    if isinstance(schema, dict):
        json_schema = schema.get("schema", schema) if "schema" in schema else schema
        return json_schema, _structured_payload

    raise NotImplementedError(
        f"claude_code structured output needs a pydantic model or a JSON schema dict, "
        f"got {type(schema).__name__}"
    )


def _structured_payload(message: AIMessage) -> dict[str, Any]:
    """Pull the validated object out of a reply, or raise so callers fall back."""
    payload = message.additional_kwargs.get("structured_output")
    if payload is None and message.content:
        try:
            payload = json.loads(_message_text(message.content))
        except json.JSONDecodeError:
            payload = None
    if not isinstance(payload, dict):
        raise ValueError("Claude Code returned no structured output for the bound schema")
    return payload


# --------------------------------------------------------------------------
# Provider client
# --------------------------------------------------------------------------

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "max_tokens", "callbacks", "effort",
    "temperature", "safe_mode", "workspace_dir", "max_budget_usd",
    "extra_cli_args", "cli_path",
)


class ClaudeCodeClient(BaseLLMClient):
    """Client for the locally installed Claude Code CLI (subscription auth)."""

    provider = "claude_code"

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        if self.base_url:
            # The CLI owns its own endpoint/auth; a backend URL here is a
            # leftover from another provider's menu selection.
            logger.debug("claude_code: ignoring backend_url=%r", self.base_url)

        llm_kwargs: dict[str, Any] = {"model": self.model or DEFAULT_MODEL}
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs and self.kwargs[key] is not None:
                llm_kwargs[key] = self.kwargs[key]

        if llm_kwargs.get("temperature") is not None:
            logger.warning(
                "claude_code: the Claude Code CLI has no temperature control; "
                "the configured temperature is ignored."
            )

        return ChatClaudeCode(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("claude_code", self.model)
