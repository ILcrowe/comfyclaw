"""
AgentBackend — abstract interface for the agent's tool-use loop.

Each backend takes a system prompt, a user message, a tool schema, and a
``dispatch`` callable that maps tool name + args to a result string.  The
backend is responsible for running the multi-round conversation until the
LLM either signals "done" (e.g. ``finalize_workflow``) or hits the round
budget.

Backends emit lightweight events through ``on_event`` so the UI can show
thinking / tool-call / tool-result entries in real time.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """One tool invocation issued by the LLM.

    Attributes
    ----------
    name     : Tool name as declared in the tool schema.
    args     : Parsed JSON arguments (always a dict, possibly empty).
    call_id  : Opaque round-trip id the backend uses to match the reply.
               Different providers use different id formats; the harness
               never inspects this — it just round-trips it back.
    """

    name: str
    args: dict
    call_id: str = ""


# Type aliases for callback signatures
DispatchFn = Callable[[ToolCall], "tuple[str, bool]"]
"""Dispatcher: ``(call) -> (result_text, should_stop)``.

``should_stop`` is ``True`` when the call corresponds to a terminal tool
such as ``finalize_workflow`` — the backend should exit the loop after
delivering the result back to the model.
"""

EventFn = Callable[[str, str, str, "dict | None"], None]
"""Event emitter: ``(event_type, content, tool_name, tool_args_or_None)``."""


@runtime_checkable
class AgentBackend(Protocol):
    """Pluggable driver for the tool-use loop."""

    #: Stable lowercase identifier used in HarnessConfig.agent_backend.
    name: str

    def is_available(self) -> bool:
        """Return True if this backend can run in the current environment.

        For LiteLLM this is always True (any provider with an API key
        works).  For CLI backends this checks that the required binary
        is on ``$PATH``.
        """
        ...

    def run_tool_loop(
        self,
        system: str,
        user: str,
        tools: list[dict],
        dispatch: DispatchFn,
        on_event: EventFn | None = None,
        max_rounds: int = 40,
    ) -> str:
        """Run the LLM tool-use conversation to completion.

        Parameters
        ----------
        system     : System prompt string.
        user       : User message string.
        tools      : Tool schema in OpenAI function-calling format.
        dispatch   : Callable that executes a tool call and returns
                     ``(result_text, should_stop)``.
        on_event   : Optional listener for thinking / tool-call /
                     tool-result events.
        max_rounds : Hard cap on tool-call rounds.

        Returns
        -------
        Final rationale string emitted by the model (typically the
        ``rationale`` argument of ``finalize_workflow``, or any trailing
        assistant text).
        """
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(
    name: str,
    *,
    model: str = "",
    api_key: str = "",
    api_base: str | None = None,
    extra: dict[str, Any] | None = None,
) -> AgentBackend:
    """Return a backend instance by name.

    Falls back to LiteLLM with a **prominent** printed warning if the
    requested CLI backend is unavailable.  We deliberately distinguish two
    failure modes so the user sees the real reason in the server log:

    * Binary missing  → "CLI not installed".
    * Binary present but not signed in → "CLI installed but not signed in
      — falling back to litellm (which needs an API key)."

    The latter case is the most common source of "why is it asking for my
    API key?" confusion from teammates who installed e.g. ``codex`` but
    never ran ``codex login``.
    """
    name = (name or "litellm").strip().lower().replace("_", "-")
    extra = extra or {}

    if name == "litellm":
        from .litellm_backend import LiteLLMBackend

        return LiteLLMBackend(model=model, api_key=api_key, api_base=api_base)

    def _check_cli_auth(canonical: str) -> None:
        """Log a clear warning when the CLI is installed but not signed in.

        We *do* return the CLI backend in that case (the user explicitly
        asked for it), but the agent's first call will fail with a cryptic
        ``rc=1`` from the CLI — so we log the actual reason up-front in the
        server log to short-circuit the inevitable "why is this asking for
        an API key?" debugging session.
        """
        statuses = {s.name: s for s in probe_all()}
        st = statuses.get(canonical)
        if not st or st.state == "ok":
            return
        if st.state == "needs_auth":
            print(
                f"[agent_backends] '{canonical}' CLI is installed but not "
                f"signed in ({st.detail}). The next agent call will fail "
                f"until you complete the sign-in flow from the panel "
                f"(or run the CLI's `login` command manually). No fallback "
                f"to litellm — you picked this backend on purpose."
            )

    def _missing_warn(canonical: str) -> None:
        """Log the fallback-to-litellm case (binary actually missing)."""
        statuses = {s.name: s for s in probe_all()}
        st = statuses.get(canonical)
        if not st:
            return
        print(
            f"[agent_backends] '{canonical}' CLI not found on PATH "
            f"({st.detail}). Falling back to litellm (API key required)."
        )

    if name in ("claude-code", "claude"):
        from .claude_code_backend import ClaudeCodeBackend

        be = ClaudeCodeBackend(
            model=model or extra.get("model", ""),
            session_key=str(extra.get("session_id") or ""),
        )
        if be.is_available():
            _check_cli_auth("claude-code")
            return be
        _missing_warn("claude-code")
    elif name in ("codex", "openai-codex"):
        from .codex_backend import CodexBackend

        be = CodexBackend(
            model=model or extra.get("model", ""),
            session_key=str(extra.get("session_id") or ""),
        )
        if be.is_available():
            _check_cli_auth("codex")
            return be
        _missing_warn("codex")
    elif name in ("gemini-cli", "gemini"):
        from .gemini_backend import GeminiCLIBackend

        be = GeminiCLIBackend(
            model=model or extra.get("model", ""),
            session_key=str(extra.get("session_id") or ""),
        )
        if be.is_available():
            _check_cli_auth("gemini-cli")
            return be
        _missing_warn("gemini-cli")
    elif name in ("grok-cli", "grok"):
        from .grok_backend import GrokCLIBackend

        be = GrokCLIBackend(
            model=model or extra.get("model", ""),
            session_key=str(extra.get("session_id") or ""),
        )
        if be.is_available():
            _check_cli_auth("grok-cli")
            return be
        raise RuntimeError("Grok Build CLI is not installed. Install it, then run `grok login`.")
    else:
        print(f"[agent_backends] Unknown backend {name!r} — falling back to litellm.")

    from .litellm_backend import LiteLLMBackend

    return LiteLLMBackend(model=model, api_key=api_key, api_base=api_base)


# ---------------------------------------------------------------------------
# Availability probes (used by sync_server to populate the panel UI)
# ---------------------------------------------------------------------------


BackendState = Literal["ok", "needs_install", "needs_auth", "error", "unsupported"]


@dataclass
class BackendStatus:
    name: str
    # ``available`` is kept for backward compatibility with older panels.
    # New panels read ``state`` and treat anything other than ``"ok"`` as
    # requiring user action.
    available: bool
    state: BackendState = "ok"
    binary_path: str = ""
    auth_method: str = ""  # e.g. "claudeai" | "console" | ""
    detail: str = ""
    # When True, the panel may surface an "Install" affordance for this
    # backend (currently only claude-code).
    can_install: bool = False


# Search list for the `claude` binary when it isn't on $PATH. Order matters —
# we prefer locations under the user's home so we don't accidentally pick a
# system-wide install we can't refresh later.
_CLAUDE_PATH_HINTS: tuple[str, ...] = (
    "~/.local/share/fnm/node-versions/*/installation/bin/claude",
    "~/.local/bin/claude",
    "~/.npm-global/bin/claude",
    "~/.nvm/versions/node/*/bin/claude",
    "~/.volta/bin/claude",
    "/opt/local/bin/claude",
    "/usr/local/bin/claude",
)


def _resolve_claude_bin() -> str:
    """Find a usable `claude` binary even if it isn't on $PATH.

    Resolution order:
      1. ``$COMFYCLAW_CLAUDE_BIN`` if set and executable.
      2. ``shutil.which("claude")``.
      3. Common install locations (fnm, nvm, ~/.local/bin, /usr/local/bin, …).
    """
    override = os.environ.get("COMFYCLAW_CLAUDE_BIN", "").strip()
    if override and os.access(override, os.X_OK):
        return override

    via_path = shutil.which("claude")
    if via_path:
        return via_path

    for pattern in _CLAUDE_PATH_HINTS:
        for candidate in glob.glob(os.path.expanduser(pattern)):
            if os.access(candidate, os.X_OK):
                return candidate
    return ""


_CLAUDE_EXTERNAL_API_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "CLAUDE_CODE_API_KEY",
)


def _env_with_claude_path(binary: str, *, scrub_external_api_keys: bool = True) -> dict[str, str]:
    """Return an env dict where the claude binary's directory is on PATH.

    Necessary because the Claude Code CLI is a node script (``#!/usr/bin/env
    node``); if ``node`` itself isn't on PATH the kernel returns exit 127
    with ``/usr/bin/env: 'node': No such file or directory`` *before*
    the script can even start.  We co-locate node + claude in the same
    bin dir under fnm/nvm/npm-global, so prepending that dir to PATH is
    enough to make both reachable to the kernel and to the script.

    ComfyClaw's ``claude-code`` backend is subscription-login based.
    If the parent ComfyUI process has a stale ``ANTHROPIC_API_KEY`` or
    related external API override, Claude Code prefers it over the local
    login and fails with "Invalid API key · Fix external API key".  Strip
    those variables by default so the CLI uses ``claude auth login`` state.
    """
    env = {**os.environ}
    if scrub_external_api_keys:
        for key in _CLAUDE_EXTERNAL_API_ENV:
            env.pop(key, None)
    binary_dir = os.path.dirname(binary) if binary else ""
    if binary_dir:
        existing = env.get("PATH", "")
        if binary_dir not in existing.split(os.pathsep):
            env["PATH"] = binary_dir + (os.pathsep + existing if existing else "")
    return env


def _probe_claude_auth(binary: str) -> tuple[BackendState, str, str]:
    """Run ``claude auth status --json`` and parse the result.

    Returns ``(state, auth_method, detail)`` where ``state`` is one of
    ``"ok"`` / ``"needs_auth"`` / ``"error"``.
    """
    try:
        proc = subprocess.run(
            [binary, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            env=_env_with_claude_path(binary),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return "error", "", f"auth probe failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "error", "", f"auth probe failed: {exc}"

    raw = (proc.stdout or "").strip() or (proc.stderr or "").strip()

    # Exit 127 with no/garbled stdout usually means the shebang interpreter
    # (node) isn't on PATH.  Surface that as a real error rather than
    # silently "needs_auth" so the panel can show a useful message.
    if proc.returncode == 127:
        combined = ((proc.stderr or "") + " " + (proc.stdout or "")).lower()
        if "node" in combined and ("no such file" in combined or "not found" in combined):
            return (
                "error",
                "",
                "Claude Code is installed but its `node` runtime can't be "
                "found on PATH. Add the binary's directory to PATH.",
            )

    if not raw:
        # Non-zero with no output: treat as unauthenticated rather than error
        # so the panel still surfaces a Sign-in affordance.
        if proc.returncode != 0:
            return "needs_auth", "", "Not signed in"
        return "error", "", "auth probe returned no output"

    # Strip ANSI codes some CLI builds emit even with --json.
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        if "logged in" in raw.lower() or "loggedin" in raw.lower():
            return "ok", "", ""
        return "needs_auth", "", "Not signed in"

    logged_in = bool(payload.get("loggedIn"))
    auth_method = str(payload.get("authMethod") or "")
    if logged_in:
        detail = (
            f"Signed in via {auth_method}" if auth_method and auth_method != "none" else "Signed in"
        )
        return "ok", auth_method, detail
    return "needs_auth", auth_method, "Not signed in"


def _probe_codex_auth(binary: str) -> tuple[BackendState, str, str]:
    """Run ``codex login status`` and parse the result.

    Returns ``(state, auth_method, detail)`` where ``auth_method`` is one of:

    * ``"chatgpt"``  — signed in with ChatGPT (Pro/Team/Enterprise subscription),
    * ``"apikey"``   — `~/.codex/auth.json` holds an ``OPENAI_API_KEY``,
    * ``""``         — not signed in / unknown.

    The Codex CLI exits 0 + prints ``Logged in using ChatGPT`` (or ``Logged in
    using API key``) when authenticated, and exits 1 + prints ``Not logged in``
    otherwise.
    """
    try:
        proc = subprocess.run(
            [binary, "login", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return "error", "", f"auth probe failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "error", "", f"auth probe failed: {exc}"

    raw = ((proc.stdout or "") + " " + (proc.stderr or "")).strip()
    low = raw.lower()

    if proc.returncode == 0 and "logged in" in low:
        if "chatgpt" in low:
            return "ok", "chatgpt", "Signed in with ChatGPT"
        if "api key" in low or "apikey" in low:
            return "ok", "apikey", "Signed in with API key"
        return "ok", "", raw[:160] or "Signed in"

    # rc != 0 OR "not logged in" prose -> needs auth.
    if "not logged in" in low or proc.returncode != 0:
        return "needs_auth", "", "Not signed in"
    return "error", "", raw[:160] or "auth probe returned unexpected output"


def _probe_gemini_auth(binary: str) -> tuple[BackendState, str, str]:
    """Detect whether the user can talk to Gemini without an API key.

    The Gemini CLI does **not** expose an ``auth status`` subcommand (its
    OAuth flow is purely interactive on first run), so we use a file-and-env
    heuristic that matches what dr-claw's ``gemini-api.js`` checks:

    * ``$GEMINI_API_KEY`` / ``$GOOGLE_API_KEY`` set → API-key auth available;
    * ``~/.gemini/oauth_creds.json`` present and non-empty → OAuth credentials
      cached by a previous ``gemini`` invocation.

    Returns ``(state, auth_method, detail)``.
    """
    # 1. API-key path (still counts as "ok" — user can hit Gemini even without
    #    a subscription).
    for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(env_var, "").strip():
            return "ok", "apikey", f"Using ${env_var} (API)"

    # 2. OAuth credentials cached by an earlier `gemini` run.
    creds_path = os.path.expanduser("~/.gemini/oauth_creds.json")
    try:
        st = os.stat(creds_path)
    except FileNotFoundError:
        return (
            "needs_auth",
            "",
            "Run `gemini` once in a terminal and sign in with your Google account",
        )
    except OSError as exc:
        return "error", "", f"oauth_creds.json unreadable: {exc}"

    if st.st_size <= 2:  # empty `{}` or empty file
        return "needs_auth", "", "Gemini oauth_creds.json is empty"

    return "ok", "oauth", "Signed in via Google OAuth"


def probe_all() -> list[BackendStatus]:
    """Return availability for every registered backend.

    Suitable for sending to the panel as the ``agent_backends`` WS message
    so the user can see which CLIs are installed and authenticated.
    """
    out: list[BackendStatus] = [
        BackendStatus(
            name="litellm",
            available=True,
            state="ok",
            detail="Always available",
        )
    ]

    # ── claude-code ──────────────────────────────────────────────────────────
    claude_bin = _resolve_claude_bin()
    if not claude_bin:
        out.append(
            BackendStatus(
                name="claude-code",
                available=False,
                state="needs_install",
                detail="Claude Code is not installed",
                can_install=True,
            )
        )
    else:
        state, auth_method, detail = _probe_claude_auth(claude_bin)
        out.append(
            BackendStatus(
                name="claude-code",
                available=(state == "ok"),
                state=state,
                binary_path=claude_bin,
                auth_method=auth_method,
                detail=detail or f"`claude` at {claude_bin}",
                can_install=True,
            )
        )

    # ── codex ────────────────────────────────────────────────────────────────
    codex_bin = shutil.which("codex") or ""
    # We can drive the install ourselves if either brew (mac) or npm (any
    # platform with Node.js) is on PATH — see ``setup_flows._INSTALL_COMMANDS``.
    codex_installer = bool(shutil.which("brew") or shutil.which("npm"))
    if not codex_bin:
        out.append(
            BackendStatus(
                name="codex",
                available=False,
                state="needs_install",
                detail=(
                    "Codex CLI not on PATH. Install with `brew install codex` "
                    "or `npm i -g @openai/codex`, then sign in with your "
                    "ChatGPT account."
                ),
                can_install=codex_installer,
            )
        )
    else:
        c_state, c_method, c_detail = _probe_codex_auth(codex_bin)
        out.append(
            BackendStatus(
                name="codex",
                available=(c_state == "ok"),
                state=c_state,
                binary_path=codex_bin,
                auth_method=c_method,
                detail=c_detail or f"`codex` at {codex_bin}",
                can_install=codex_installer,
            )
        )

    # ── gemini-cli ───────────────────────────────────────────────────────────
    gemini_bin = shutil.which("gemini") or ""
    gemini_installer = bool(shutil.which("npm"))
    if not gemini_bin:
        out.append(
            BackendStatus(
                name="gemini-cli",
                available=False,
                state="needs_install",
                detail=(
                    "Gemini CLI not on PATH. Install with "
                    "`npm i -g @google/gemini-cli`, then run `gemini` once "
                    "and sign in with your Google account."
                ),
                can_install=gemini_installer,
            )
        )
    else:
        g_state, g_method, g_detail = _probe_gemini_auth(gemini_bin)
        out.append(
            BackendStatus(
                name="gemini-cli",
                available=(g_state == "ok"),
                state=g_state,
                binary_path=gemini_bin,
                auth_method=g_method,
                detail=g_detail or f"`gemini` at {gemini_bin}",
                can_install=gemini_installer,
            )
        )
    # ── grok-cli ─────────────────────────────────────────────────────────────
    from .grok_backend import _grok_bin, _guard_subscription_config, _oauth_cache_present

    grok_bin = shutil.which(_grok_bin()) or ""
    if not grok_bin:
        out.append(BackendStatus(
            name="grok-cli", available=False, state="needs_install",
            detail="Grok Build CLI is not installed. Install it, then run `grok login`.",
        ))
    else:
        try:
            _guard_subscription_config()
            signed_in = _oauth_cache_present()
            state: BackendState = "ok" if signed_in else "needs_auth"
            detail = (
                "Signed in with Grok OAuth (cached session; verified on first call)"
                if signed_in else "Grok CLI is installed but not authenticated. Run `grok login`."
            )
        except RuntimeError as exc:
            state, detail = "error", str(exc)
        out.append(BackendStatus(
            name="grok-cli", available=(state == "ok"), state=state,
            binary_path=grok_bin, auth_method="oauth" if state == "ok" else "",
            detail=detail,
        ))
    return out
