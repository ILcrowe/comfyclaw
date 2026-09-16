"""Grok Build CLI backend using the user's OAuth subscription session.

Grok emits text-only JSON envelopes. ComfyClaw parses and dispatches every
requested tool; Grok's native tools are excluded from the child process.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from pathlib import Path

from . import _stream_session
from .base import DispatchFn, EventFn

_GROK_SESSION_BY_KEY: dict[str, str] = {}
_GROK_SESSION_LOCK = threading.Lock()
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{5,127}$")
_API_ENV = ("XAI_API_KEY", "GROK_CODE_XAI_API_KEY", "GROK_DEPLOYMENT_KEY")
_ROUTING_ENV = (
    "GROK_MODELS_BASE_URL",
    "GROK_MODELS_LIST_URL",
    "GROK_CLI_CHAT_PROXY_BASE_URL",
)


def _grok_bin() -> str:
    return os.environ.get("COMFYCLAW_GROK_BIN", "").strip() or "grok"


def _subscription_only() -> bool:
    value = os.environ.get("COMFYCLAW_GROK_SUBSCRIPTION_ONLY", "true").strip().lower()
    if value not in {"true", "false"}:
        raise RuntimeError("COMFYCLAW_GROK_SUBSCRIPTION_ONLY must be true or false")
    if value != "true":
        raise RuntimeError(
            "grok-cli supports subscription OAuth only; set COMFYCLAW_GROK_SUBSCRIPTION_ONLY=true"
        )
    return True


def _grok_home() -> Path:
    return Path(os.environ.get("GROK_HOME") or Path.home() / ".grok")


def _oauth_cache_present() -> bool:
    """Check Grok's documented auth cache without exposing its credentials.

    A cached token can expire; the CLI remains the final authority at run time.
    An API-key-only cache never counts as subscription authentication.
    """
    try:
        payload = json.loads((_grok_home() / "auth.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not inspect Grok OAuth cache: {exc}") from exc
    if not isinstance(payload, dict):
        return False
    return any(
        key != "xai::api_key"
        and isinstance(value, dict)
        and isinstance(value.get("key"), str)
        and bool(value["key"])
        and ("x.ai" in key or "grok.com" in key)
        for key, value in payload.items()
    )


def _guard_subscription_config() -> None:
    """Fail closed when a Grok config could route around the OAuth session."""
    if not _subscription_only():
        return
    # Grok's model-specific api_key/env_key wins over OAuth. Reject these
    # rather than editing the user's config. The CLI-side auth lockdown below
    # is defense in depth, including the global API-key fallback.
    config = _grok_home() / "config.toml"
    try:
        content = config.read_text(encoding="utf-8") if config.exists() else ""
    except OSError as exc:
        raise RuntimeError(f"BLOCKED: Cannot inspect Grok config: {exc}") from exc
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        # Python 3.10 has no stdlib TOML parser. Any custom model becomes
        # ambiguous, so subscription-only operation fails closed.
        if re.search(r"(?m)^\s*\[(?:model\.|auth\]|grok_com_config\.|endpoints\])", content):
            raise RuntimeError("BLOCKED: Cannot verify Grok auth config on Python 3.10") from None
    else:
        try:
            config_data = tomllib.loads(content) if content else {}
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"BLOCKED: Cannot parse Grok config: {exc}") from exc
        models = config_data.get("model", {})
        for model in models.values():
            if isinstance(model, dict) and any(
                key in model
                for key in (
                    "api_key",
                    "env_key",
                    "base_url",
                    "extra_headers",
                    "env_http_headers",
                    "auth_provider",
                )
            ):
                raise RuntimeError("BLOCKED: API credential may override subscription auth")
        auth = config_data.get("auth", {})
        if auth.get("preferred_method") == "api_key" or auth.get("auth_provider_command"):
            raise RuntimeError("BLOCKED: Grok auth configuration may override subscription auth")
        if config_data.get("grok_com_config", {}).get("oidc"):
            raise RuntimeError("BLOCKED: Grok OIDC configuration may override subscription auth")
        if config_data.get("endpoints"):
            raise RuntimeError("BLOCKED: Grok endpoint configuration may reroute subscription auth")
    if any(os.environ.get(key) for key in ("GROK_CONFIG", "GROK_CONFIG_PATH")):
        raise RuntimeError("BLOCKED: Grok config overlay may override subscription auth")
    if any(
        os.environ.get(key)
        for key in ("GROK_AUTH_PROVIDER_COMMAND", "GROK_OIDC_ISSUER", "GROK_OIDC_CLIENT_ID")
    ):
        raise RuntimeError("BLOCKED: External Grok authentication may override subscription auth")
    if any(os.environ.get(key) for key in _ROUTING_ENV):
        raise RuntimeError("BLOCKED: Grok endpoint environment may reroute subscription auth")


def _grok_env() -> dict[str, str]:
    env = os.environ.copy()
    if _subscription_only():
        for key in _API_ENV:
            env.pop(key, None)
        # Official Grok Build auth lockdown. User config cannot turn it off.
        env["GROK_DISABLE_API_KEY_AUTH"] = "1"
    return env


def _get_recorded_grok_session(session_key: str) -> str:
    if not session_key:
        return ""
    with _GROK_SESSION_LOCK:
        return _GROK_SESSION_BY_KEY.get(session_key, "")


def _record_grok_session(session_key: str, grok_session_id: str) -> None:
    if not session_key or not grok_session_id:
        return
    with _GROK_SESSION_LOCK:
        _GROK_SESSION_BY_KEY[session_key] = grok_session_id


class GrokCLIBackend:
    name = "grok-cli"

    def __init__(self, model: str = "", session_key: str = "") -> None:
        self.model = model
        self.session_key = session_key
        self._bin = _grok_bin()

    def is_available(self) -> bool:
        return shutil.which(self._bin) is not None

    def run_tool_loop(
        self,
        system: str,
        user: str,
        tools: list[dict],
        dispatch: DispatchFn,
        on_event: EventFn | None = None,
        max_rounds: int = 40,
    ) -> str:
        _guard_subscription_config()
        if _subscription_only() and not _oauth_cache_present():
            raise RuntimeError("Grok CLI is installed but not authenticated. Run `grok login`.")

        grok_session_id = _get_recorded_grok_session(self.session_key)
        protocol = _stream_session.envelope_protocol_instructions(tools)
        rules = system + protocol
        env = _grok_env()
        first_invocation = True

        def _invoke(prompt: str) -> str:
            nonlocal grok_session_id, first_invocation
            if first_invocation:
                # --resume can retain a prior turn's tool catalog. The current
                # catalog must be visible in the new user turn as well.
                prompt = (
                    "Use only the exact ComfyClaw tool names listed below; "
                    "never infer a tool name from prior turns.\n"
                    + protocol
                    + "\n\n## Current request\n"
                    + prompt
                )
                first_invocation = False
            argv = [
                self._bin,
                "--no-auto-update",
                "-p",
                prompt,
                "--output-format",
                "json",
                "--cwd",
                str(_grok_home()),
                "--rules",
                rules,
                # An empty --tools value is not a reliable empty allowlist.
                # Start with one documented inert tool ID and remove it;
                # exclude the separately injected MCP discovery/execution tools.
                "--tools",
                "todo_write",
                "--no-subagents",
                "--disable-web-search",
                "--disallowed-tools",
                "todo_write,search_tool,use_tool,Agent",
                "--permission-mode",
                "dontAsk",
                "--deny",
                "Bash",
                "--deny",
                "Read",
                "--deny",
                "Edit",
                "--deny",
                "Write",
                "--deny",
                "Grep",
                "--deny",
                "WebFetch",
                "--deny",
                "MCPTool",
                "--max-turns",
                "1",
            ]
            if grok_session_id:
                argv.extend(("--resume", grok_session_id))
            # Saved panel model IDs can belong to LiteLLM. Let the signed-in
            # CLI select its actual subscription default.
            rc, stdout, stderr = _stream_session.run_cli_oneshot(
                argv, "", timeout=420, env=env, encoding="utf-8"
            )
            try:
                payload = json.loads(stdout)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Grok CLI returned invalid JSON (rc={rc}): {stderr[:300]}"
                ) from exc
            if rc != 0 or payload.get("type") == "error":
                message = payload.get("message") or stderr or f"Grok CLI rc={rc}"
                raise RuntimeError(str(message)[:500])
            if not isinstance(payload.get("text"), str):
                raise RuntimeError("Grok CLI response has no text")
            sid = payload.get("sessionId")
            if isinstance(sid, str) and _SESSION_ID.fullmatch(sid):
                grok_session_id = sid
                _record_grok_session(self.session_key, sid)
            elif not grok_session_id:
                raise RuntimeError("Grok CLI response has no resumable sessionId")
            return payload["text"]

        return _stream_session.run_envelope_loop(
            backend_name=self.name,
            invoke=_invoke,
            system=system,
            user=user,
            tools=tools,
            dispatch=dispatch,
            on_event=on_event,
            max_rounds=max_rounds,
            incremental_session=True,
            raise_on_error=True,
            start_message="Continuing Grok session" if grok_session_id else "Starting Grok session",
        )
