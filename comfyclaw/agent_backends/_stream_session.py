"""
_stream_session — reusable helper for CLI agents that don't natively
expose a tool-use protocol.

Strategy
--------
Several CLI agents (``codex``, ``gemini``) only expose a single text
output — they don't speak the Anthropic-style ``tool_use`` /
``tool_result`` block protocol.  To still drive our tool-use loop we
prompt the model to emit a strict JSON envelope on every turn, parse
that envelope ourselves, dispatch the tool calls locally, and feed the
result back in the next turn.

Envelope format we instruct the model to use::

    {
      "tool_calls": [
        {"name": "<tool>", "arguments": { ... }},
        ...
      ],
      "rationale": "<optional final rationale>",
      "done": false
    }

When ``done`` is true (or a terminal tool such as ``finalize_workflow``
returns ``should_stop=True``), the loop exits.

This module only provides the low-level subprocess plumbing — each
backend is responsible for crafting the exact CLI invocation and the
prompt that tells the model how to format its output.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

# ---------------------------------------------------------------------------
# JSON envelope helpers
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def extract_json_envelope(text: str) -> dict | None:
    """Return the first valid JSON object from *text*.

    Tolerates fenced code blocks (```json ... ```) and stray prose
    around the envelope.  Returns ``None`` if no JSON object can be
    parsed.
    """
    if not text:
        return None
    text = text.strip()

    # Try a fenced block first
    for m in _FENCE_RE.finditer(text):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue

    # Try the whole thing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Greedy: longest substring starting with `{` that parses
    starts = [i for i, c in enumerate(text) if c == "{"]
    for s in starts:
        for e in range(len(text), s, -1):
            chunk = text[s:e]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def envelope_protocol_instructions(tool_schema: list[dict]) -> str:
    """Return a system-prompt fragment that documents the envelope protocol.

    Backends append this to their system prompt so the model knows how
    to format tool calls when the underlying CLI doesn't speak the
    native Anthropic / OpenAI tool-use protocol.
    """
    schema_lines = []
    for t in tool_schema:
        fn = t.get("function", t)
        nm = fn.get("name", "?")
        desc = fn.get("description", "").strip().split("\n")[0]
        params = fn.get("parameters", {}).get("properties", {}) or {}
        plist = ", ".join(params.keys()) if params else "(no params)"
        schema_lines.append(f"  • {nm}({plist}) — {desc}")

    return (
        "\n\n## Tool-use protocol\n"
        "On EVERY response you MUST output exactly one JSON object — no prose,\n"
        "no markdown, no code fences. The shape is:\n\n"
        "{\n"
        '  "tool_calls": [\n'
        '    {"name": "<tool>", "arguments": { ... }},\n'
        "    ...\n"
        "  ],\n"
        '  "rationale": "<optional summary>",\n'
        '  "done": false\n'
        "}\n\n"
        "Set `done: true` when you have called `finalize_workflow` and have no\n"
        "further actions.  Multiple tool calls per turn are allowed and they\n"
        "execute in order; the next turn will include all their results.\n\n"
        "Available tools:\n" + "\n".join(schema_lines) + "\n"
    )


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------


def run_cli_oneshot(
    argv: list[str],
    stdin_text: str,
    timeout: float = 600.0,
    env: dict[str, str] | None = None,
    env_overrides: dict[str, str | None] | None = None,
    encoding: str | None = None,
) -> tuple[int, str, str]:
    """Run ``argv`` with ``stdin_text`` piped in, return ``(rc, stdout, stderr)``."""
    import os

    child_env = dict(env) if env is not None else os.environ.copy()
    if env_overrides:
        for key, value in env_overrides.items():
            if value is None:
                child_env.pop(key, None)
            else:
                child_env[key] = value
    try:
        proc = subprocess.run(
            argv,
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding=encoding,
            timeout=timeout,
            check=False,
            env=child_env,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + f"\n[timeout after {timeout}s]"
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except Exception as exc:  # noqa: BLE001 — surface any unexpected error
        return 1, "", str(exc)


def format_tool_results_for_next_turn(results: list[dict]) -> str:
    """Format dispatch results as a USER message for the next CLI turn.

    Each result dict has ``{name, arguments, result, error?}``.
    """
    lines = ["## Previous tool results"]
    for r in results:
        nm = r.get("name", "?")
        args = r.get("arguments", {})
        try:
            args_repr = json.dumps(args, ensure_ascii=False)[:300]
        except Exception:
            args_repr = "<unserializable>"
        if r.get("error"):
            lines.append(f"- {nm}({args_repr}) -> ERROR: {r['error']}")
        else:
            res = (r.get("result") or "").strip().replace("\n", " ")
            if len(res) > 600:
                res = res[:600] + "…"
            lines.append(f"- {nm}({args_repr}) -> {res}")
    lines.append(
        "\nNow output the next tool-call envelope, or set `done: true` if the workflow is complete."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generic envelope-loop driver
# ---------------------------------------------------------------------------


def run_envelope_loop(
    *,
    backend_name: str,
    invoke: Any,
    system: str,
    user: str,
    tools: list[dict],
    dispatch,  # DispatchFn
    on_event,  # EventFn | None
    max_rounds: int,
    protocol_in_system: bool = False,
    start_message: str | None = None,
    incremental_session: bool = False,
    raise_on_error: bool = False,
) -> str:
    """Drive a CLI agent that doesn't natively support tool-use.

    ``invoke(prompt: str) -> str`` is a callable supplied by each backend
    that runs one turn of the CLI and returns the model's raw text output.

    ``incremental_session``
        When True, the backend keeps conversation history itself (e.g. via the
        Claude CLI's ``--resume``), so each round sends only the *newest* turn
        — the initial user message, then just the tool results — instead of
        replaying the whole transcript every round.  When False (default) the
        loop is stateless and re-sends the full transcript each round, which is
        what backends without native session resume require.
    """
    full_system = system if protocol_in_system else system + envelope_protocol_instructions(tools)
    convo: list[str] = [
        f"<<SYSTEM>>\n{full_system}\n<<END SYSTEM>>",
        f"<<USER>>\n{user}\n<<END USER>>",
    ]
    # In incremental mode the very first turn carries the system prompt (via the
    # backend's own flags) plus the user message; later turns carry only the
    # fresh tool results.
    pending_turn = user
    rationale = "(no rationale provided)"

    if on_event:
        on_event("info", start_message or f"Starting {backend_name} turn", "", None)

    for _round_idx in range(1, max_rounds + 1):
        prompt = pending_turn if incremental_session else "\n\n".join(convo)
        try:
            raw = invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            if on_event:
                on_event("error", f"{backend_name} CLI failed: {exc}", "", None)
            print(f"[{backend_name}] CLI failed: {exc}", file=sys.stderr)
            if raise_on_error:
                raise
            break

        env = extract_json_envelope(raw)
        if env is None:
            if on_event:
                on_event(
                    "error",
                    f"Could not parse JSON envelope from {backend_name}; raw: {raw[:200]!r}",
                    "",
                    None,
                )
            print(
                f"[{backend_name}] Could not parse envelope, raw output: {raw[:500]}",
                file=sys.stderr,
            )
            if raise_on_error:
                raise RuntimeError(f"Could not parse JSON envelope from {backend_name}")
            break

        if env.get("rationale"):
            rationale = str(env["rationale"]) or rationale
            if on_event:
                on_event("thinking", str(env["rationale"]), "", None)

        tool_calls = env.get("tool_calls") or []
        results: list[dict] = []
        done_this_round = bool(env.get("done"))

        for tc in tool_calls:
            name = str(tc.get("name", "")).strip()
            args = tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            if not name:
                continue
            from .base import ToolCall

            if on_event:
                on_event(
                    "tool_call",
                    f"Calling {name}",
                    name,
                    {
                        k: (v if not isinstance(v, str) or len(v) < 120 else v[:120] + "…")
                        for k, v in args.items()
                    },
                )
            try:
                result_text, should_stop = dispatch(ToolCall(name=name, args=args))
            except Exception as exc:  # noqa: BLE001
                results.append({"name": name, "arguments": args, "error": str(exc)})
                if on_event:
                    on_event("error", f"Tool error: {exc}", name, None)
                continue
            results.append({"name": name, "arguments": args, "result": result_text})
            if on_event:
                on_event("tool_result", result_text[:300], name, None)
            if should_stop:
                rationale = args.get("rationale", rationale)
                done_this_round = True

        if done_this_round and not results:
            # Model said done but called nothing — accept and exit
            if on_event:
                on_event("info", f"{backend_name} finished planning.", "", None)
            break
        if done_this_round and any(r.get("name") == "finalize_workflow" for r in results):
            if on_event:
                on_event("info", f"Finalized: {rationale[:200]}", "", None)
            break

        # Feed results back as next user turn
        next_user_turn = format_tool_results_for_next_turn(results)
        convo.append(f"<<ASSISTANT>>\n{json.dumps(env)}\n<<END ASSISTANT>>")
        convo.append(f"<<USER>>\n{next_user_turn}\n<<END USER>>")
        # In incremental mode the backend's session already holds the prior
        # turns, so the next call sends only the new tool results.
        pending_turn = next_user_turn

    return rationale
