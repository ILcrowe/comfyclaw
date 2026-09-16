"""
ChatAgent — lightweight conversational LLM for the ComfyClaw chat panel.

Handles free-form user questions about the current workflow and general
ComfyUI assistance.  Streams response tokens back as an async generator.

Supported backends:

* ``"litellm"`` (default) — direct API call via LiteLLM with an
  ``ANTHROPIC_API_KEY`` / equivalent env var.
* ``"claude-code"`` — drives the local ``claude`` CLI in non-interactive
  ``-p`` mode and streams JSON events back into tokens.  Uses the
  ChatGPT/Claude subscription the user signed in with through the
  in-panel OAuth flow — no API key required.
* ``"codex"`` — drives the local ``codex exec --json`` CLI and parses
  its JSON event stream.  Uses the ChatGPT subscription the user signed
  in with through ``codex login``.
* ``"gemini-cli"`` — drives the local ``gemini -p`` CLI.  Uses the
  Google OAuth session the user signed in with through ``gemini``.
* ``"grok-cli"`` — drives Grok Build CLI with subscription OAuth.

The dispatcher :func:`chat_stream` picks the backend based on the
``agent_backend`` argument so the WebSocket handler can route chat
requests through whatever the user selected in the picker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncGenerator

log = logging.getLogger(__name__)

_SYSTEM_BASE = """\
You are ComfyClaw, an expert ComfyUI workflow assistant embedded in the live
ComfyClaw plugin.  Your role is to answer questions about ComfyUI, explain
workflow topologies, suggest improvements, and help debug problems.

Guidelines:
- Be concise and actionable. Format node class names and parameters in `backticks`.
- When a user wants to modify the workflow (add LoRA, change sampler, etc.),
  describe what you would change, then remind them to click ▶ Generate in
  "Improve Current" mode so the agent applies it.
- If the user asks what is currently in the workflow, summarise it from the
  node list below.
"""


def _summarize_workflow(workflow: dict | None) -> str:
    if not workflow:
        return "\n\nCurrent workflow: (empty — no nodes yet)"
    lines: list[str] = []
    for nid, node in list(workflow.items())[:40]:
        ct = node.get("class_type", "?")
        scalar_inputs = {
            k: v
            for k, v in (node.get("inputs") or {}).items()
            if not isinstance(v, list) and len(str(v)) < 80
        }
        inp_str = ", ".join(f"{k}={v!r}" for k, v in list(scalar_inputs.items())[:3])
        lines.append(f"  [{nid}] {ct}" + (f"  ({inp_str})" if inp_str else ""))
    if len(workflow) > 40:
        lines.append(f"  … and {len(workflow) - 40} more nodes")
    return "\n\nCurrent workflow nodes:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Skill library access for chat (hybrid: always-on catalogue + /skill trigger)
# ─────────────────────────────────────────────────────────────────────────────

# Chat is a single-shot completion (no tool loop), so instead of a ``read_skill``
# tool we inject the skill catalogue plus the full bodies of the most relevant /
# explicitly requested skills straight into the system prompt.
_CHAT_SKILL_MAX_BODIES = 3
_CHAT_SKILL_MAX_CHARS = 9000
_SKILL_CMD_RE = re.compile(r"/skills?\s+([a-zA-Z0-9 ,_\-]+)", re.IGNORECASE)

_chat_skill_registry = None  # lazy module-level fallback registry


def _default_chat_registry():
    """Build (once) a fallback SkillsRegistry when the caller passes none."""
    global _chat_skill_registry
    if _chat_skill_registry is None:
        try:
            from .skill_manager import SkillsRegistry

            _chat_skill_registry = SkillsRegistry(quiet=True)
        except Exception:
            _chat_skill_registry = False  # sentinel: tried and failed
    return _chat_skill_registry or None


def _latest_user_text(messages: list[dict] | None) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _parse_skill_commands(text: str, valid_names: set[str]) -> list[str]:
    """Extract skills the user force-loaded via ``/skill <name>`` (or ``/skills``).

    Everything after the command is tokenised; only tokens that exactly match a
    known skill name are kept, so the user's normal prose is ignored.
    """
    out: list[str] = []
    for m in _SKILL_CMD_RE.finditer(text or ""):
        for tok in re.split(r"[\s,]+", m.group(1).strip()):
            name = tok.strip().lower()
            if name in valid_names and name not in out:
                out.append(name)
    return out


def _build_skill_block(registry, messages: list[dict] | None) -> str:
    """Return the skill section to append to the chat system prompt.

    Always lists the enabled skill catalogue; additionally inlines the full
    instructions for skills the user explicitly requested via ``/skill`` and the
    ones that look relevant to the latest message (bounded in count and size).
    """
    if registry is None:
        return ""
    try:
        catalogue = registry.build_available_skills_xml()
        valid = set(registry.enabled_skill_names)
    except Exception:
        return ""
    if not valid:
        return ""

    latest = _latest_user_text(messages)
    explicit = _parse_skill_commands(latest, valid)
    try:
        relevant = registry.detect_relevant_skills(latest)
    except Exception:
        relevant = []

    load_order: list[str] = []
    for name in [*explicit, *relevant]:
        if name in valid and name not in load_order:
            load_order.append(name)

    bodies: list[tuple[str, str]] = []
    used = 0
    for name in load_order[:_CHAT_SKILL_MAX_BODIES]:
        try:
            body = (registry.get_body(name) or "").strip()
        except Exception:
            continue
        if not body:
            continue
        if bodies and used + len(body) > _CHAT_SKILL_MAX_CHARS:
            break
        bodies.append((name, body))
        used += len(body)

    parts = [
        "\n\n## ComfyClaw skill library\n"
        "You have a library of reusable skills (specialised instruction sets). The "
        "catalogue below lists each enabled skill's name and description. When a skill is "
        "relevant to the user's request, follow its guidance and tell the user which skill "
        "you applied. The user can force-load a skill by typing `/skill <name>` — treat that "
        "as a request to use that skill and do not echo the command syntax back.\n",
        catalogue,
    ]
    if bodies:
        parts.append(
            "\n\n## Loaded skill instructions\n"
            "Full instructions for the skills most relevant to this turn "
            "(apply these directly):\n"
        )
        for name, body in bodies:
            parts.append(f'\n<skill name="{name}">\n{body}\n</skill>\n')
    return "".join(parts)


def _flatten_history(messages: list[dict]) -> tuple[str, str]:
    """Split chat history into (prior_transcript, latest_user_message).

    The Claude CLI's ``-p`` mode takes a single prompt string; we prepend
    earlier turns as a transcript so the model has the context.  The
    most recent ``user`` message becomes the actual prompt.
    """
    if not messages:
        return "", ""
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx < 0:
        return "", ""
    prior = messages[:last_user_idx]
    latest = str(messages[last_user_idx].get("content") or "")
    if not prior:
        return "", latest
    transcript_lines: list[str] = []
    for m in prior:
        role = m.get("role", "user")
        content = str(m.get("content") or "")
        label = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            role, role.title()
        )
        transcript_lines.append(f"{label}: {content}")
    return "\n".join(transcript_lines), latest


# ─────────────────────────────────────────────────────────────────────────────
# Backend dispatch
# ─────────────────────────────────────────────────────────────────────────────


async def chat_stream(
    messages: list[dict],
    workflow: dict | None,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    agent_backend: str = "litellm",
    images: list[dict] | None = None,
    session_id: str = "",
    skills_registry: object = None,
) -> AsyncGenerator[str, None]:
    """Stream chat tokens from the selected backend.

    Parameters
    ----------
    messages, workflow:
        Conversation history and current workflow snapshot.
    model:
        Model id.  Used by the LiteLLM backend directly; the Claude
        backend translates it to a CLI alias if recognised.
    api_key, api_base:
        Only used by the LiteLLM backend.
    agent_backend:
        ``"litellm"`` (default), ``"claude-code"``, ``"codex"``, or
        ``"gemini-cli"``, or ``"grok-cli"``. Anything else falls back to LiteLLM with a
        log warning.
    """
    backend = (agent_backend or "litellm").strip().lower().replace("_", "-")
    # Resolve the skill library once and inject it into every backend's system
    # prompt (hybrid: always-on catalogue + /skill force-load + auto-relevance).
    # Callers can pass ``skills_registry=False`` to opt out (e.g. internal tasks
    # like skill refinement that shouldn't see the catalogue).
    if skills_registry is False:
        registry = None
    elif skills_registry is not None:
        registry = skills_registry
    else:
        registry = _default_chat_registry()
    skill_block = _build_skill_block(registry, messages)
    if backend in ("claude-code", "claude"):
        if images:
            yield "⚠️  Image attachments are currently supported in API mode (LiteLLM) only. "
            yield "Switch Access to API to run vision chat."
            return
        async for tok in _claude_chat_stream(messages, workflow, model, skill_block=skill_block):
            yield tok
        return
    if backend in ("codex", "openai-codex"):
        if images:
            yield "⚠️  Image attachments are currently supported in API mode (LiteLLM) only. "
            yield "Switch Access to API to run vision chat."
            return
        async for tok in _codex_chat_stream(
            messages, workflow, model, session_id=session_id, skill_block=skill_block
        ):
            yield tok
        return
    if backend in ("gemini-cli", "gemini"):
        if images:
            yield "⚠️  Image attachments are currently supported in API mode (LiteLLM) only. "
            yield "Switch Access to API to run vision chat."
            return
        async for tok in _gemini_chat_stream(messages, workflow, model, skill_block=skill_block):
            yield tok
        return
    if backend in ("grok-cli", "grok"):
        if images:
            yield "⚠️  Image attachments are currently supported in API mode (LiteLLM) only."
            return
        from .agent_backends.grok_backend import GrokCLIBackend, _get_recorded_grok_session

        prior, latest = _flatten_history(messages)
        user_text = latest if _get_recorded_grok_session(session_id) else (
            f"Previous conversation:\n{prior}\n\nLatest user message:\n{latest}" if prior else latest
        )
        grok = GrokCLIBackend(model=model, session_key=session_id)
        try:
            reply = await asyncio.to_thread(
                grok.run_tool_loop,
                _SYSTEM_BASE + skill_block + _summarize_workflow(workflow)
                + "\nFor this chat answer, use no tools. Put the full answer in rationale, "
                "set tool_calls to [] and done to true.",
                user_text,
                [],
                lambda _call: (_ for _ in ()).throw(RuntimeError("Chat tools are disabled")),
                None,
                1,
            )
            yield reply
        except RuntimeError as exc:
            yield f"⚠️  Grok CLI error: {exc}"
        return
    if backend != "litellm":
        log.warning(
            "[chat] Unknown agent_backend %r — falling back to LiteLLM.",
            agent_backend,
        )
    async for tok in _litellm_chat_stream(
        messages, workflow, model, api_key, api_base, images, skill_block=skill_block
    ):
        yield tok


# ─────────────────────────────────────────────────────────────────────────────
# LiteLLM backend (legacy)
# ─────────────────────────────────────────────────────────────────────────────


async def _litellm_chat_stream(
    messages: list[dict],
    workflow: dict | None,
    model: str,
    api_key: str | None,
    api_base: str | None,
    images: list[dict] | None,
    skill_block: str = "",
) -> AsyncGenerator[str, None]:
    import litellm  # lazy import — not always needed

    system = _SYSTEM_BASE + skill_block + _summarize_workflow(workflow)
    full_messages = [{"role": "system", "content": system}] + list(messages)

    # If images are attached, lift the last user turn into multimodal content.
    if images and full_messages:
        for i in range(len(full_messages) - 1, -1, -1):
            if full_messages[i].get("role") == "user":
                txt = str(full_messages[i].get("content") or "")
                blocks: list[dict] = [{"type": "text", "text": txt}]
                for img in images[:4]:
                    b64 = str(img.get("base64") or "").strip()
                    mime = str(img.get("mime") or "image/png").strip() or "image/png"
                    if not b64:
                        continue
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
                if len(blocks) > 1:
                    full_messages[i] = {"role": "user", "content": blocks}
                break

    kwargs: dict = {"model": model, "messages": full_messages, "stream": True}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = await litellm.acompletion(**kwargs)
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ─────────────────────────────────────────────────────────────────────────────
# Claude Code CLI backend
# ─────────────────────────────────────────────────────────────────────────────


def _claude_cli_model(model: str) -> str:
    """Translate a LiteLLM-style model string into a Claude CLI alias.

    Mirrors the heuristic in :mod:`.agent_backends.claude_code_backend`
    so chat uses the same model family as the agent does.
    """
    raw = (model or "").strip()
    if not raw:
        return ""
    if "/" in raw:
        _, _, suffix = raw.partition("/")
    else:
        suffix = raw
    suffix = suffix.lower()
    if suffix in {"default", "sonnet", "opus", "haiku"}:
        return suffix
    if suffix.startswith("claude-") and suffix[-8:].isdigit():
        return suffix
    if "sonnet" in suffix:
        return "sonnet"
    if "opus" in suffix:
        return "opus"
    if "haiku" in suffix:
        return "haiku"
    return ""


# Codex CLI model ids that can be forwarded as-is for ChatGPT-account
# auth.  Keep this list close to the CLI's own model picker, not the
# Responses API model page.
_CODEX_DEFAULT_MODEL = "gpt-5.5"
_CODEX_KNOWN_MODELS = {
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "o3",
    "o3-mini",
    "o4-mini",
}


def _codex_pick_model(model: str) -> str:
    """Pick a codex model from a user-supplied string.

    Resolution order:
        1. Explicit ``COMFYCLAW_CODEX_MODEL`` env-var override
           (operator escape hatch).
        2. The ``model`` arg, with any LiteLLM provider prefix
           (``openai/``, ``azure/openai/``, …) stripped when it names a
           Codex CLI model.
        3. ``gpt-5.5`` (recommended by the Codex CLI docs for most
           Codex tasks).

    Returning a single string lets ``_codex_chat_stream`` and
    ``CodexBackend.run_tool_loop`` share the same logic.
    """
    pin = os.environ.get("COMFYCLAW_CODEX_MODEL", "").strip()
    if pin:
        return pin
    raw = (model or "").strip().lower()
    if raw:
        suffix = raw.rsplit("/", 1)[-1]  # strip any provider prefix
        if suffix in _CODEX_KNOWN_MODELS:
            return suffix
        # Plain ``gpt-5`` is not a valid ChatGPT-account Codex CLI model
        # id.  ``gpt-5-codex`` is a Responses API model id and may also be
        # present in localStorage from older ComfyClaw builds.  Route both
        # to the current Codex CLI default.
        if suffix in {"gpt-5", "gpt-5-codex"}:
            return _CODEX_DEFAULT_MODEL
        # Preserve future Codex GPT-5.x model ids instead of collapsing
        # them to plain ``gpt-5``.
        if suffix.startswith("gpt-5."):
            return suffix
        if suffix.startswith("o3"):
            return "o3"
        if suffix.startswith("o4"):
            return "o4-mini"
    return _CODEX_DEFAULT_MODEL


# Gemini CLI defaults to Gemini 2.5 Pro on signed-in accounts; this set
# matches the public model list documented by ``gemini -m`` help and
# Google's pricing page.
_GEMINI_KNOWN_MODELS = {
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
}


def _gemini_pick_model(model: str) -> str:
    """Pick a Gemini CLI model id from a user-supplied string.

    Same resolution scheme as :func:`_codex_pick_model`.  Returns an
    empty string when no override applies, which the caller treats as
    "let the CLI fall back to its plan default".
    """
    pin = os.environ.get("COMFYCLAW_GEMINI_MODEL", "").strip()
    if pin:
        return pin
    raw = (model or "").strip().lower()
    if raw:
        suffix = raw.rsplit("/", 1)[-1]
        if suffix in _GEMINI_KNOWN_MODELS:
            return suffix
        if suffix.startswith("gemini-2.5"):
            return "gemini-2.5-pro" if "pro" in suffix else "gemini-2.5-flash"
        if suffix.startswith("gemini-2.0"):
            return "gemini-2.0-flash" if "flash" in suffix else "gemini-2.0-pro"
        if suffix.startswith("gemini-1.5"):
            return "gemini-1.5-pro" if "pro" in suffix else "gemini-1.5-flash"
    return ""


async def _claude_chat_stream(
    messages: list[dict],
    workflow: dict | None,
    model: str,
    skill_block: str = "",
) -> AsyncGenerator[str, None]:
    """Drive ``claude -p`` for a single chat reply.

    Uses ``--output-format stream-json --include-partial-messages`` so we
    get incremental ``text_delta`` events from the CLI; falls back to
    final ``assistant`` content blocks if no partials arrive (e.g. older
    CLI builds that don't support partials).
    """
    from .agent_backends.base import _env_with_claude_path, _resolve_claude_bin

    binary = _resolve_claude_bin()
    if not binary:
        yield (
            "⚠️  Claude Code is not installed on the server. "
            "Open the backend picker and click **Install Claude Code**, "
            "then try again."
        )
        return

    system_prompt = _SYSTEM_BASE + skill_block + _summarize_workflow(workflow)
    transcript, latest_user = _flatten_history(messages)
    if not latest_user:
        return
    if transcript:
        prompt = f"{transcript}\n\nUser: {latest_user}"
    else:
        prompt = latest_user

    argv = [
        binary,
        "-p",
        prompt,
        "--append-system-prompt",
        system_prompt,
        "--output-format",
        "stream-json",
        # stream-json requires --verbose on the CLI side; without it the
        # binary refuses to start with "requires --verbose".
        "--verbose",
        "--include-partial-messages",
        "--disable-slash-commands",
    ]
    cli_model = _claude_cli_model(model)
    if cli_model:
        argv += ["--model", cli_model]

    env = _env_with_claude_path(binary)

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError as exc:
        yield f"⚠️  Could not launch claude: {exc}"
        return

    saw_partial = False
    final_text_blocks: list[str] = []
    try:
        assert proc.stdout is not None
        while True:
            raw_line = await proc.stdout.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue

            mtype = evt.get("type")
            if mtype == "partial_message":
                delta = evt.get("delta") or evt.get("event", {}).get("delta") or {}
                # Two shapes we've seen:
                #   {"type":"text_delta","text":"..."} — current SDK
                #   {"text":"..."}                       — older SDK
                dtype = delta.get("type") or "text_delta"
                txt = delta.get("text")
                if not txt and isinstance(delta.get("partial_json"), str):
                    # tool-use input streaming — irrelevant for chat
                    continue
                if dtype == "text_delta" and txt:
                    saw_partial = True
                    yield txt
            elif mtype == "assistant":
                msg = evt.get("message") or {}
                for blk in msg.get("content") or []:
                    if blk.get("type") == "text":
                        txt = blk.get("text") or ""
                        if txt:
                            final_text_blocks.append(txt)
            elif mtype == "result":
                # Final aggregate event.  When ``is_error`` is true it
                # carries a human-readable description (e.g. "Not logged
                # in · Please run /login") that the user needs to see.
                if evt.get("is_error"):
                    err = evt.get("result") or evt.get("error") or "unknown error"
                    if "not logged in" in str(err).lower():
                        yield (
                            "⚠️  Claude Code is not signed in. Open the "
                            "backend picker and click **Sign in to Claude**, "
                            "then try again."
                        )
                    else:
                        yield f"⚠️  Claude CLI error: {err}"
                    saw_partial = True  # suppress empty-output fallback
                elif not saw_partial and not final_text_blocks:
                    # Older builds without partials AND without assistant
                    # messages emit only this single ``result`` event.
                    txt = evt.get("result") or ""
                    if txt:
                        yield str(txt)
                        saw_partial = True
            elif mtype == "error":
                err = evt.get("error") or evt.get("message") or line
                yield f"\n\n⚠️  Claude CLI error: {err}"

        # Drain stderr if anything went wrong but no error event arrived.
        rc = await proc.wait()
        if not saw_partial and not final_text_blocks:
            err_bytes = await proc.stderr.read() if proc.stderr else b""
            err_text = err_bytes.decode("utf-8", errors="replace").strip()
            if rc != 0 or err_text:
                yield (
                    f"\n\n⚠️  Claude CLI exited with code {rc}."
                    + (f"\n\n```\n{err_text[-600:]}\n```" if err_text else "")
                )
        elif not saw_partial and final_text_blocks:
            # We only got the final assistant message (no partials).
            for blk in final_text_blocks:
                yield blk
    except asyncio.CancelledError:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Codex CLI backend
# ─────────────────────────────────────────────────────────────────────────────


def _codex_extract_event(evt: dict) -> tuple[str, str, str]:
    """Parse one ``codex exec --json`` event into ``(kind, item_id, text)``.

    Aligned with the ``@openai/codex-sdk`` event taxonomy that the CLI
    re-uses verbatim (see ``reference/openai-codex.js`` for the
    Node-side mapping we mirror):

      * ``thread.started``                 → ``("ignore", "", "")``
      * ``turn.started`` / ``turn.completed``
                                          → ``("ignore", "", "")``
      * ``turn.failed``                    → ``("error", "", err_msg)``
      * ``error``                          → ``("error", "", err_msg)``
      * ``item.started`` (any item)        → ``("ignore", "", "")``
      * ``item.updated`` + ``agent_message``
                                          → ``("delta", id, text)``
                                            (text is cumulative — caller
                                             computes a real delta by
                                             diffing against what's been
                                             yielded for this id)
      * ``item.updated`` (other types)     → ``("ignore", "", "")``
      * ``item.completed`` + ``agent_message``
                                          → ``("message", id, full_text)``
      * ``item.completed`` + ``reasoning`` → ``("ignore", "", "")``
        (internal thinking noise — the reference UI explicitly drops it)
      * ``item.completed`` (other types)   → ``("ignore", "", "")``

    Older codex builds that pre-date the SDK schema may still emit
    flattened events shaped like ``{"msg":{"type":"agent_message",...}}``;
    we fall through to a legacy parser for those.
    """
    etype = (evt.get("type") or "").lower()

    # — Turn / thread lifecycle —————————————————————————————
    if etype in {"thread.started", "turn.started", "turn.completed"}:
        return "ignore", "", ""

    if etype in {"turn.failed", "error"}:
        err = evt.get("error") or evt.get("message") or ""
        if isinstance(err, dict):
            err = err.get("message") or err.get("error") or str(err)
        return ("error", "", str(err)) if err else ("ignore", "", "")

    # — Item lifecycle ————————————————————————————————————————
    if etype in {"item.started", "item.updated", "item.completed"}:
        item = evt.get("item") or {}
        itype = (item.get("type") or "").lower()
        item_id = str(item.get("id") or "")
        if itype == "agent_message":
            text = item.get("text") or ""
            if not isinstance(text, str) or not text:
                return "ignore", item_id, ""
            if etype == "item.updated":
                # Cumulative — caller will compute the new tail.
                return "delta", item_id, text
            if etype == "item.completed":
                return "message", item_id, text
            return "ignore", item_id, ""
        if itype == "error":
            err = item.get("message") or item.get("error") or ""
            return ("error", item_id, str(err)) if err else ("ignore", item_id, "")
        # reasoning / command_execution / file_change / mcp_tool_call /
        # web_search / todo_list — none of these belong in a chat reply.
        return "ignore", item_id, ""

    # — Legacy fallback for older codex CLIs ——————————————————————
    msg = evt.get("msg")
    if isinstance(msg, dict):
        mt = (msg.get("type") or "").lower()
        if "error" in mt:
            err = msg.get("message") or msg.get("error") or ""
            return ("error", "", str(err)) if err else ("ignore", "", "")
        if "delta" in mt:
            text = msg.get("delta") or msg.get("text") or msg.get("content") or ""
            return ("delta", "", text) if isinstance(text, str) and text else ("ignore", "", "")
        if mt in {"agent_message", "agent.response", "assistant_message"}:
            text = msg.get("message") or msg.get("text") or msg.get("content") or ""
            return ("message", "", text) if isinstance(text, str) and text else ("ignore", "", "")

    return "ignore", "", ""


# Known harmless codex internal log lines we never want to surface as a
# "Codex CLI error" in the chat panel.  These include codex's Rust tracing
# warnings about transient session state (rollout-items write racing with
# thread shutdown — happens on every short-lived ``codex exec --json`` run)
# and the chatty "Reading additional input from stdin..." status line some
# codex builds print even when stdin is closed.  Setting ``RUST_LOG=off``
# kills most of them at the source; this regex is the defense-in-depth.
_CODEX_NOISE_RE = re.compile(
    r"failed to record rollout items"
    r"|thread [0-9a-f-]+ not found"
    r"|reading additional input from stdin"
    r"|rollout (?:write|item)"
    r"|track_(?:session|usage)",
    re.IGNORECASE,
)


def _codex_is_noise(text: str) -> bool:
    return bool(_CODEX_NOISE_RE.search(text or ""))


async def _codex_chat_stream(
    messages: list[dict],
    workflow: dict | None,
    model: str,
    session_id: str = "",
    skill_block: str = "",
) -> AsyncGenerator[str, None]:
    """Drive ``codex exec --json`` for a single chat reply.

    Codex's headless mode reads the prompt from argv (no stdin); we
    flatten the chat history into a transcript prefix and let codex
    answer the latest user turn.

    The ``model`` parameter is intentionally ignored — codex's allowed
    model list is bound to the user's ChatGPT subscription and doesn't
    overlap with the LiteLLM dropdown values, so forwarding the LiteLLM
    model id only ever produces ``model not supported with ChatGPT
    account`` errors.  Operators can pin a specific codex model with
    ``COMFYCLAW_CODEX_MODEL=<id>``.

    We mute codex's Rust tracing logs (``RUST_LOG=off``) and drain
    stderr concurrently so those harmless internal warnings can't leak
    into the chat reply.  Any remaining noise that arrives as a JSON
    ``task_error`` event is filtered by :data:`_CODEX_NOISE_RE`.
    """
    import shutil as _shutil

    binary = _shutil.which("codex")
    if not binary:
        yield (
            "⚠️  Codex CLI is not installed on the server. Open Settings → "
            "Agents and click **Install** next to Codex, or run "
            "`brew install codex` / `npm i -g @openai/codex` in a terminal."
        )
        return

    system_prompt = _SYSTEM_BASE + skill_block + _summarize_workflow(workflow)
    transcript, latest_user = _flatten_history(messages)
    if not latest_user:
        return
    prompt = f"{system_prompt}\n\n"
    if transcript:
        prompt += f"{transcript}\n\nUser: {latest_user}"
    else:
        prompt += f"User: {latest_user}"

    # Mirror the ``threadOptions`` the @openai/codex-sdk reference passes
    # in (``reference/openai-codex.js`` ~line 395):
    #   skipGitRepoCheck: true    → --skip-git-repo-check
    #   sandboxMode: read-only    → --sandbox read-only  (chat doesn't
    #                               need to mutate the filesystem)
    # ``codex exec`` already implies a non-interactive ``approvalPolicy:
    # never`` so we don't need to pass it explicitly.
    argv = [binary, "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only"]
    # Codex's model registry is *not* the LiteLLM registry.  When the
    # user is signed in with a ChatGPT subscription, codex accepts ids
    # such as ``gpt-5.5`` and ``gpt-5.4``; plain ``gpt-5`` is rejected.
    # Translate the user's UI selection through :func:`_codex_pick_model`,
    # then apply it via ``-c model=…`` so the override beats the user's
    # ``~/.codex/config.toml`` without mutating that file.
    codex_model = _codex_pick_model(model)
    argv += ["-c", f'model="{codex_model}"']
    from .agent_backends.codex_backend import (
        _extract_codex_session_id,
        _get_recorded_codex_session,
        _record_codex_session,
    )

    codex_session_id = _get_recorded_codex_session(session_id)
    if codex_session_id:
        argv = [
            binary,
            "exec",
            "resume",
            "--json",
            "--skip-git-repo-check",
            "-c",
            f'model="{codex_model}"',
            codex_session_id,
            prompt,
        ]
    else:
        argv.append(prompt)

    # Mute the noisy log sources at three layers:
    #   - RUST_LOG=off       Rust tracing (e.g. the "failed to record
    #                        rollout items: thread … not found" ERROR)
    #   - CODEX_LOG_LEVEL    codex's own logger fallback
    #   - NO_COLOR=1         skip ANSI styling that would otherwise need
    #                        re-stripping in the noise filter
    env = {
        **os.environ,
        "RUST_LOG": "off",
        "CODEX_LOG_LEVEL": "error",
        "NO_COLOR": "1",
    }

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError as exc:
        yield f"⚠️  Could not launch codex: {exc}"
        return

    # Drain stderr concurrently — never yield it to the chat surface,
    # but keep the buffer empty so codex can't stall on a full pipe, and
    # hold onto the content for the fallback "not signed in" sniffing.
    stderr_buf = bytearray()

    async def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                break
            stderr_buf.extend(chunk)

    stderr_task = asyncio.create_task(_drain_stderr())

    saw_text = False
    # Map of agent_message item-id → length already yielded.  ``item.updated``
    # events carry the *cumulative* text for that item, so we slice off the
    # portion we haven't sent yet and yield only that.  This makes the chat
    # reply stream live without ever yielding the same characters twice.
    yielded_lens: dict[str, int] = {}
    final_texts: dict[str, str] = {}
    try:
        assert proc.stdout is not None
        while True:
            raw_line = await proc.stdout.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("{"):
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            new_session_id = _extract_codex_session_id(line)
            if new_session_id:
                _record_codex_session(session_id, new_session_id)
            kind, item_id, text = _codex_extract_event(evt)
            if kind == "ignore" or not text:
                continue
            # Filter known-harmless codex internal log strings before
            # surfacing anything to the user.  Without this, the rollout
            # warning shows up as "⚠️ Codex CLI error: failed to record
            # rollout items…" inside the chat reply.
            if _codex_is_noise(text):
                continue
            if kind == "delta":
                # ``text`` is cumulative for this agent_message item — yield
                # only the new tail since our last update for the same id.
                already = yielded_lens.get(item_id, 0)
                if len(text) > already:
                    new_part = text[already:]
                    yielded_lens[item_id] = len(text)
                    saw_text = True
                    yield new_part
            elif kind == "message":
                # ``item.completed`` arrived — yield any text not already
                # delivered via streaming deltas, then remember the full
                # message in case no deltas were emitted at all.
                already = yielded_lens.get(item_id, 0)
                if len(text) > already:
                    new_part = text[already:]
                    yielded_lens[item_id] = len(text)
                    saw_text = True
                    yield new_part
                final_texts[item_id] = text
            elif kind == "error":
                # Special-case the common "model not supported with
                # ChatGPT account" rejection so users get an actionable
                # fix-it message instead of a raw JSON blob.  This is
                # triggered when either the user's ``~/.codex/config.toml``
                # pins a model that isn't on their ChatGPT plan, or when
                # our own ``-c model=…`` override picks something the
                # plan doesn't include.
                low = text.lower()
                if "not supported when using codex with a chatgpt account" in low:
                    yield (
                        "\n\n⚠️  Codex rejected the requested model. This usually "
                        "means your ChatGPT plan doesn't include it, or your "
                        "`~/.codex/config.toml` pins a non-ChatGPT model "
                        "(e.g. `azure/openai/...`).\n\n"
                        "**Fix:** set a supported codex model via env-var before "
                        "starting comfyclaw, e.g.\n\n"
                        "```\n"
                        "export COMFYCLAW_CODEX_MODEL=gpt-5.5\n"
                        "comfyclaw serve\n"
                        "```\n\n"
                        "Valid choices on a ChatGPT subscription typically "
                        "include `gpt-5.5`, `gpt-5.4`, `o3`, `o4-mini`. "
                        "Original error:\n\n"
                        f"```\n{text}\n```"
                    )
                else:
                    yield f"\n\n⚠️  Codex CLI error: {text}"
                saw_text = True

        rc = await proc.wait()
        try:
            await stderr_task
        except Exception:  # noqa: BLE001
            pass
        # Some older codex builds emit only the final ``item.completed``
        # without preceding ``item.updated`` events — and the legacy parser
        # branches return ``item_id=""``.  Yield any aggregated text we have
        # left over so we don't drop the entire reply on that path.
        if not saw_text:
            for text in final_texts.values():
                if text:
                    yield text
                    saw_text = True
        if not saw_text:
            err_text = stderr_buf.decode("utf-8", errors="replace").strip()
            lower = err_text.lower()
            if "not signed in" in lower or "log in" in lower or "login" in lower:
                yield (
                    "⚠️  Codex CLI is not signed in. Open Settings → Agents "
                    "and click **Sign in** next to Codex, then try again."
                )
            elif rc != 0 or err_text:
                # Strip our known-noise patterns from stderr too so the
                # fallback block only shows lines that actually matter.
                cleaned = "\n".join(
                    ln for ln in err_text.splitlines() if ln.strip() and not _codex_is_noise(ln)
                )
                if rc != 0 or cleaned:
                    yield (
                        f"\n\n⚠️  Codex CLI exited with code {rc}."
                        + (f"\n\n```\n{cleaned[-600:]}\n```" if cleaned else "")
                    )
    except asyncio.CancelledError:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        stderr_task.cancel()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Gemini CLI backend
# ─────────────────────────────────────────────────────────────────────────────


async def _gemini_chat_stream(
    messages: list[dict],
    workflow: dict | None,
    model: str,
    skill_block: str = "",
) -> AsyncGenerator[str, None]:
    """Drive ``gemini -p "<prompt>"`` for a single chat reply.

    Gemini CLI's headless mode prints the model's answer to stdout with
    no structured framing, so we stream raw stdout bytes through as
    chunks.  We don't pass ``-m`` unless the model string clearly
    targets Gemini — passing an Anthropic / OpenAI string trips the CLI
    into an immediate error.
    """
    import shutil as _shutil

    binary = _shutil.which("gemini")
    if not binary:
        yield (
            "⚠️  Gemini CLI is not installed on the server. Open Settings → "
            "Agents and click **Install** next to Gemini CLI, or run "
            "`npm i -g @google/gemini-cli` in a terminal."
        )
        return

    system_prompt = _SYSTEM_BASE + skill_block + _summarize_workflow(workflow)
    transcript, latest_user = _flatten_history(messages)
    if not latest_user:
        return
    prompt = f"{system_prompt}\n\n"
    if transcript:
        prompt += f"{transcript}\n\nUser: {latest_user}"
    else:
        prompt += f"User: {latest_user}"

    # As with codex above, gemini's allowed-model list (tied to the
    # signed-in Google account) doesn't overlap with the LiteLLM panel
    # registry, so we translate the user's UI selection through
    # :func:`_gemini_pick_model` (LiteLLM prefix stripped, whitelisted
    # against the known-good set).  Empty result means "let the CLI use
    # its plan default" — we don't pass ``-m`` in that case so gemini
    # picks whatever its config says.
    gemini_model = _gemini_pick_model(model)
    if gemini_model:
        argv = [binary, "-m", gemini_model, "-p", prompt]
    else:
        argv = [binary, "-p", prompt]

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        yield f"⚠️  Could not launch gemini: {exc}"
        return

    # Drain stderr concurrently so its pipe can't fill up and stall gemini.
    # Same approach as the Codex path — we never forward stderr into the
    # chat reply on the success path, only on the empty-output fallback.
    stderr_buf = bytearray()

    async def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                break
            stderr_buf.extend(chunk)

    stderr_task = asyncio.create_task(_drain_stderr())

    saw_text = False
    try:
        assert proc.stdout is not None
        # Stream raw chunks as they arrive rather than line-by-line, so
        # the UI feels responsive even when gemini's response is one big
        # paragraph.
        while True:
            chunk = await proc.stdout.read(256)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            if text:
                saw_text = True
                yield text

        rc = await proc.wait()
        try:
            await stderr_task
        except Exception:  # noqa: BLE001
            pass
        if not saw_text:
            err_text = stderr_buf.decode("utf-8", errors="replace").strip()
            lower = err_text.lower()
            if "not authenticated" in lower or "login" in lower or "oauth" in lower:
                yield (
                    "⚠️  Gemini CLI is not signed in. Run `gemini` once in a "
                    "terminal to complete Google OAuth, then try again."
                )
            elif rc != 0 or err_text:
                yield (
                    f"\n\n⚠️  Gemini CLI exited with code {rc}."
                    + (f"\n\n```\n{err_text[-600:]}\n```" if err_text else "")
                )
    except asyncio.CancelledError:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        stderr_task.cancel()
        raise
