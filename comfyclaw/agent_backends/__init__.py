"""
ComfyClaw agent backends — pluggable LLM drivers for the workflow tool-use loop.

A backend takes a system prompt, a user message, a tool schema (OpenAI
function-calling format), and a dispatch callback.  It runs the tool-call
conversation to completion and returns the final rationale string.

Built-in backends
-----------------
``litellm``        : Default — uses LiteLLM's unified API for any cloud or
                     local provider (Anthropic, OpenAI, Gemini, Groq, Ollama, …).
``claude-code``    : Claude Code CLI (``claude``) over a persistent
                     ``--output-format stream-json`` stdio session.
``codex``          : OpenAI Codex CLI (``codex``) over ``codex exec --json``.
``gemini-cli``     : Google Gemini CLI (``gemini``) over ``gemini -p --output-format json``.
``grok-cli``       : Grok Build CLI (``grok``) with OAuth subscription auth.

Selection
---------
``HarnessConfig.agent_backend`` chooses the backend by name.  The CLI
backends require their respective binary to be on ``$PATH``; if not, the
factory returns ``None`` from ``is_available()`` and the harness falls
back to ``litellm`` with a warning.
"""

from __future__ import annotations

from .base import AgentBackend, BackendStatus, ToolCall, get_backend, probe_all
from .litellm_backend import LiteLLMBackend

__all__ = [
    "AgentBackend",
    "BackendStatus",
    "ToolCall",
    "LiteLLMBackend",
    "get_backend",
    "probe_all",
]
