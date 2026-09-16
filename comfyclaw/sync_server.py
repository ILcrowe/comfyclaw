"""
SyncServer — lightweight WebSocket server that bridges the Python harness
to connected ComfyUI browser tabs.

Architecture (v2 — per-connection state isolation)
===================================================

One ComfyUI tab  =  one WebSocket connection  =  one _ConnState.

Each _ConnState owns:
  • workflow      — last known API-format workflow dict for this tab
  • checkpoints   — per-tab snapshots
  • cancel        — threading.Event to abort the running generation
  • feedback_fut  — asyncio.Future awaiting human feedback
  • refinement_fut— asyncio.Future awaiting user refinement text

The serve loop (cli.py) calls wait_for_trigger() which returns
``(trigger_dict, source_ws)`` so every subsequent broadcast / status
message can be routed back to exactly the tab that fired the trigger.

Multiple ComfyClaw sessions can share the same workflow (workflowId).
That is handled entirely on the frontend — the backend is session-agnostic.

Message types (server → client):
  workflow_update   — full workflow snapshot (initial / reconnect)
  workflow_diff     — incremental add/remove/update ops
  request_feedback  — ask human for feedback
  generation_status — progress during a run
  generation_complete
  generation_error
  agent_event       — thinking-log entry
  chat_response     — streaming LLM token
  checkpoints_list  — list of checkpoints for this connection
  checkpoint_saved
  checkpoint_restored
  debug_status / debug_result

Message types (client → server):
  hello               — connection registration (connection_id optional)
  trigger_generation
  cancel_generation
  human_feedback
  user_refinement
  chat_message            — free-form chat (accepts agent_backend to pick
                            litellm vs claude-code)
  debug_workflow
  save_checkpoint
  restore_checkpoint
  list_checkpoints
  list_agent_backends
  list_provider_keys       — ask which LiteLLM provider env-vars are set
                              (panel filters its provider bar accordingly)
  backend_install_start    — kick off CLI installer
                              (claude-code / codex / gemini-cli / grok-cli)
  backend_install_cancel
  backend_auth_start       — drive sign-in flow for claude-code or codex.
                              Optional ``auth_method`` selects the variant
                              ("claudeai"/"console" for claude; "browser"/
                              "device_code" for codex).  ``force=True``
                              runs ``<binary> logout`` first (re-login).
  backend_auth_paste_code  — forward the redirect URL into claude's stdin
  backend_auth_cancel
  model_url_download       — download a pasted model URL into ComfyUI/models

Additional message types (server → client):
  agent_backends           — backend availability (with state per backend)
  backend_install_progress
  backend_install_complete
  backend_auth_url         — server-captured "Open in browser:" URL
  backend_auth_progress
  backend_auth_complete
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import websockets
    import websockets.server

    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Feedback image helper
# ──────────────────────────────────────────────────────────────────────────────


def _encode_image_data_url(image_path: str | None) -> str:
    """Read *image_path* and return a ``data:`` URL, or ``""`` on any failure.

    The feedback panel renders in the browser, which can't reach a server-side
    filesystem path, so the image bytes are inlined as a base64 data URL.
    """
    if not image_path:
        return ""
    try:
        import base64

        data = Path(image_path).read_bytes()
    except Exception:
        return ""
    if not data:
        return ""
    head = data[:12]
    if head[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    elif head[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif head[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    elif head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


# ──────────────────────────────────────────────────────────────────────────────
# Diff helpers
# ──────────────────────────────────────────────────────────────────────────────


def diff_workflows(old: dict, new: dict) -> list[dict]:
    """Compare two API-format workflow dicts and return a list of ops."""
    ops: list[dict] = []
    old_keys, new_keys = set(old), set(new)
    for nid in sorted(new_keys - old_keys, key=lambda k: int(k)):
        ops.append({"op": "add_node", "id": nid, "data": new[nid]})
    for nid in sorted(old_keys - new_keys, key=lambda k: int(k)):
        ops.append({"op": "remove_node", "id": nid})
    for nid in sorted(old_keys & new_keys, key=lambda k: int(k)):
        if old[nid] != new[nid]:
            ops.append({"op": "update_node", "id": nid, "data": new[nid]})
    return ops


_MAX_CHECKPOINTS = 20
_CHECKPOINT_DEFAULT_SESSION = "__default__"


def _checkpoint_session_key(session_id: str = "") -> str:
    session_id = str(session_id or "").strip()
    return session_id or _CHECKPOINT_DEFAULT_SESSION


# ──────────────────────────────────────────────────────────────────────────────
# Per-connection state
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _ConnState:
    """All mutable state scoped to a single WebSocket connection (= one ComfyUI tab)."""

    ws: Any  # websocket handle
    connection_id: str = ""  # opaque id sent by the client
    workflow: dict | None = None  # last known API-format workflow
    checkpoints: list = field(default_factory=list)
    cp_counter: int = 0
    checkpoints_by_session: dict[str, list] = field(default_factory=dict)
    cp_counter_by_session: dict[str, int] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)

    # asyncio futures — only valid inside the event-loop thread
    feedback_fut: Any = None  # asyncio.Future[dict] | None
    refinement_fut: Any = None  # asyncio.Future[dict] | None
    skill_evolution_fut: Any = None  # asyncio.Future[dict] | None
    model_download_fut: Any = None  # asyncio.Future[dict] | None
    compute_confirm_fut: Any = None  # asyncio.Future[dict] | None

    # ── Run-mode early-stop (accept_now) ──────────────────────────────────────
    accept_now: threading.Event = field(default_factory=threading.Event)

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def _checkpoint_bucket(self, session_id: str = "") -> list:
        key = _checkpoint_session_key(session_id)
        if not self.checkpoints_by_session and self.checkpoints:
            self.checkpoints_by_session[_CHECKPOINT_DEFAULT_SESSION] = self.checkpoints
            self.cp_counter_by_session[_CHECKPOINT_DEFAULT_SESSION] = self.cp_counter
        return self.checkpoints_by_session.setdefault(key, [])

    def save_checkpoint(self, workflow: dict, label: str = "", session_id: str = "") -> str:
        key = _checkpoint_session_key(session_id)
        counter = self.cp_counter_by_session.get(key, 0) + 1
        self.cp_counter_by_session[key] = counter
        self.cp_counter += 1
        cp_id = f"cp_{int(time.time())}_{self.cp_counter}"
        bucket = self._checkpoint_bucket(session_id)
        bucket.append(
            {
                "id": cp_id,
                "label": label or f"Snapshot #{counter}",
                "workflow": copy.deepcopy(workflow),
                "timestamp": time.time(),
                "session_id": session_id or "",
            }
        )
        if len(bucket) > _MAX_CHECKPOINTS:
            self.checkpoints_by_session[key] = bucket[-_MAX_CHECKPOINTS:]
        if key == _CHECKPOINT_DEFAULT_SESSION:
            self.checkpoints = self.checkpoints_by_session[key]
        return cp_id

    def list_checkpoints(self, session_id: str = "") -> list[dict]:
        bucket = self._checkpoint_bucket(session_id)
        return [
            {
                "id": c["id"],
                "label": c["label"],
                "timestamp": c["timestamp"],
                "session_id": c.get("session_id", session_id or ""),
            }
            for c in reversed(bucket)
        ]

    def restore_checkpoint(self, cp_id: str, session_id: str = "") -> dict | None:
        """Return the workflow dict for *cp_id*, or None if not found."""
        cp = next((c for c in self._checkpoint_bucket(session_id) if c["id"] == cp_id), None)
        return copy.deepcopy(cp["workflow"]) if cp else None


# ──────────────────────────────────────────────────────────────────────────────
# SyncServer
# ──────────────────────────────────────────────────────────────────────────────


class SyncServer:
    """
    Parameters
    ----------
    port            : TCP port to listen on (default 8765).
    host            : Bind address (default ``"0.0.0.0"``).
    model           : Default LiteLLM model for chat / debug agents.
    api_key         : Default LLM API key (falls back to env vars if None).
    server_address  : ComfyUI server address used by the debug agent.
    """

    def __init__(
        self,
        port: int = 8765,
        host: str = "0.0.0.0",
        model: str = "anthropic/claude-sonnet-4-5",
        api_key: str | None = None,
        server_address: str = "127.0.0.1:8188",
        skills_dir: str | None = None,
        quiet: bool = False,
    ) -> None:
        self.port = port
        self.host = host
        self._model = model
        self._api_key = api_key
        self._server_address = server_address
        self._quiet = quiet

        # Skills registry (lazy)
        self._skills_dir = skills_dir
        self._skills_registry: Any = None
        self._skills_lock = threading.Lock()

        # websocket → _ConnState
        self._conns: dict[Any, _ConnState] = {}
        self._conns_lock = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._started_ok = False

        # Global trigger queue — any tab can trigger, serve loop consumes one at a time.
        # Items: (trigger_dict, source_ws)
        self._trigger_queue: asyncio.Queue | None = None

        # Convenience property kept for legacy code in cli.py that checks
        # ``sync.cancel_requested``; it now delegates to the active connection.
        self._active_ws: Any = None  # set when trigger arrives
        self._active_ws_lock = threading.Lock()

        # In-flight backend setup flows (install/OAuth), one per ws.
        from .setup_flows import SetupFlowRegistry

        self._setup_flows = SetupFlowRegistry()

    # ── convenience: active connection's cancel flag ────────────────────────

    @property
    def cancel_requested(self) -> threading.Event:
        with self._active_ws_lock:
            ws = self._active_ws
        if ws:
            with self._conns_lock:
                conn = self._conns.get(ws)
            if conn:
                return conn.cancel
        # Fallback: return a dummy never-set event
        return threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if not _WS_AVAILABLE:
            log.warning(
                "[SyncServer] 'websockets' not installed — live sync disabled. "
                "Install with: pip install 'comfyclaw[sync]'"
            )
            return
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._started_ok = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="comfyclaw-sync")
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None

    def is_running(self) -> bool:
        return bool(self._started_ok and self._thread and self._thread.is_alive())

    def has_clients(self) -> bool:
        with self._conns_lock:
            return bool(self._conns)

    # ── per-connection workflow broadcast ────────────────────────────────────

    def broadcast(self, workflow: dict, target_ws: Any = None) -> None:
        """
        Send an incremental diff (or full snapshot on first call) to
        *target_ws* only.  If *target_ws* is None, broadcast to all connections.

        Safe to call from any thread.
        """
        targets: list[_ConnState]
        with self._conns_lock:
            if target_ws is not None:
                conn = self._conns.get(target_ws)
                targets = [conn] if conn else []
            else:
                targets = list(self._conns.values())

        if not targets:
            return

        for conn in targets:
            prev = conn.workflow
            if prev is None:
                payload = json.dumps({"type": "workflow_update", "workflow": workflow})
            else:
                ops = diff_workflows(prev, workflow)
                if not ops:
                    conn.workflow = copy.deepcopy(workflow)
                    continue
                payload = json.dumps({"type": "workflow_diff", "ops": ops, "full": workflow})
            conn.workflow = copy.deepcopy(workflow)

            if self._loop:
                asyncio.run_coroutine_threadsafe(self._send_to(conn.ws, payload), self._loop)

    def reset(self, target_ws: Any = None, *, empty: bool = False) -> None:
        """Clear the remembered workflow for *target_ws* (or all connections)."""
        with self._conns_lock:
            conns = (
                [self._conns[target_ws]]
                if target_ws and target_ws in self._conns
                else list(self._conns.values())
            )
        for conn in conns:
            conn.workflow = {} if empty else None

    # ── send helpers ─────────────────────────────────────────────────────────

    def _send_json_to(self, ws: Any, msg: dict) -> None:
        """Send a JSON message to a specific websocket (thread-safe)."""
        if not self._loop or not self.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._send_to(ws, json.dumps(msg)), self._loop)

    def _send_json(self, msg: dict, target_ws: Any = None) -> None:
        """Send to target_ws only, or broadcast to all if target_ws is None."""
        if not self._loop or not self.is_running():
            return
        with self._conns_lock:
            targets = (
                {target_ws} if target_ws and target_ws in self._conns else set(self._conns.keys())
            )
        if not targets:
            return
        payload = json.dumps(msg)
        asyncio.run_coroutine_threadsafe(self._async_broadcast(payload, targets), self._loop)

    # ── generation status — routed to the triggering tab ────────────────────

    def send_status(
        self, state: str, iteration: int = 0, detail: str = "", target_ws: Any = None
    ) -> None:
        self._send_json(
            {
                "type": "generation_status",
                "state": state,
                "iteration": iteration,
                "detail": detail,
            },
            target_ws=target_ws,
        )

    def send_complete(
        self,
        score: float,
        iterations_used: int,
        image_path: str = "",
        target_ws: Any = None,
        answer: str = "",
    ) -> None:
        self._send_json(
            {
                "type": "generation_complete",
                "score": score,
                "iterations_used": iterations_used,
                "image_path": image_path,
                "answer": answer,
            },
            target_ws=target_ws,
        )

    def send_error(self, error: str, target_ws: Any = None) -> None:
        self._send_json({"type": "generation_error", "error": error}, target_ws=target_ws)

    def send_agent_event(
        self,
        event_type: str,
        content: str,
        *,
        iteration: int = 0,
        tool_name: str = "",
        tool_args: dict | None = None,
        target_ws: Any = None,
    ) -> None:
        msg: dict[str, Any] = {
            "type": "agent_event",
            "event_type": event_type,
            "content": content,
            "timestamp": time.time(),
            "iteration": iteration,
        }
        if tool_name:
            msg["tool_name"] = tool_name
        if tool_args:
            msg["tool_args"] = tool_args
        self._send_json(msg, target_ws=target_ws)

    def request_skill_evolution(
        self,
        proposal: dict,
        target_ws: Any = None,
        timeout: float = 600.0,
    ) -> dict:
        """Ask one ComfyUI tab whether a post-run skill proposal should apply.

        Returns a dict ``{"approved": bool, "proposal": dict | None}`` where
        ``proposal`` is the (possibly human-edited) skill the reviewer chose to
        apply, or ``{"approved": False}`` if no client could be reached.
        """
        if not self._loop or not self.is_running():
            return {"approved": False}

        async def _ask() -> dict:
            with self._conns_lock:
                ws = (
                    target_ws
                    if target_ws and target_ws in self._conns
                    else next(iter(self._conns.keys()), None)
                )
                conn = self._conns.get(ws) if ws else None
            if ws is None or conn is None:
                return {"approved": False}
            fut = self._loop.create_future()  # type: ignore[union-attr]
            conn.skill_evolution_fut = fut
            await ws.send(
                json.dumps(
                    {
                        "type": "skill_evolution_proposal",
                        "proposal": proposal,
                    }
                )
            )
            try:
                reply = await asyncio.wait_for(fut, timeout=timeout)
            finally:
                conn.skill_evolution_fut = None
            return reply if isinstance(reply, dict) else {"approved": bool(reply)}

        fut = asyncio.run_coroutine_threadsafe(_ask(), self._loop)
        try:
            result = fut.result(timeout=timeout + 5.0)
            return result if isinstance(result, dict) else {"approved": bool(result)}
        except Exception:
            return {"approved": False}

    def request_model_download(
        self,
        request: dict,
        target_ws: Any = None,
        timeout: float = 600.0,
    ) -> dict:
        """Ask one ComfyUI tab whether the agent may download model weights."""
        if not self._loop or not self.is_running():
            return {"approved": False, "reason": "No connected ComfyClaw panel."}

        async def _ask() -> dict:
            with self._conns_lock:
                ws = (
                    target_ws
                    if target_ws and target_ws in self._conns
                    else next(iter(self._conns.keys()), None)
                )
                conn = self._conns.get(ws) if ws else None
            if ws is None or conn is None:
                return {"approved": False, "reason": "No connected ComfyClaw panel."}
            fut = self._loop.create_future()  # type: ignore[union-attr]
            if conn.model_download_fut and not conn.model_download_fut.done():
                conn.model_download_fut.cancel()
            conn.model_download_fut = fut
            await ws.send(
                json.dumps(
                    {
                        "type": "model_download_request",
                        "request": request,
                    }
                )
            )
            try:
                reply = await asyncio.wait_for(fut, timeout=timeout)
            finally:
                conn.model_download_fut = None
            return dict(reply or {})

        fut = asyncio.run_coroutine_threadsafe(_ask(), self._loop)
        try:
            reply = fut.result(timeout=timeout + 5.0)
            return {
                "approved": bool(reply.get("approved")),
                "reason": str(reply.get("reason") or ""),
            }
        except Exception as exc:
            return {"approved": False, "reason": f"Timed out waiting for approval: {exc}"}

    def request_generation_compute_confirmation(
        self,
        risk: dict,
        target_ws: Any = None,
        timeout: float = 600.0,
    ) -> dict:
        """Ask one ComfyUI tab whether to run generation despite compute risk."""
        if not self._loop or not self.is_running():
            return {"approved": False, "reason": "No connected ComfyClaw panel."}

        async def _ask() -> dict:
            with self._conns_lock:
                ws = (
                    target_ws
                    if target_ws and target_ws in self._conns
                    else next(iter(self._conns.keys()), None)
                )
                conn = self._conns.get(ws) if ws else None
            if ws is None or conn is None:
                return {"approved": False, "reason": "No connected ComfyClaw panel."}
            fut = self._loop.create_future()  # type: ignore[union-attr]
            if conn.compute_confirm_fut and not conn.compute_confirm_fut.done():
                conn.compute_confirm_fut.cancel()
            conn.compute_confirm_fut = fut
            await ws.send(
                json.dumps(
                    {
                        "type": "generation_compute_warning",
                        "risk": risk,
                    }
                )
            )
            try:
                reply = await asyncio.wait_for(fut, timeout=timeout)
            finally:
                conn.compute_confirm_fut = None
            return dict(reply or {})

        fut = asyncio.run_coroutine_threadsafe(_ask(), self._loop)
        try:
            reply = fut.result(timeout=timeout + 5.0)
            return {
                "approved": bool(reply.get("approved")),
                "reason": str(reply.get("reason") or ""),
            }
        except Exception as exc:
            return {"approved": False, "reason": f"Timed out waiting for approval: {exc}"}

    # ── trigger queue (serve loop) ───────────────────────────────────────────

    def wait_for_trigger(self, timeout: float = 0) -> tuple[dict, Any] | None:
        """
        Block until any client sends ``trigger_generation``.

        Returns ``(trigger_dict, source_ws)`` or ``None`` on timeout / not running.
        *timeout=0* means wait forever.
        """
        if not self._loop or not self.is_running():
            return None

        # Run a coroutine that fetches one item from the asyncio Queue
        async def _get() -> tuple[dict, Any]:
            assert self._trigger_queue is not None
            return await self._trigger_queue.get()

        fut = asyncio.run_coroutine_threadsafe(_get(), self._loop)
        try:
            result = fut.result(timeout=timeout if timeout > 0 else None)
            with self._active_ws_lock:
                self._active_ws = result[1]
            return result
        except Exception:
            return None

    # ── human-in-the-loop feedback ───────────────────────────────────────────

    def request_feedback(
        self,
        image_path: str | None = None,
        vlm_summary: str | None = None,
        iteration: int = 0,
        prompt: str = "",
        target_ws: Any = None,
        verifier: dict | None = None,
    ) -> None:
        # Browsers can't read a server-side filesystem path, so embed the
        # rendered image as a data URL the panel can display inline. This is
        # the single choke-point every feedback caller flows through, so doing
        # it here means all of them (VLM, hybrid, pure-human) get the preview.
        self._send_json(
            {
                "type": "request_feedback",
                "image_path": image_path,
                "image_b64": _encode_image_data_url(image_path),
                "vlm_summary": vlm_summary,
                "verifier": verifier,
                "iteration": iteration,
                "prompt": prompt,
            },
            target_ws=target_ws,
        )

    def wait_for_human_feedback(self, timeout: float = 600.0, source_ws: Any = None) -> dict | None:
        """Block until the client that owns *source_ws* sends human_feedback."""
        if not self._loop or not self.is_running():
            return None
        with self._conns_lock:
            conn = self._conns.get(source_ws) if source_ws else None
            if conn is None and self._conns:
                conn = next(iter(self._conns.values()))
        if conn is None:
            return None

        async def _create() -> asyncio.Future[dict]:
            lp = asyncio.get_running_loop()
            if conn.feedback_fut and not conn.feedback_fut.done():
                conn.feedback_fut.cancel()
            conn.feedback_fut = lp.create_future()
            return conn.feedback_fut

        fut_outer = asyncio.run_coroutine_threadsafe(_create(), self._loop).result(5)
        blocking = asyncio.run_coroutine_threadsafe(self._await_future(fut_outer), self._loop)
        try:
            return blocking.result(timeout=timeout)
        except Exception:
            conn.feedback_fut = None
            return None

    # ── user refinement ──────────────────────────────────────────────────────

    def enable_refinement_listening(self, source_ws: Any = None) -> None:
        if not self._loop or not self.is_running():
            return
        with self._conns_lock:
            conn = (
                self._conns.get(source_ws) if source_ws else next(iter(self._conns.values()), None)
            )
        if conn is None:
            return

        async def _arm() -> None:
            lp = asyncio.get_running_loop()
            if conn.refinement_fut and not conn.refinement_fut.done():
                conn.refinement_fut.cancel()
            conn.refinement_fut = lp.create_future()

        asyncio.run_coroutine_threadsafe(_arm(), self._loop)

    def poll_refinement(self, source_ws: Any = None) -> dict | None:
        with self._conns_lock:
            conn = (
                self._conns.get(source_ws) if source_ws else next(iter(self._conns.values()), None)
            )
        if conn is None:
            return None
        if conn.refinement_fut and conn.refinement_fut.done():
            try:
                return conn.refinement_fut.result()
            except Exception:
                return None
            finally:
                conn.refinement_fut = None
        return None

    # ── legacy checkpoint shims (delegate to active connection) ─────────────

    def save_checkpoint(
        self,
        workflow: dict,
        label: str = "",
        target_ws: Any = None,
        session_id: str = "",
    ) -> str:
        with self._conns_lock:
            conn = (
                self._conns.get(target_ws) if target_ws else next(iter(self._conns.values()), None)
            )
        if conn is None:
            return ""
        return conn.save_checkpoint(workflow, label, session_id=session_id)

    def list_checkpoints(self, target_ws: Any = None, session_id: str = "") -> list[dict]:
        with self._conns_lock:
            conn = (
                self._conns.get(target_ws) if target_ws else next(iter(self._conns.values()), None)
            )
        return conn.list_checkpoints(session_id=session_id) if conn else []

    # ── Iteration scoreboard ─────────────────────────────────────────────────

    def send_iteration_score(
        self,
        iteration: int,
        score: float | None,
        delta: float | None,
        critique: str = "",
        image_path: str = "",
        target_ws: Any = None,
    ) -> None:
        """Emit a scoreboard event after each iteration's verifier pass."""
        self._send_json(
            {
                "type": "iteration_score",
                "iteration": iteration,
                "score": score,
                "delta": delta,
                "critique": critique,
                "image_path": image_path,
                "timestamp": time.time(),
            },
            target_ws=target_ws,
        )

    def accept_requested(self, target_ws: Any = None) -> bool:
        """True if the user clicked 'Accept now' on this connection."""
        with self._conns_lock:
            conn = (
                self._conns.get(target_ws) if target_ws else next(iter(self._conns.values()), None)
            )
        return bool(conn and conn.accept_now.is_set())

    # ── Skills registry access ───────────────────────────────────────────────

    def skills_registry(self):
        """Lazily build and return a shared SkillsRegistry."""
        with self._skills_lock:
            if self._skills_registry is None:
                from .skill_manager import SkillsRegistry

                self._skills_registry = SkillsRegistry(self._skills_dir)
            return self._skills_registry

    def reload_skills(self) -> None:
        with self._skills_lock:
            if self._skills_registry is not None:
                self._skills_registry.reload()
        # Push the refreshed list to every connected tab so newly written skills
        # (e.g. from approved post-run evolution) show up in the Skills panel
        # immediately, without the user having to hit the manual refresh button.
        try:
            manifest = self.skills_registry().get_manifest()
        except Exception:
            return
        self._send_json({"type": "skills_manifest", "skills": manifest})

    # ── async helpers for chat + debug ───────────────────────────────────────

    async def _send_skills_manifest(self, ws: Any) -> None:
        try:
            manifest = self.skills_registry().get_manifest()
        except Exception as exc:  # noqa: BLE001
            await self._send_skill_error(ws, f"Could not list skills: {exc}")
            return
        await ws.send(json.dumps({"type": "skills_manifest", "skills": manifest}))

    async def _send_skill_error(self, ws: Any, message: str) -> None:
        await ws.send(json.dumps({"type": "skill_error", "error": message}))

    async def _refine_skill_preview(self, ws: Any, msg: dict) -> None:
        """Generate a refined SKILL.md body with the selected model/backend."""
        name = str(msg.get("name") or "").strip()
        if not name:
            await self._send_skill_error(ws, "Missing skill name.")
            return

        try:
            reg = self.skills_registry()
            props = reg.get_properties(name)
            body = reg.get_body(name)
        except KeyError:
            await self._send_skill_error(ws, "Skill not found.")
            return

        model = str(msg.get("model") or "").strip() or self._model
        api_key = (msg.get("api_key") or "").strip() or self._api_key
        api_base = (msg.get("api_base") or "").strip() or None
        backend = (
            (msg.get("agent_backend") or "").strip().lower()
            or os.environ.get("COMFYCLAW_AGENT_BACKEND", "").strip().lower()
            or "litellm"
        )
        if backend in {"claude-code", "codex", "gemini-cli", "grok-cli"}:
            api_key = None
            api_base = None

        from .chat_agent import chat_stream

        prompt = (
            "Refine this ComfyClaw skill for clarity and practical usefulness. "
            "Keep the same intent and tone, improve structure and actionable details. "
            "Return ONLY the improved Markdown body (no YAML frontmatter, no code fences).\n\n"
            f"Skill name: {name}\n"
            f"Description: {props.description}\n\n"
            "Current body:\n"
            f"{body}\n"
        )

        chunks: list[str] = []
        try:
            async for tok in chat_stream(
                [{"role": "user", "content": prompt}],
                workflow=None,
                model=model,
                api_key=api_key,
                api_base=api_base,
                agent_backend=backend,
                skills_registry=False,
            ):
                chunks.append(tok)
            refined = "".join(chunks).strip()
            if not refined:
                await self._send_skill_error(ws, "Model returned empty refinement.")
                return

            await ws.send(
                json.dumps(
                    {
                        "type": "skill_refine_preview",
                        "name": name,
                        "original_body": body,
                        "refined_body": refined,
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            await self._send_skill_error(ws, f"Refinement failed: {exc}")

    async def _apply_skill_refine(self, ws: Any, msg: dict) -> None:
        name = str(msg.get("name") or "").strip()
        refined_body = str(msg.get("refined_body") or "")
        if not name:
            await self._send_skill_error(ws, "Missing skill name.")
            return
        if not refined_body.strip():
            await self._send_skill_error(ws, "Refined body is empty.")
            return

        try:
            self.skills_registry().update_body(name, refined_body)
            await self._send_skills_manifest(ws)
            await ws.send(
                json.dumps(
                    {
                        "type": "skill_refine_result",
                        "ok": True,
                        "name": name,
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            await self._send_skill_error(ws, f"Could not apply refinement: {exc}")

    async def _import_skill_folder(self, ws: Any, msg: dict) -> None:
        path = msg.get("path", "")
        try:
            name = self.skills_registry().import_from_folder(path)
            await self._send_skills_manifest(ws)
            await ws.send(
                json.dumps(
                    {
                        "type": "skill_import_result",
                        "ok": True,
                        "action": "folder",
                        "name": name,
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            await self._send_skill_error(ws, f"Folder import failed: {exc}")

    async def _import_skill_zip(self, ws: Any, msg: dict) -> None:
        b64 = msg.get("base64", "")
        origin = msg.get("filename", "<uploaded.zip>")
        try:
            name = self.skills_registry().import_from_zip_b64(b64, origin=origin)
            await self._send_skills_manifest(ws)
            await ws.send(
                json.dumps(
                    {
                        "type": "skill_import_result",
                        "ok": True,
                        "action": "zip",
                        "name": name,
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            await self._send_skill_error(ws, f"Zip import failed: {exc}")

    async def _import_skill_git(self, ws: Any, msg: dict) -> None:
        url = msg.get("url", "")
        ref = msg.get("ref") or None
        try:
            # git clone is synchronous and short — we run it directly.
            name = self.skills_registry().import_from_git(url, ref)
            await self._send_skills_manifest(ws)
            await ws.send(
                json.dumps(
                    {
                        "type": "skill_import_result",
                        "ok": True,
                        "action": "git",
                        "name": name,
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            await self._send_skill_error(ws, f"Git import failed: {exc}")

    # ── Backend setup flows (install + OAuth) ────────────────────────────────

    # ── Provider API-key probe ──────────────────────────────────────────────────
    # The panel filters its LiteLLM provider bar by which keys are actually
    # present on this server.  We *never* leak the values — just booleans —
    # because the panel only needs "is this provider usable?" semantics.
    #
    # Wildcard providers (OpenRouter / Azure) can route to any underlying
    # model, so if one of them is configured we tell the panel to leave all
    # provider tabs unlocked.
    _PROVIDER_ENV: dict[str, tuple[str, ...]] = {
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        # Ollama is local — no key needed.  Marker stays empty so the panel
        # always shows it (we don't probe the daemon here).
        "ollama": (),
    }
    _WILDCARD_ENV: dict[str, tuple[str, ...]] = {
        "openrouter": ("OPENROUTER_API_KEY",),
        "azure": ("AZURE_API_KEY",),
    }

    async def _send_provider_keys(self, ws: Any) -> None:
        """Emit a ``provider_keys`` snapshot to a single connection."""
        per_provider: dict[str, bool] = {}
        for prov, env_vars in self._PROVIDER_ENV.items():
            per_provider[prov] = (
                True
                if not env_vars  # always-available providers (ollama)
                else any(os.environ.get(v) for v in env_vars)
            )

        wildcards: list[str] = []
        for name, env_vars in self._WILDCARD_ENV.items():
            if any(os.environ.get(v) for v in env_vars):
                wildcards.append(name)

        await ws.send(
            json.dumps(
                {
                    "type": "provider_keys",
                    "providers": per_provider,
                    "wildcards": wildcards,
                }
            )
        )

    async def _send_agent_backends(self, ws: Any) -> None:
        """Emit one ``agent_backends`` snapshot to a single connection."""
        from .agent_backends import probe_all

        await ws.send(
            json.dumps(
                {
                    "type": "agent_backends",
                    "backends": [
                        {
                            "name": s.name,
                            "available": s.available,
                            "state": s.state,
                            "binary_path": s.binary_path,
                            "auth_method": s.auth_method,
                            "detail": s.detail,
                            "can_install": s.can_install,
                        }
                        for s in probe_all()
                    ],
                }
            )
        )

    def _broadcast_agent_backends(self, ws: Any) -> None:
        """Thread-safe re-probe + push, used from background flow callbacks."""
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._send_agent_backends(ws), self._loop)

    def _comfyui_dir(self) -> Path:
        raw = os.environ.get("COMFYUI_DIR", "").strip()
        if raw:
            return Path(raw).expanduser()
        return Path.home() / "Documents" / "ComfyUI"

    async def _handle_local_llm_check(self, ws: Any, msg: dict) -> None:
        from .model_bundles import probe_openai_compatible

        api_base = (
            msg.get("api_base")
            or os.environ.get("COMFYCLAW_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or ""
        ).strip()
        model = (msg.get("model") or os.environ.get("COMFYCLAW_MODEL") or "").strip()
        if not api_base:
            await ws.send(
                json.dumps(
                    {
                        "type": "local_llm_status",
                        "ok": False,
                        "detail": "Missing API base URL.",
                        "models": [],
                        "model": model,
                        "api_base": api_base,
                    }
                )
            )
            return

        loop = asyncio.get_running_loop()
        ok, detail, models = await loop.run_in_executor(
            None, lambda: probe_openai_compatible(api_base, timeout=5)
        )
        expected = model.removeprefix("openai/")
        listed = not expected or expected in models or model in models
        await ws.send(
            json.dumps(
                {
                    "type": "local_llm_status",
                    "ok": ok and listed,
                    "reachable": ok,
                    "model_listed": listed,
                    "detail": detail
                    if listed
                    else f"Endpoint reachable, but {model!r} was not listed.",
                    "models": models,
                    "model": model,
                    "api_base": api_base,
                }
            )
        )

    async def _send_model_bundle_status(self, ws: Any, bundle_name: str) -> None:
        from .model_bundles import MODEL_BUNDLES, bundle_status

        bundle = MODEL_BUNDLES.get(bundle_name)
        if not bundle:
            await ws.send(
                json.dumps(
                    {
                        "type": "model_bundle_status",
                        "ok": False,
                        "bundle": bundle_name,
                        "error": f"Unknown model bundle: {bundle_name}",
                    }
                )
            )
            return
        comfyui_dir = self._comfyui_dir()
        statuses = bundle_status(comfyui_dir, bundle)
        files = [
            {
                "name": mf.dest_name,
                "repo": mf.repo,
                "path": mf.path,
                "target": str(target),
                "optional": mf.optional,
                "exists": exists,
            }
            for mf, target, exists in statuses
        ]
        required_missing = [f for f in files if not f["optional"] and not f["exists"]]
        await ws.send(
            json.dumps(
                {
                    "type": "model_bundle_status",
                    "ok": not required_missing,
                    "bundle": bundle.name,
                    "description": bundle.description,
                    "comfyui_dir": str(comfyui_dir),
                    "files": files,
                    "notes": list(bundle.notes),
                }
            )
        )

    async def _handle_model_bundle_download(self, ws: Any, msg: dict) -> None:
        from .model_bundles import MODEL_BUNDLES, bundle_status, download_model_file

        bundle_name = msg.get("bundle", "")
        include_optional = bool(msg.get("include_optional", False))
        bundle = MODEL_BUNDLES.get(bundle_name)
        if not bundle:
            await ws.send(
                json.dumps(
                    {
                        "type": "model_bundle_download_complete",
                        "bundle": bundle_name,
                        "success": False,
                        "error": f"Unknown model bundle: {bundle_name}",
                    }
                )
            )
            return
        comfyui_dir = self._comfyui_dir()
        statuses = bundle_status(comfyui_dir, bundle)
        selected = [
            (mf, target)
            for mf, target, exists in statuses
            if not exists and (include_optional or not mf.optional)
        ]
        if not selected:
            await ws.send(
                json.dumps(
                    {
                        "type": "model_bundle_download_complete",
                        "bundle": bundle.name,
                        "success": True,
                        "detail": "All selected files are already present.",
                    }
                )
            )
            await self._send_model_bundle_status(ws, bundle.name)
            return

        async def send(msg_out: dict) -> None:
            await ws.send(json.dumps(msg_out))

        async def run_downloads() -> None:
            loop = asyncio.get_running_loop()
            for idx, (mf, target) in enumerate(selected, start=1):
                await send(
                    {
                        "type": "model_bundle_download_progress",
                        "bundle": bundle.name,
                        "index": idx,
                        "total": len(selected),
                        "file": mf.dest_name,
                        "target": str(target),
                        "state": "downloading",
                    }
                )
                try:
                    await loop.run_in_executor(
                        None, lambda mf=mf, target=target: download_model_file(mf, target)
                    )
                except Exception as exc:  # noqa: BLE001
                    await send(
                        {
                            "type": "model_bundle_download_complete",
                            "bundle": bundle.name,
                            "success": False,
                            "error": f"{mf.dest_name}: {exc}",
                        }
                    )
                    await self._send_model_bundle_status(ws, bundle.name)
                    return
            await send(
                {
                    "type": "model_bundle_download_complete",
                    "bundle": bundle.name,
                    "success": True,
                    "detail": "Download complete. Restart ComfyUI so model dropdowns refresh.",
                }
            )
            await self._send_model_bundle_status(ws, bundle.name)

        asyncio.ensure_future(run_downloads())

    async def _handle_model_url_download(self, ws: Any, msg: dict) -> None:
        from .model_bundles import download_model_from_url

        url = str(msg.get("url") or "").strip()
        dest_subdir = str(msg.get("dest_subdir") or "checkpoints").strip()
        filename = str(msg.get("filename") or "").strip() or None
        comfyui_dir = self._comfyui_dir()

        async def send(msg_out: dict) -> None:
            await ws.send(json.dumps(msg_out))

        async def run_download() -> None:
            await send(
                {
                    "type": "model_url_download_progress",
                    "state": "downloading",
                    "url": url,
                    "dest_subdir": dest_subdir,
                }
            )
            try:
                loop = asyncio.get_running_loop()
                target = await loop.run_in_executor(
                    None,
                    lambda: download_model_from_url(
                        url=url,
                        comfyui_dir=comfyui_dir,
                        dest_subdir=dest_subdir,
                        filename=filename,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                await send(
                    {
                        "type": "model_url_download_complete",
                        "success": False,
                        "error": str(exc),
                        "url": url,
                    }
                )
                return
            await send(
                {
                    "type": "model_url_download_complete",
                    "success": True,
                    "target": str(target),
                    "detail": "Download complete. Restart ComfyUI so model dropdowns refresh.",
                }
            )

        asyncio.ensure_future(run_download())

    async def _handle_backend_install_start(self, ws: Any, msg: dict) -> None:
        from .setup_flows import _INSTALL_COMMANDS, CliInstallFlow

        backend = msg.get("backend", "claude-code")
        if backend not in _INSTALL_COMMANDS:
            self._send_json_to(
                ws,
                {
                    "type": "backend_install_complete",
                    "backend": backend,
                    "success": False,
                    "error": f"No installer wired up for {backend!r} yet.",
                },
            )
            return

        existing = self._setup_flows.get(ws)
        if existing and existing.is_running:
            self._send_json_to(
                ws,
                {
                    "type": "backend_install_complete",
                    "backend": backend,
                    "success": False,
                    "error": "Another setup flow is already running for this connection.",
                },
            )
            return

        def on_line(level: str, text: str) -> None:
            self._send_json_to(
                ws,
                {
                    "type": "backend_install_progress",
                    "backend": backend,
                    "level": level,
                    "line": text,
                },
            )

        def on_complete(success: bool, detail: str) -> None:
            self._send_json_to(
                ws,
                {
                    "type": "backend_install_complete",
                    "backend": backend,
                    "success": success,
                    "error": "" if success else detail,
                    "detail": detail,
                },
            )
            self._setup_flows.pop(ws)
            self._broadcast_agent_backends(ws)

        flow = CliInstallFlow(backend, on_line=on_line, on_complete=on_complete)
        prev = self._setup_flows.set(ws, flow)
        if prev:
            try:
                prev.cancel()
            except Exception:  # noqa: BLE001
                pass
        self._send_json_to(
            ws,
            {
                "type": "backend_install_progress",
                "backend": backend,
                "level": "info",
                "line": f"Starting install: {flow.command or '<no candidate on PATH>'}",
            },
        )
        flow.start()

    async def _handle_backend_auth_start(self, ws: Any, msg: dict) -> None:
        from .setup_flows import ClaudeAuthFlow, CodexAuthFlow, GeminiLogoutFlow

        backend = msg.get("backend", "claude-code")
        force = bool(msg.get("force", False))

        # Gemini CLI has no non-TUI auth subcommand — we can only do the
        # logout half (delete cached creds) and ask the user to run `gemini`
        # in a terminal to finish a fresh sign-in.  Only the "force" path
        # (Re-login button) is meaningful for Gemini; a plain "Sign in" with
        # no force flag just tells the user what to do.
        if backend == "gemini-cli":
            if not force:
                self._send_json_to(
                    ws,
                    {
                        "type": "backend_auth_complete",
                        "backend": backend,
                        "success": False,
                        "error": (
                            "Gemini CLI sign-in is interactive. Run `gemini` once "
                            "in a terminal to complete Google OAuth."
                        ),
                    },
                )
                return

            def gemini_on_progress(level: str, text: str) -> None:
                self._send_json_to(
                    ws,
                    {
                        "type": "backend_auth_progress",
                        "backend": backend,
                        "level": level,
                        "message": text,
                    },
                )

            def gemini_on_complete(success: bool, detail: str) -> None:
                self._send_json_to(
                    ws,
                    {
                        "type": "backend_auth_complete",
                        "backend": backend,
                        "success": success,
                        "error": "" if success else detail,
                        "detail": detail,
                    },
                )
                self._setup_flows.pop(ws)
                self._broadcast_agent_backends(ws)

            flow = GeminiLogoutFlow(on_progress=gemini_on_progress, on_complete=gemini_on_complete)
            prev = self._setup_flows.set(ws, flow)
            if prev:
                try:
                    prev.cancel()
                except Exception:  # noqa: BLE001
                    pass
            flow.start()
            return

        if backend not in ("claude-code", "codex"):
            self._send_json_to(
                ws,
                {
                    "type": "backend_auth_complete",
                    "backend": backend,
                    "success": False,
                    "error": f"No auth flow wired up for {backend!r} yet.",
                },
            )
            return

        existing = self._setup_flows.get(ws)
        if existing and existing.is_running:
            self._send_json_to(
                ws,
                {
                    "type": "backend_auth_complete",
                    "backend": backend,
                    "success": False,
                    "error": "Another setup flow is already running for this connection.",
                },
            )
            return

        def on_url(url: str) -> None:
            self._send_json_to(
                ws,
                {"type": "backend_auth_url", "backend": backend, "url": url},
            )

        def on_progress(level: str, text: str) -> None:
            self._send_json_to(
                ws,
                {
                    "type": "backend_auth_progress",
                    "backend": backend,
                    "level": level,
                    "message": text,
                },
            )

        def on_complete(success: bool, detail: str) -> None:
            self._send_json_to(
                ws,
                {
                    "type": "backend_auth_complete",
                    "backend": backend,
                    "success": success,
                    "error": "" if success else detail,
                    "detail": detail,
                },
            )
            self._setup_flows.pop(ws)
            self._broadcast_agent_backends(ws)

        if backend == "claude-code":
            flow = ClaudeAuthFlow(on_url=on_url, on_progress=on_progress, on_complete=on_complete)
        else:  # backend == "codex"
            flow = CodexAuthFlow(on_url=on_url, on_progress=on_progress, on_complete=on_complete)

        prev = self._setup_flows.set(ws, flow)
        if prev:
            try:
                prev.cancel()
            except Exception:  # noqa: BLE001
                pass

        if isinstance(flow, ClaudeAuthFlow):
            auth_method = (msg.get("auth_method") or "claudeai").strip() or "claudeai"
            flow.start(auth_method=auth_method, force=force)
        else:
            # CodexAuthFlow now supports two modes — "browser" (default; uses
            # the local 1455 callback) and "device_code" (headless servers).
            mode = (msg.get("auth_method") or "browser").strip() or "browser"
            flow.start(mode=mode, force=force)

    def _handle_backend_auth_paste(self, ws: Any, msg: dict) -> None:
        from .setup_flows import ClaudeAuthFlow, CodexAuthFlow

        backend = msg.get("backend", "claude-code")
        flow = self._setup_flows.get(ws)

        # CodexAuthFlow uses device-code polling — no paste-back step needed,
        # so quietly ignore a stray paste (don't surface a scary error).
        if isinstance(flow, CodexAuthFlow):
            self._send_json_to(
                ws,
                {
                    "type": "backend_auth_progress",
                    "backend": backend,
                    "level": "info",
                    "message": "Codex sign-in completes automatically once you approve in the browser.",
                },
            )
            return

        if not isinstance(flow, ClaudeAuthFlow):
            self._send_json_to(
                ws,
                {
                    "type": "backend_auth_progress",
                    "backend": backend,
                    "level": "error",
                    "message": "No sign-in flow is running. Click Sign in again.",
                },
            )
            return
        ok, message = flow.submit_redirect_url(msg.get("url", ""))
        if not ok:
            self._send_json_to(
                ws,
                {
                    "type": "backend_auth_progress",
                    "backend": backend,
                    "level": "error",
                    "message": message,
                },
            )

    async def _handle_chat_message(self, websocket: Any, msg: dict) -> None:
        from .chat_agent import chat_stream

        with self._conns_lock:
            conn = self._conns.get(websocket)
        workflow = msg.get("workflow")
        if not isinstance(workflow, dict) or not workflow:
            workflow = copy.deepcopy(conn.workflow) if conn and conn.workflow else None

        message_id: str = msg.get("message_id", "")
        session_id: str = str(msg.get("session_id") or "")
        messages: list = msg.get("messages", [])
        images: list = msg.get("images", [])
        model: str = (msg.get("model") or "").strip() or self._model
        api_key: str | None = (msg.get("api_key") or "").strip() or self._api_key
        api_base: str | None = (msg.get("api_base") or "").strip() or None
        agent_backend: str = (
            (msg.get("agent_backend") or "").strip().lower()
            or os.environ.get("COMFYCLAW_AGENT_BACKEND", "").strip().lower()
            or "litellm"
        )

        # CLI backends must use their own local auth and ignore provider
        # API-key/base overrides sent by the panel.
        if agent_backend in {"claude-code", "codex", "gemini-cli", "grok-cli"}:
            api_key = None
            api_base = None

        try:
            async for token in chat_stream(
                messages,
                workflow,
                model,
                api_key,
                api_base,
                agent_backend=agent_backend,
                images=images,
                session_id=session_id,
                skills_registry=self.skills_registry(),
            ):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "chat_response",
                            "message_id": message_id,
                            "token": token,
                            "done": False,
                        }
                    )
                )
        except Exception as exc:
            await websocket.send(
                json.dumps(
                    {
                        "type": "chat_response",
                        "message_id": message_id,
                        "token": f"\n\n⚠️ Error: {exc}",
                        "done": False,
                    }
                )
            )
        finally:
            await websocket.send(
                json.dumps(
                    {
                        "type": "chat_response",
                        "message_id": message_id,
                        "token": "",
                        "done": True,
                    }
                )
            )

    async def _handle_debug_workflow(self, websocket: Any, msg: dict) -> None:
        from .debug_agent import debug_workflow as _debug

        with self._conns_lock:
            conn = self._conns.get(websocket)

        workflow: dict = msg.get("workflow") or {}
        model: str = (msg.get("model") or "").strip() or self._model
        api_key: str | None = (msg.get("api_key") or "").strip() or self._api_key
        api_base: str | None = (msg.get("api_base") or "").strip() or None
        backend = (msg.get("agent_backend") or "").strip().lower() or "litellm"
        if backend in {"claude-code", "codex", "gemini-cli", "grok-cli"}:
            api_key = None
            api_base = None

        if not workflow and conn and conn.workflow:
            workflow = copy.deepcopy(conn.workflow)

        if not workflow:
            await websocket.send(
                json.dumps(
                    {
                        "type": "debug_result",
                        "summary": "⚠️ No workflow to debug (canvas is empty).",
                        "issues": [],
                        "fixed_workflow": None,
                    }
                )
            )
            return

        await websocket.send(
            json.dumps(
                {
                    "type": "debug_status",
                    "state": "running",
                    "detail": "Analysing workflow…",
                }
            )
        )

        try:
            result = await _debug(workflow, self._server_address, model, api_key, api_base)
        except Exception as exc:
            result = {
                "issues": [],
                "summary": f"⚠️ Debug agent error: {exc}",
                "fixed_workflow": None,
            }

        await websocket.send(
            json.dumps(
                {
                    "type": "debug_result",
                    "summary": result["summary"],
                    "issues": result["issues"],
                    "fixed_workflow": result["fixed_workflow"],
                }
            )
        )

        if result.get("fixed_workflow") and conn:
            conn.workflow = None  # force full snapshot on next broadcast
            self.broadcast(result["fixed_workflow"], target_ws=websocket)

    # ── asyncio internals ────────────────────────────────────────────────────

    @staticmethod
    async def _await_future(fut: asyncio.Future) -> Any:
        return await fut

    async def _send_to(self, ws: Any, payload: str) -> None:
        try:
            await ws.send(payload)
        except Exception as exc:
            log.debug("[SyncServer] Send error: %s", exc)

    async def _async_broadcast(self, payload: str, targets: set[Any]) -> None:
        import websockets.exceptions

        for ws in list(targets):
            try:
                await ws.send(payload)
            except websockets.exceptions.ConnectionClosed:
                with self._conns_lock:
                    self._conns.pop(ws, None)
            except Exception as exc:
                log.debug("[SyncServer] Broadcast error: %s", exc)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        # Create the trigger queue inside the event loop
        self._trigger_queue = asyncio.Queue()
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()
            self._loop = None

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        try:
            async with websockets.server.serve(
                self._handler,
                self.host,
                self.port,
                reuse_port=(os.name != "nt"),
            ):
                log.info("[SyncServer] Listening on ws://%s:%d", self.host, self.port)
                if not self._quiet:
                    print(f"[SyncServer] ✅ Listening on ws://{self.host}:{self.port}")
                self._started_ok = True
                self._ready.set()
                await self._stop_event.wait()
        except Exception as exc:
            log.error("[SyncServer] Failed to start: %s", exc)
            # Errors are always shown.
            print(f"[SyncServer] ❌ Failed to start: {exc}")
            self._ready.set()

    async def _handler(self, websocket: Any) -> None:
        conn = _ConnState(ws=websocket)
        with self._conns_lock:
            self._conns[websocket] = conn
        n = len(self._conns)
        log.debug("[SyncServer] Client connected (%d total)", n)

        # Bootstrap: send this connection's current workflow snapshot
        if conn.workflow is not None:
            try:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "workflow_update",
                            "workflow": conn.workflow,
                        }
                    )
                )
            except Exception:
                pass

        # Also push the skills manifest proactively so the Skills tab is
        # populated even if the user hasn't clicked into it. Without this
        # the panel only fetches on tab-activate and looks empty otherwise.
        try:
            await self._send_skills_manifest(websocket)
        except Exception:
            pass

        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        continue
                    await self._dispatch(websocket, conn, msg)
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass
        finally:
            with self._conns_lock:
                self._conns.pop(websocket, None)
            with self._active_ws_lock:
                if self._active_ws is websocket:
                    self._active_ws = None
            # Reap any in-flight backend setup flow so we don't orphan a
            # subprocess waiting on stdin from a panel that's no longer there.
            try:
                self._setup_flows.cancel_for(websocket)
            except Exception:  # noqa: BLE001
                pass
            log.debug("[SyncServer] Client disconnected (%d remaining)", len(self._conns))

    async def _dispatch(self, ws: Any, conn: _ConnState, msg: dict) -> None:
        """Route a single inbound message to the right handler."""
        t = msg.get("type")

        if t == "hello":
            conn.connection_id = msg.get("connection_id", "")
            session_id = str(msg.get("session_id") or "")
            log.debug("[SyncServer] hello from connection_id=%r", conn.connection_id)
            # Send current checkpoint list for this connection
            await ws.send(
                json.dumps(
                    {
                        "type": "checkpoints_list",
                        "checkpoints": conn.list_checkpoints(session_id=session_id),
                        "session_id": session_id,
                    }
                )
            )

        elif t == "trigger_generation":
            log.info("[SyncServer] trigger_generation from %r", conn.connection_id)
            conn.cancel.clear()
            assert self._trigger_queue is not None
            await self._trigger_queue.put((msg, ws))

        elif t == "cancel_generation":
            log.info("[SyncServer] cancel_generation from %r", conn.connection_id)
            conn.cancel.set()

        elif t == "human_feedback":
            if conn.feedback_fut and not conn.feedback_fut.done():
                conn.feedback_fut.set_result(msg)
                conn.feedback_fut = None

        elif t == "user_refinement":
            if conn.refinement_fut and not conn.refinement_fut.done():
                conn.refinement_fut.set_result(msg)
                conn.refinement_fut = None

        elif t == "model_download_decision":
            if conn.model_download_fut and not conn.model_download_fut.done():
                conn.model_download_fut.set_result(msg)
                conn.model_download_fut = None

        elif t == "generation_compute_decision":
            if conn.compute_confirm_fut and not conn.compute_confirm_fut.done():
                conn.compute_confirm_fut.set_result(msg)
                conn.compute_confirm_fut = None

        elif t == "chat_message":
            asyncio.ensure_future(self._handle_chat_message(ws, msg))

        elif t == "debug_workflow":
            asyncio.ensure_future(self._handle_debug_workflow(ws, msg))

        elif t == "save_checkpoint":
            workflow = msg.get("workflow") or conn.workflow or {}
            label = msg.get("label", "")
            session_id = str(msg.get("session_id") or "")
            cp_id = conn.save_checkpoint(workflow, label, session_id=session_id)
            await ws.send(
                json.dumps(
                    {
                        "type": "checkpoint_saved",
                        "id": cp_id,
                        "label": label or "Snapshot",
                        "session_id": session_id,
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "checkpoints_list",
                        "checkpoints": conn.list_checkpoints(session_id=session_id),
                        "session_id": session_id,
                    }
                )
            )

        elif t == "restore_checkpoint":
            cp_id = msg.get("id", "")
            session_id = str(msg.get("session_id") or "")
            wf = conn.restore_checkpoint(cp_id, session_id=session_id)
            ok = wf is not None
            await ws.send(
                json.dumps(
                    {
                        "type": "checkpoint_restored",
                        "id": cp_id,
                        "success": ok,
                        "session_id": session_id,
                    }
                )
            )
            if ok:
                conn.workflow = None  # force full snapshot
                self.broadcast(wf, target_ws=ws)

        elif t == "list_checkpoints":
            session_id = str(msg.get("session_id") or "")
            await ws.send(
                json.dumps(
                    {
                        "type": "checkpoints_list",
                        "checkpoints": conn.list_checkpoints(session_id=session_id),
                        "session_id": session_id,
                    }
                )
            )

        elif t == "accept_now":
            log.info("[SyncServer] accept_now from %r", conn.connection_id)
            conn.accept_now.set()

        # ── Skills CRUD ──────────────────────────────────────────────────────
        elif t == "list_skills":
            await self._send_skills_manifest(ws)

        elif t == "read_skill_body":
            try:
                reg = self.skills_registry()
                body = reg.get_body(msg.get("name", ""))
                props = reg.get_properties(msg.get("name", ""))
                await ws.send(
                    json.dumps(
                        {
                            "type": "skill_body",
                            "name": msg.get("name", ""),
                            "body": body,
                            "description": props.description,
                            "location": str(props.location),
                            "license": getattr(props, "license", "") or "",
                        }
                    )
                )
            except KeyError:
                await self._send_skill_error(ws, "Skill not found.")

        elif t == "set_skill_enabled":
            try:
                self.skills_registry().set_enabled(
                    msg.get("name", ""), bool(msg.get("enabled", True))
                )
                await self._send_skills_manifest(ws)
            except Exception as exc:  # noqa: BLE001
                await self._send_skill_error(ws, f"Could not update skill: {exc}")

        elif t == "import_skill_folder":
            asyncio.ensure_future(self._import_skill_folder(ws, msg))

        elif t == "import_skill_zip":
            asyncio.ensure_future(self._import_skill_zip(ws, msg))

        elif t == "import_skill_git":
            asyncio.ensure_future(self._import_skill_git(ws, msg))

        elif t == "delete_skill":
            try:
                self.skills_registry().delete(msg.get("name", ""))
                await self._send_skills_manifest(ws)
                await ws.send(
                    json.dumps(
                        {
                            "type": "skill_import_result",
                            "ok": True,
                            "action": "delete",
                            "name": msg.get("name", ""),
                        }
                    )
                )
            except (KeyError, PermissionError, OSError) as exc:
                await self._send_skill_error(ws, f"Delete failed: {exc}")

        elif t == "reload_skills":
            self.reload_skills()
            await self._send_skills_manifest(ws)

        elif t == "refine_skill_preview":
            asyncio.ensure_future(self._refine_skill_preview(ws, msg))

        elif t == "apply_skill_refine":
            asyncio.ensure_future(self._apply_skill_refine(ws, msg))

        elif t == "apply_skill_evolution":
            approved = bool(msg.get("approved", False))
            if conn.skill_evolution_fut and not conn.skill_evolution_fut.done():
                # ``proposal`` carries any human edits made in the review modal.
                conn.skill_evolution_fut.set_result(
                    {"approved": approved, "proposal": msg.get("proposal")}
                )
            else:
                await self._send_skill_error(ws, "No pending skill evolution proposal.")

        # ── Agent backend availability probe ─────────────────────────────────
        elif t == "list_agent_backends":
            await self._send_agent_backends(ws)

        # ── Provider API-key probe (LiteLLM filtering) ───────────────────────
        elif t == "list_provider_keys":
            await self._send_provider_keys(ws)

        # ── Local setup helpers used by the ComfyUI panel ───────────────────
        elif t == "local_llm_check":
            asyncio.ensure_future(self._handle_local_llm_check(ws, msg))

        elif t == "model_bundle_check":
            await self._send_model_bundle_status(ws, msg.get("bundle", ""))

        elif t == "model_bundle_download":
            await self._handle_model_bundle_download(ws, msg)

        elif t == "model_url_download":
            await self._handle_model_url_download(ws, msg)

        # ── Backend setup flows (install + OAuth) ────────────────────────────
        elif t == "backend_install_start":
            await self._handle_backend_install_start(ws, msg)

        elif t == "backend_install_cancel":
            self._setup_flows.cancel_for(ws)
            self._send_json_to(
                ws,
                {
                    "type": "backend_install_complete",
                    "backend": msg.get("backend", "claude-code"),
                    "success": False,
                    "error": "cancelled",
                },
            )

        elif t == "backend_auth_start":
            await self._handle_backend_auth_start(ws, msg)

        elif t == "backend_auth_paste_code":
            self._handle_backend_auth_paste(ws, msg)

        elif t == "backend_auth_cancel":
            self._setup_flows.cancel_for(ws)
            self._send_json_to(
                ws,
                {
                    "type": "backend_auth_complete",
                    "backend": msg.get("backend", "claude-code"),
                    "success": False,
                    "error": "cancelled",
                },
            )
