"""
ClawHarness — top-level orchestrator for the self-evolving generate–verify loop.

Usage (context-manager)::

    cfg = HarnessConfig(api_key="sk-ant-...", max_iterations=4)
    with ClawHarness.from_workflow_file("examples/workflows/sd15_dreamshaper_lcm.json", cfg) as h:
        image_bytes = h.run("a red fox at dawn, photorealistic")

Topology accumulation
---------------------
When ``evolve_from_best=True`` (the default) each iteration starts from the
**best workflow snapshot** produced so far rather than resetting to the
original base workflow.  This means LoRA / ControlNet nodes added in round 1
persist into round 2, and the agent only needs to add *incremental* upgrades.
Set ``evolve_from_best=False`` to revert to the old reset-each-iteration
behaviour.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .agent import ClawAgent
from .client import ComfyClient
from .memory import ClawMemory
from .skill_evolver import SkillEvolutionProposal, SkillEvolver
from .sync_server import SyncServer
from .verifier import ClawVerifier, VerifierResult
from .workflow import WorkflowManager

# Error messages that indicate a transient infrastructure fault in ComfyUI
# (broken pipe from tqdm/progress-bar writing to a closed stderr, etc.).
# These are NOT workflow logic errors — the agent should not attempt a repair;
# instead, harness should retry the same workflow after a short pause.
_INFRA_ERROR_SIGNALS = (
    "[Errno 5] Input/output error",
    "[Errno 32] Broken pipe",
    "BrokenPipeError",
    "app/logger.py",
    "status_printer",
    "sys.stderr",
    "tqdm",
)

_StatusCallback = Callable[[str, int, str], None]

log = logging.getLogger(__name__)


def resolve_verifier_model(
    verifier_model: str | None,
    agent_model: str,
    agent_backend: str,
) -> str:
    """Resolve raw verifier model to a valid LiteLLM vision model matching the provider."""
    vmodel = (verifier_model or "").strip()
    backend = (agent_backend or "litellm").strip().lower().replace("_", "-")
    amodel = (agent_model or "").strip()

    backend_defaults = {
        "claude-code": "anthropic/claude-3-5-sonnet",
        "codex": "openai/gpt-5.4",
        "gemini-cli": "gemini/gemini-2.0-flash",
    }

    # 1. If the verifier model is not explicitly set, prefer a backend-local
    # default so the evaluator stays in the same provider family as the agent.
    if not vmodel and backend in backend_defaults:
        return backend_defaults[backend]

    # 2. Determine the raw model string we want to evaluate.
    raw_model = vmodel if vmodel else amodel

    # 3. Fall back to provider-specific defaults if empty.
    if not raw_model:
        return backend_defaults.get(backend, "anthropic/claude-sonnet-4-5")

    raw_lower = raw_model.lower()

    # 4. Check if the model indicates Anthropic
    if backend == "claude-code" or "claude" in raw_lower or "anthropic" in raw_lower:
        if "opus" in raw_lower:
            return "anthropic/claude-3-opus"
        if "haiku" in raw_lower:
            return "anthropic/claude-3-haiku"
        if "claude-3-5-sonnet" in raw_lower or "claude-sonnet-4-5" in raw_lower:
            return raw_model if "/" in raw_model else f"anthropic/{raw_model}"
        return "anthropic/claude-3-5-sonnet"

    # 5. Check if the model indicates OpenAI / Codex
    elif "gpt" in raw_lower or "openai" in raw_lower or raw_lower in ("o3", "o3-mini", "o4-mini"):
        if "gpt-4o-mini" in raw_lower or "o4-mini" in raw_lower:
            return "openai/gpt-5.4-mini"
        if "gpt-4" in raw_lower:
            if "/" in raw_model:
                return raw_model
            return f"openai/{raw_model}"
        return "openai/gpt-5.4"

    # 6. Check if the model indicates Gemini
    elif "gemini" in raw_lower or "google" in raw_lower:
        if "/" in raw_model and (
            "gemini-2.0" in raw_lower or "gemini-2.5" in raw_lower or "gemini-1.5" in raw_lower
        ):
            return raw_model
        return "gemini/gemini-2.0-flash"

    # 7. Fallback: return as-is
    return raw_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """
    All tuning knobs for a ``ClawHarness`` run in one place.

    Parameters
    ----------
    api_key               : Anthropic API key (required).
    server_address        : ComfyUI HTTP address, e.g. ``"127.0.0.1:8188"``.
    model                 : Claude model for both agent and verifier.
    max_iterations        : Maximum agent–generate–verify cycles.
    success_threshold     : Stop early when verifier score reaches this value.
    sync_port             : WebSocket port for live UI sync; 0 to disable.
    skills_dir            : Path to SKILL.md directory; ``None`` uses built-in skills.
    evolve_from_best      : Start each iteration from the best previous workflow.
    max_images            : Max images kept in RAM across attempts (see ClawMemory).
    score_weights         : ``(req_weight, detail_weight)`` for verifier score blend.
    image_model           : Pin the ComfyUI checkpoint / UNET to this name.
                            Must be the **exact filename** as reported by ComfyUI
                            (e.g. ``"qwen_image_2512_fp8_e4m3fn.safetensors"``),
                            not a HuggingFace-style path.  ``None`` leaves the
                            workflow's existing model untouched.
    max_repair_attempts   : When ComfyUI rejects a workflow (HTTP 4xx / execution
                            error), the agent gets up to this many chances to
                            inspect the error and fix the topology before the
                            iteration is abandoned.  Set to 0 to disable repairs.
    """

    api_key: str = ""
    server_address: str = "127.0.0.1:8188"
    model: str = "anthropic/claude-sonnet-4-5"
    verifier_model: str | None = None
    max_iterations: int = 3
    success_threshold: float = 0.85
    sync_port: int = 8765
    skills_dir: str | None = None
    evolve_from_best: bool = True
    max_images: int = 5
    score_weights: tuple[float, float] = field(default_factory=lambda: (0.6, 0.4))
    image_model: str | None = None
    max_repair_attempts: int = 2
    verifier_mode: str = "vlm"  # "vlm", "human", or "hybrid"
    api_base: str | None = None
    agent_backend: str = "litellm"  # "litellm" | "claude-code" | "codex" | "gemini-cli"
    run_mode: str = "auto"  # "manual" | "auto" | "copilot"
    modality: str = "image"  # "image" | "video"
    video_frames: int = 6  # frames sampled per clip for the verifier
    generation_timeout: int = 0  # seconds; 0 = image/video defaults
    enable_skill_evolution: bool = True
    skill_evolution_min_confidence: float = 0.55
    skill_evolution_auto_apply: bool = False
    agent_session_id: str = ""
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """
    Pin the image-generation model (checkpoint / UNET) used by ComfyUI.

    When set, this model name is written into every loader node
    (``CheckpointLoaderSimple``, ``UNETLoader``, etc.) in the workflow
    at startup and after each topology evolution, so the agent cannot
    accidentally swap it out.

    Examples::

        image_model = "qwen_image_2512_fp8_e4m3fn.safetensors"  # exact ComfyUI filename
        image_model = "realisticVisionV51.safetensors"           # local checkpoint
        image_model = None   # do not override — use whatever the workflow has
    """

    def __post_init__(self) -> None:
        self.verifier_model = resolve_verifier_model(
            self.verifier_model,
            self.model,
            self.agent_backend,
        )


# ---------------------------------------------------------------------------
# Evolution log
# ---------------------------------------------------------------------------


@dataclass
class EvolutionEntry:
    iteration: int
    node_count_before: int
    node_count_after: int
    node_ids_added: list[str]
    rationale: str
    verifier_score: float | None = None

    def summary(self) -> str:
        diff = self.node_count_after - self.node_count_before
        sign = "+" if diff >= 0 else ""
        added = ", ".join(self.node_ids_added) or "none"
        return (
            f"  Iter {self.iteration}: nodes {self.node_count_before}→{self.node_count_after} "
            f"({sign}{diff}), added=[{added}], score={self.verifier_score}"
        )


class EvolutionLog:
    def __init__(self) -> None:
        self.entries: list[EvolutionEntry] = []

    def record(self, entry: EvolutionEntry) -> None:
        self.entries.append(entry)

    def format(self) -> str:
        if not self.entries:
            return "  (no entries yet)"
        return "\n".join(e.summary() for e in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class ComputeRisk:
    ok: bool
    reason: str
    required_vram_gb: float
    available_vram_gb: float | None
    device: str
    workload: str

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "required_vram_gb": self.required_vram_gb,
            "available_vram_gb": self.available_vram_gb,
            "device": self.device,
            "workload": self.workload,
        }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class ClawHarness:
    """
    Orchestrates the ClawAgent → ComfyUI → ClawVerifier self-evolving loop.

    Prefer constructing via ``from_workflow_file`` or ``from_workflow_dict``.
    Use as a context manager to ensure the sync server is stopped cleanly.
    """

    def __init__(self, base_workflow: dict, config: HarnessConfig) -> None:
        self.base_workflow = copy.deepcopy(base_workflow)
        self.config = config

        # Apply any pinned image model to the base workflow immediately,
        # so it is the starting point for every iteration.
        if config.image_model:
            # Warn early if the value looks like a HuggingFace path rather than
            # a ComfyUI filename — ComfyUI only accepts exact filenames and will
            # return HTTP 400 otherwise.
            im = config.image_model
            if "/" in im and not any(
                im.endswith(ext) for ext in (".safetensors", ".ckpt", ".pt", ".gguf", ".bin")
            ):
                log.warning(
                    "image_model=%r looks like a HuggingFace path, not a ComfyUI filename. "
                    "ComfyUI requires the exact local filename (e.g. 'qwen_image_2512_fp8_e4m3fn.safetensors'). "
                    "The workflow submission will likely fail with HTTP 400.",
                    im,
                )
            wm = WorkflowManager(self.base_workflow)
            updated = wm.apply_image_model(config.image_model)
            self.base_workflow = wm.workflow
            if updated:
                log.info(
                    "Pinned image model %r on %d loader node(s): %s",
                    config.image_model,
                    len(updated),
                    ", ".join(f"[{nid}].{p}" for nid, p in updated),
                )
            else:
                log.warning(
                    "image_model=%r set but no loader nodes found in workflow; "
                    "the model pin will have no effect.",
                    config.image_model,
                )

        self._client = ComfyClient(config.server_address)
        self._sync = SyncServer(port=config.sync_port) if config.sync_port else None
        self._memory = ClawMemory(max_images=config.max_images)
        self._evolution_log = EvolutionLog()

        self.on_status: _StatusCallback | None = None

        self._agent = ClawAgent(
            api_key=config.api_key,
            model=config.model,
            server_address=config.server_address,
            skills_dir=config.skills_dir,
            on_change=self._on_workflow_change,
            pinned_image_model=config.image_model,
            backend_name=config.agent_backend,
            api_base=config.api_base,
            agent_session_id=config.agent_session_id,
            model_download_callback=self._request_model_download,
        )
        self._agent.on_agent_event = self._on_agent_event
        self._current_iteration = 0

        if (config.agent_backend or "").strip().lower().replace("_", "-") == "grok-cli":
            # Grok subscription auth only covers the agent CLI. The existing
            # image verifier would silently use LiteLLM and an API key.
            # Generate once and return the image without an API verifier.
            self._verifier = None
            log.info("Grok CLI subscription: image verification skipped (no API fallback)")
            return

        if config.modality == "video":
            from .video_verifier import VideoVerifier

            vlm_verifier = VideoVerifier(
                api_key=config.api_key,
                model=config.verifier_model or config.model,
                score_weights=config.score_weights,
                n_frames=config.video_frames,
            )
        else:
            vlm_verifier = ClawVerifier(
                api_key=config.api_key,
                model=config.verifier_model or config.model,
                score_weights=config.score_weights,
                backend=config.agent_backend,
            )

        # run_mode is the user-facing knob; verifier_mode is derived.
        run_mode = (config.run_mode or "auto").lower()
        if run_mode == "manual":
            # Manual mode: do not call any verifier, single round.
            self._verifier = None  # type: ignore[assignment]
            log.info("Run mode: manual (no verifier, single round)")
        elif run_mode == "copilot" or config.verifier_mode == "hybrid":
            from .human_verifier import HybridVerifier

            self._verifier = HybridVerifier(
                vlm_verifier=vlm_verifier,
                sync_server=self._sync,
                timeout=600.0,
            )
            log.info("Run mode: copilot (VLM + human override)")
        elif config.verifier_mode == "human":
            from .human_verifier import HumanVerifier

            self._verifier = HumanVerifier(
                sync_server=self._sync,
                timeout=600.0,
            )
            log.info("Verifier mode: human")
        else:
            self._verifier = vlm_verifier
            log.info("Run mode: auto (VLM verifier)")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ClawHarness:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._sync:
            self._sync.start()

    def stop(self) -> None:
        if self._sync:
            self._sync.stop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, prompt: str, dry_run: bool = False) -> bytes | None:
        """
        Run the self-evolving generate–verify loop.

        Parameters
        ----------
        prompt  : Image generation prompt.
        dry_run : If ``True``, skip actual ComfyUI execution.

        Returns
        -------
        Raw image bytes of the best result, or ``None`` on failure / dry-run.
        """
        cfg = self.config
        log.info("Starting run: %r", prompt)
        print(f"\n{'=' * 60}")
        print(f"[ClawHarness] Run: {prompt!r}")
        print(f"{'=' * 60}")

        self._memory.clear()
        self._evolution_log = EvolutionLog()
        best_image: bytes | None = None
        best_score = -1.0
        best_workflow_snapshot: dict | None = None
        last_result: VerifierResult | None = None

        # Reset the connection-scoped accept_now flag on every run.
        if self._sync:
            ws = getattr(self, "_sync_ws", None)
            try:
                with self._sync._conns_lock:
                    conn = (
                        self._sync._conns.get(ws)
                        if ws
                        else next(iter(self._sync._conns.values()), None)
                    )
                if conn is not None:
                    conn.accept_now.clear()
            except Exception:
                pass

        for iteration in range(1, cfg.max_iterations + 1):
            self._current_iteration = iteration
            print(f"\n--- Iteration {iteration}/{cfg.max_iterations} ---")
            self._emit_status("running", iteration, f"Iteration {iteration}/{cfg.max_iterations}")

            # ── Choose starting workflow ───────────────────────────────────
            if cfg.evolve_from_best and best_workflow_snapshot is not None:
                start_wf = copy.deepcopy(best_workflow_snapshot)
            else:
                start_wf = copy.deepcopy(self.base_workflow)

            wm = WorkflowManager(start_wf)

            # Prepare sync: reset to empty so every subsequent broadcast
            # produces add_node diffs (not a single full snapshot).
            _ws = getattr(self, "_sync_ws", None)
            if self._sync:
                self._sync.reset(target_ws=_ws, empty=True)
                self._sync.enable_refinement_listening(source_ws=_ws)

            # Broadcast base workflow nodes one-by-one so the ComfyUI
            # canvas shows them appearing incrementally.
            if self._sync and start_wf:
                partial: dict = {}
                for nid in sorted(start_wf.keys(), key=lambda k: int(k)):
                    partial[nid] = copy.deepcopy(start_wf[nid])
                    self._sync.broadcast(copy.deepcopy(partial), target_ws=_ws)

            node_ids_before = set(wm.workflow.keys())

            # ── Agent evolves the workflow ─────────────────────────────────
            verifier_feedback = self._build_feedback(last_result)

            # Check for user refinement messages sent from the thinking panel
            user_refinement = self._poll_user_refinement()
            if user_refinement:
                prefix = f"[User refinement request]: {user_refinement}\n\n"
                verifier_feedback = prefix + (verifier_feedback or "")
                print(f"[ClawHarness] 👤 User refinement: {user_refinement[:100]}")

            memory_summary = (
                self._memory.format_history_for_agent() if self._memory.attempts else None
            )
            conversation_summary = self._format_conversation_history()
            if conversation_summary and memory_summary:
                memory_summary = f"{conversation_summary}\n\n{memory_summary}"
            elif conversation_summary:
                memory_summary = conversation_summary

            print("[ClawHarness] 🤖 Agent is evolving the workflow…")
            rationale = self._agent.plan_and_patch(
                workflow_manager=wm,
                original_prompt=prompt,
                verifier_feedback=verifier_feedback,
                memory_summary=memory_summary,
                iteration=iteration,
            )

            direct_answer = getattr(self._agent, "last_direct_answer", "")
            if isinstance(direct_answer, str) and direct_answer.strip():
                print("[ClawHarness] 💬 Agent answered without workflow changes.")
                self._emit_status("complete", iteration, "Answered.")
                return None

            node_ids_after = set(wm.workflow.keys())
            added_ids = sorted(node_ids_after - node_ids_before)
            evo = EvolutionEntry(
                iteration=iteration,
                node_count_before=len(node_ids_before),
                node_count_after=len(node_ids_after),
                node_ids_added=added_ids,
                rationale=rationale,
            )
            if added_ids:
                new_classes = [wm.workflow[nid].get("class_type", "?") for nid in added_ids]
                print(f"[ClawHarness] 🔧 Added nodes {added_ids} → {new_classes}")

            # Re-apply pinned model after agent evolution — the agent may have
            # added new loader nodes (e.g. a LoRA) and we must ensure the
            # primary checkpoint / UNET still points at the configured model.
            if cfg.image_model:
                wm.apply_image_model(cfg.image_model)

            self._on_workflow_change(wm.workflow)

            # ── Dry-run mode ───────────────────────────────────────────────
            if dry_run:
                print("[ClawHarness] ⏭  dry_run=True — skipping ComfyUI execution.")
                print(f"[ClawHarness] Final workflow ({len(wm)} nodes):")
                print(json.dumps(wm.workflow, indent=2)[:3000])
                self._evolution_log.record(evo)
                return None

            risk = self._compute_generation_risk(wm.workflow)
            if not risk.ok:
                print(f"[ClawHarness] ⚠  Compute risk: {risk.reason}")
                self._emit_status("awaiting_confirmation", iteration, risk.reason)
                if not self._confirm_generation_compute_risk(risk):
                    print("[ClawHarness] ⏭  Generation skipped after compute warning.")
                    self._emit_status(
                        "dry_run_done",
                        iteration,
                        "Workflow built. Generation skipped after compute warning.",
                    )
                    self._evolution_log.record(evo)
                    return None

            # ── Submit with repair loop ────────────────────────────────────
            # When ComfyUI rejects a workflow (HTTP 4xx / execution error),
            # the agent gets up to cfg.max_repair_attempts chances to inspect
            # the error message and fix the topology before this iteration is
            # abandoned.
            prompt_id: str | None = None
            submission_error: str | None = None

            for repair_round in range(cfg.max_repair_attempts + 1):
                label = (
                    "Submitting"
                    if repair_round == 0
                    else f"Repair {repair_round}/{cfg.max_repair_attempts}"
                )
                print(f"[ClawHarness] 🚀 {label} to ComfyUI…")
                if repair_round > 0:
                    self._emit_status("repairing", iteration, f"Repair attempt {repair_round}")

                # On repair rounds let the agent fix the workflow in-place.
                if repair_round > 0:
                    repair_feedback = self._build_repair_feedback(submission_error, last_result)
                    self._agent.plan_and_patch(
                        workflow_manager=wm,
                        original_prompt=prompt,
                        verifier_feedback=repair_feedback,
                        iteration=iteration,
                    )
                    if cfg.image_model:
                        wm.apply_image_model(cfg.image_model)
                    self._on_workflow_change(wm.workflow)

                try:
                    queue_resp = self._client.queue_prompt(wm.workflow)
                    prompt_id = queue_resp["prompt_id"]
                    submission_error = None
                    if repair_round > 0:
                        print(f"[ClawHarness] ✅ Repair {repair_round} accepted by ComfyUI.")
                    break
                except Exception as exc:
                    submission_error = str(exc)
                    print(
                        f"[ClawHarness] ❌ {'Repair' if repair_round > 0 else 'Queue'} error: {exc}"
                    )

            if prompt_id is None:
                self._record_error(
                    iteration, wm.workflow, submission_error or "unknown queue error"
                )
                self._evolution_log.record(evo)
                continue

            # ── Wait for completion ────────────────────────────────────────
            timeout = self._generation_timeout()
            try:
                history = self._client.wait_for_completion(prompt_id, timeout=timeout)
            except TimeoutError as exc:
                print(f"[ClawHarness] ❌ Timeout: {exc}")
                self._record_error(iteration, wm.workflow, str(exc))
                self._evolution_log.record(evo)
                continue

            # ── Handle ComfyUI execution-time error ────────────────────────
            if "error" in history:
                exec_error = history["error"]
                print(f"[ClawHarness] ❌ ComfyUI execution error: {exec_error}")

                # ── Infra fault (BrokenPipe from tqdm stderr) — not a workflow bug
                # Retry the SAME workflow once after a short pause; do NOT ask the
                # agent to repair anything.
                if self._is_infra_error(history):
                    print(
                        "[ClawHarness] ⚠  Transient infrastructure error detected "
                        "(progress-bar / stderr flush). Waiting 5 s then "
                        "retrying the same workflow once."
                    )
                    time.sleep(5)
                    try:
                        rq_retry = self._client.queue_prompt(wm.workflow)
                        retry_pid = rq_retry["prompt_id"]
                        print(f"[ClawHarness] 🔄 Infra-retry submitted ({retry_pid}).")
                        history = self._client.wait_for_completion(retry_pid, timeout=timeout)
                    except Exception as infra_exc:
                        print(f"[ClawHarness] ❌ Infra-retry exception: {infra_exc}")
                        self._record_error(iteration, wm.workflow, str(infra_exc))
                        self._evolution_log.record(evo)
                        last_result = None
                        continue

                    if "error" in history:
                        infra_msg = history["error"]
                        print(f"[ClawHarness] ❌ Infra-retry also failed: {infra_msg}")
                        print(
                            "[ClawHarness]    This usually means the ComfyUI process was "
                            "started with a closed stderr/stdout. Restart ComfyUI with "
                            "output redirected to a real log file, then retry."
                        )
                        self._record_error(iteration, wm.workflow, infra_msg)
                        self._evolution_log.record(evo)
                        last_result = None
                        continue
                    # Retry succeeded — fall through to image collection below.
                    print("[ClawHarness] ✅ Infra-retry succeeded.")

                else:
                    # ── Workflow-logic error — let the agent repair the topology
                    repaired_prompt_id: str | None = None
                    exec_submission_error: str | None = exec_error

                    for repair_round in range(1, cfg.max_repair_attempts + 1):
                        print(
                            f"[ClawHarness] 🔧 Execution repair {repair_round}/{cfg.max_repair_attempts}…"
                        )
                        repair_feedback = self._build_repair_feedback(
                            exec_submission_error, last_result
                        )
                        self._agent.plan_and_patch(
                            workflow_manager=wm,
                            original_prompt=prompt,
                            verifier_feedback=repair_feedback,
                            iteration=iteration,
                        )
                        if cfg.image_model:
                            wm.apply_image_model(cfg.image_model)
                        self._on_workflow_change(wm.workflow)

                        try:
                            rq = self._client.queue_prompt(wm.workflow)
                            repaired_prompt_id = rq["prompt_id"]
                            exec_submission_error = None
                            print(f"[ClawHarness] ✅ Execution repair {repair_round} accepted.")
                            break
                        except Exception as exc2:
                            exec_submission_error = str(exc2)
                            print(
                                f"[ClawHarness] ❌ Execution repair {repair_round} failed: {exc2}"
                            )

                    if repaired_prompt_id is None:
                        self._record_error(
                            iteration, wm.workflow, exec_submission_error or exec_error
                        )
                        self._evolution_log.record(evo)
                        last_result = None
                        continue

                    # Wait for the repaired workflow to finish.
                    try:
                        history = self._client.wait_for_completion(
                            repaired_prompt_id, timeout=timeout
                        )
                    except TimeoutError as exc:
                        print(f"[ClawHarness] ❌ Timeout after repair: {exc}")
                        self._record_error(iteration, wm.workflow, str(exc))
                        self._evolution_log.record(evo)
                        continue

                    if "error" in history:
                        msg = history["error"]
                        print(f"[ClawHarness] ❌ ComfyUI error after repair: {msg}")
                        self._record_error(iteration, wm.workflow, msg)
                        self._evolution_log.record(evo)
                        last_result = None
                        continue

            if cfg.modality == "video":
                videos = self._client.collect_videos(history)
                if not videos:
                    print("[ClawHarness] ⚠  No videos in output — check workflow.")
                    self._evolution_log.record(evo)
                    continue
                image_bytes, _media_type = videos[0]
                print(f"[ClawHarness] 🎬 Got video ({len(image_bytes):,} bytes, {_media_type})")
            else:
                images = self._client.collect_images(history)
                if not images:
                    print("[ClawHarness] ⚠  No images in output — check workflow.")
                    self._evolution_log.record(evo)
                    continue
                image_bytes = images[0]
                print(f"[ClawHarness] 🖼  Got image ({len(image_bytes):,} bytes)")

            # ── Verify (skipped in manual mode) ───────────────────────────
            if self._verifier is None:
                # Manual: no verifier. Always treat the latest image as best.
                if image_bytes is not None:
                    best_image = image_bytes
                    best_workflow_snapshot = wm.to_dict()
                self._evolution_log.record(evo)
                # Emit a manual-mode scoreboard event so the panel can show
                # an inline "completed" card without a score.
                if self._sync:
                    self._sync.send_iteration_score(
                        iteration=iteration,
                        score=None,
                        delta=None,
                        critique="Manual mode — no verifier ran.",
                        image_path="",
                        target_ws=getattr(self, "_sync_ws", None),
                    )
                break

            print("[ClawHarness] 🔍 Verifying image…")
            self._emit_status("verifying", iteration, "Verifying image…")
            result = self._verifier.verify(image_bytes, prompt, iteration=iteration)
            result = self._collect_user_feedback_after_generation(
                result,
                image_bytes,
                prompt,
                iteration,
            )
            prev_score = last_result.score if last_result is not None else None
            last_result = result
            print(f"[ClawHarness] Score: {result.score:.2f}")
            print(result.format_feedback())

            if result.score > best_score:
                best_score = result.score
                best_image = image_bytes
                best_workflow_snapshot = wm.to_dict()

            evo.verifier_score = result.score
            self._evolution_log.record(evo)

            # ── Emit live scoreboard event ────────────────────────────────
            if self._sync:
                delta = None if prev_score is None else (result.score - prev_score)
                critique = (result.overall_assessment or "").strip()
                self._sync.send_iteration_score(
                    iteration=iteration,
                    score=result.score,
                    delta=delta,
                    critique=critique,
                    image_path="",
                    target_ws=getattr(self, "_sync_ws", None),
                )

            # ── Record in memory ──────────────────────────────────────────
            feedback_meta = self._feedback_metadata(result)
            experience = self._summarize_experience(
                prompt,
                result.passed,
                result.failed,
                rationale,
                feedback_meta,
            )
            self._memory.record(
                iteration=iteration,
                workflow_snapshot=wm.to_dict(),
                verifier_score=result.score,
                passed=result.passed,
                failed=result.failed,
                experience=experience,
                image_bytes=image_bytes,
                feedback_rating=feedback_meta["rating"],
                feedback_comment=feedback_meta["comment"],
                feedback_case=feedback_meta["case"],
                evolve_requested=feedback_meta["evolve"],
            )

            # ── User accept-now early stop ────────────────────────────────
            if self._sync and self._sync.accept_requested(
                target_ws=getattr(self, "_sync_ws", None)
            ):
                print("[ClawHarness] ✋ User accepted current result early — stopping.")
                break

            # ── Score-threshold early stop ────────────────────────────────
            if result.score >= cfg.success_threshold:
                print(
                    f"[ClawHarness] ✅ Score {result.score:.2f} ≥ threshold "
                    f"{cfg.success_threshold} — stopping early."
                )
                break

        self._run_skill_evolution(prompt)
        self._print_summary(best_score)
        return best_image

    # ------------------------------------------------------------------
    # Callbacks & helpers
    # ------------------------------------------------------------------

    def _emit_status(self, state: str, iteration: int = 0, detail: str = "") -> None:
        if self.on_status:
            try:
                self.on_status(state, iteration, detail)
            except Exception:
                pass

    def _generation_timeout(self) -> int:
        if self.config.generation_timeout > 0:
            return self.config.generation_timeout
        return 2400 if self.config.modality == "video" else 600

    def _compute_generation_risk(self, workflow: dict) -> ComputeRisk:
        workload, required = self._estimate_workload_requirement(workflow)
        try:
            stats = self._client.system_stats(timeout=4)
        except Exception as exc:  # noqa: BLE001
            return ComputeRisk(
                ok=False,
                reason=f"Could not read ComfyUI compute stats ({exc}). Are you sure you want to generate?",
                required_vram_gb=required,
                available_vram_gb=None,
                device="unknown",
                workload=workload,
            )

        devices = stats.get("devices") if isinstance(stats, dict) else None
        if not isinstance(devices, list) or not devices:
            return ComputeRisk(
                ok=False,
                reason="No GPU device was reported by ComfyUI. Are you sure you want to generate?",
                required_vram_gb=required,
                available_vram_gb=None,
                device="none",
                workload=workload,
            )

        best_name = "unknown"
        best_free = 0.0
        for dev in devices:
            if not isinstance(dev, dict):
                continue
            name = str(dev.get("name") or dev.get("type") or "device")
            free = self._vram_gb(
                dev.get("torch_vram_free")
                or dev.get("vram_free")
                or dev.get("vram_total")
                or dev.get("torch_vram_total")
            )
            if free is not None and free > best_free:
                best_free = free
                best_name = name

        if best_free <= 0:
            return ComputeRisk(
                ok=False,
                reason="ComfyUI did not report usable free VRAM. Are you sure you want to generate?",
                required_vram_gb=required,
                available_vram_gb=None,
                device=best_name,
                workload=workload,
            )
        if best_free < required:
            return ComputeRisk(
                ok=False,
                reason=(
                    f"{workload} likely needs about {required:.0f} GB free VRAM, "
                    f"but {best_name} reports {best_free:.1f} GB free. "
                    "Are you sure you want to generate?"
                ),
                required_vram_gb=required,
                available_vram_gb=best_free,
                device=best_name,
                workload=workload,
            )
        return ComputeRisk(
            ok=True,
            reason=f"{best_name} reports {best_free:.1f} GB free VRAM.",
            required_vram_gb=required,
            available_vram_gb=best_free,
            device=best_name,
            workload=workload,
        )

    def _estimate_workload_requirement(self, workflow: dict) -> tuple[str, float]:
        blob = json.dumps(workflow, sort_keys=True).lower()
        if self.config.modality == "video" or any(
            token in blob for token in ("wan", "animatediff", "video", "vhs_")
        ):
            return "Video generation", 16.0
        if any(token in blob for token in ("flux", "qwen", "wan2", "fp8")):
            return "Large diffusion model", 10.0
        if "sdxl" in blob:
            return "SDXL image generation", 8.0
        return "Image generation", 4.0

    @staticmethod
    def _vram_gb(value: object) -> float | None:
        try:
            n = float(value)
        except (TypeError, ValueError):
            return None
        if n <= 0:
            return None
        return n / (1024**3)

    def _format_conversation_history(self) -> str | None:
        history = self.config.conversation_history
        if not isinstance(history, list) or not history:
            return None
        lines: list[str] = []
        for item in history[-12:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content[:4000]}")
        if not lines:
            return None
        return "## Conversation History (same ComfyClaw session)\n" + "\n".join(lines)

    def _confirm_generation_compute_risk(self, risk: ComputeRisk) -> bool:
        if self._sync and self._sync.is_running():
            decision = self._sync.request_generation_compute_confirmation(
                risk.as_dict(),
                target_ws=getattr(self, "_sync_ws", None),
            )
            return bool(decision.get("approved"))
        if sys.stdin.isatty():
            answer = input(f"[ComfyClaw] {risk.reason} [y/N] ").strip().lower()
            return answer in {"y", "yes"}
        print("[ClawHarness] No interactive UI is available; proceeding despite compute warning.")
        return True

    def _is_infra_error(self, history: dict) -> bool:
        haystack = "\n".join(
            [
                str(history.get("error", "")),
                "\n".join(str(line) for line in history.get("error_traceback", []) or []),
            ]
        )
        return any(sig in haystack for sig in _INFRA_ERROR_SIGNALS)

    def _on_workflow_change(self, workflow: dict) -> None:
        if self._sync:
            self._sync.broadcast(workflow, target_ws=getattr(self, "_sync_ws", None))

    def _poll_user_refinement(self) -> str | None:
        """Check for a pending user_refinement message (non-blocking)."""
        if not self._sync:
            return None
        ws = getattr(self, "_sync_ws", None)
        self._sync.enable_refinement_listening(source_ws=ws)
        data = self._sync.poll_refinement(source_ws=ws)
        if data:
            return data.get("text", "").strip() or None
        return None

    def _collect_user_feedback_after_generation(
        self,
        result: VerifierResult,
        image_bytes: bytes,
        prompt: str,
        iteration: int,
    ) -> VerifierResult:
        """Ask for thumbs/comment/evolve feedback after a VLM-only pass."""
        if getattr(result, "feedback_source", "vlm") != "vlm":
            return result
        if not self._sync or not self._sync.has_clients():
            return result

        try:
            from .human_verifier import _sniff_extension

            out_dir = Path(tempfile.gettempdir())
            ext = _sniff_extension(image_bytes)
            image_path = out_dir / f"comfyclaw_review_iter{iteration}{ext}"
            image_path.write_bytes(image_bytes)
            self._sync.request_feedback(
                image_path=str(image_path),
                vlm_summary=result.format_feedback(),
                verifier=result.to_payload(),
                iteration=iteration,
                prompt=prompt,
                target_ws=getattr(self, "_sync_ws", None),
            )
            feedback = self._sync.wait_for_human_feedback(
                timeout=600.0,
                source_ws=getattr(self, "_sync_ws", None),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[HumanFeedback] skipped: {exc}")
            return result

        if not feedback:
            return result
        if feedback.get("action") == "skip":
            result.human_feedback = {
                **dict(feedback),
                "rating": "neutral",
                "comment": feedback.get("comment") or feedback.get("text", ""),
                "evolve": False,
            }
            return result

        comment = str(feedback.get("comment") or feedback.get("text") or "").strip()
        rating = str(feedback.get("rating") or "").strip().lower()
        if feedback.get("score") is not None:
            try:
                result.score = max(0.0, min(1.0, float(feedback["score"])))
            except (TypeError, ValueError):
                pass
        if comment:
            result.evolution_suggestions = [f"[HUMAN] {comment}"] + result.evolution_suggestions
        if rating == "down":
            result.failed = result.failed or ["Human marked this generation as a bad case."]
            result.overall_assessment = f"[Human feedback] {comment or 'Needs work.'}"
        elif rating == "up" and comment:
            result.overall_assessment = f"[Human feedback] {comment}"
        result.feedback_source = "hybrid"
        result.human_feedback = {
            **dict(feedback),
            "comment": comment,
            "rating": rating or ("up" if result.score >= 0.75 else "down"),
        }
        return result

    def _on_agent_event(
        self,
        event_type: str,
        content: str,
        tool_name: str = "",
        tool_args: dict | None = None,
    ) -> None:
        if self._sync:
            self._sync.send_agent_event(
                event_type,
                content,
                iteration=self._current_iteration,
                tool_name=tool_name,
                tool_args=tool_args,
                target_ws=getattr(self, "_sync_ws", None),
            )

    def _request_model_download(self, request: dict) -> dict:
        if not self._sync:
            return {"approved": False, "reason": "No ComfyClaw panel is connected."}
        return self._sync.request_model_download(
            request,
            target_ws=getattr(self, "_sync_ws", None),
        )

    def _build_repair_feedback(
        self, error_msg: str | None, last_result: VerifierResult | None
    ) -> str:
        """
        Feedback passed to the agent when ComfyUI rejected the workflow.

        Puts the raw error front-and-centre so the agent can fix the exact
        broken connection or invalid parameter before the next submission.
        """
        lines = [
            "## ⚠️ ComfyUI Rejected the Workflow — Repair Required",
            "",
            "Your last workflow submission was rejected with the following error:",
            f"```\n{error_msg or '(no error details)'}\n```",
            "",
            "**Repair protocol (follow in order):**",
            "1. Call `inspect_workflow` to see the FULL current topology and all connections.",
            "2. Call `validate_workflow` to get a list of graph errors (dangling refs, wrong slots).",
            "3. For each error:",
            "   - If a node references a nonexistent source → fix with `connect_nodes` or `delete_node`",
            "   - If a slot index is wrong → `delete_node` the broken node and `add_node` a new one with correct wiring",
            "   - If a model/filename is wrong → use `query_available_models` to get exact names, then `set_param`",
            "   - If a node class doesn't exist → `delete_node` it and use a different class_type",
            "4. Call `validate_workflow` again to confirm all issues are resolved.",
            "5. Call `finalize_workflow` (it will auto-validate and block if still broken).",
            "",
            "**IMPORTANT:** Do NOT just add new nodes on top of broken ones — `delete_node` the",
            "broken node first, then `add_node` a replacement with correct connections.",
            "",
            "**Output slot reference:**",
            "  CheckpointLoaderSimple → slot 0: MODEL, slot 1: CLIP, slot 2: VAE",
            "  UNETLoader / CLIPLoader / VAELoader → slot 0 only",
            "  KSampler → slot 0: LATENT",
            "  VAEDecode → slot 0: IMAGE",
            "  CLIPTextEncode → slot 0: CONDITIONING",
        ]
        if last_result:
            lines += [
                "",
                "── Previous Verifier Feedback (for context) ──",
                last_result.format_feedback(),
            ]
        return "\n".join(lines)

    def _build_feedback(self, result: VerifierResult | None) -> str | None:
        if result is None:
            return None
        lines = [result.format_feedback()]
        if self._evolution_log.entries:
            lines.append("\n── Evolution History ──")
            lines.append(self._evolution_log.format())
        lines.append(
            "\nChoose the single highest-impact structural upgrade from the "
            "evolution_suggestions above. Declare it with report_evolution_strategy first."
        )
        return "\n".join(lines)

    def _record_error(self, iteration: int, workflow: dict, msg: str) -> None:
        self._memory.record(
            iteration=iteration,
            workflow_snapshot=workflow,
            verifier_score=0.0,
            passed=[],
            failed=[f"Execution error: {msg}"],
            experience=f"Workflow failed: {msg}. Inspect and fix before next attempt.",
        )

    def _feedback_metadata(self, result: VerifierResult) -> dict[str, object]:
        raw = dict(getattr(result, "human_feedback", {}) or {})
        comment = str(raw.get("comment") or raw.get("text") or "").strip()
        rating = str(raw.get("rating") or "").strip().lower()
        if not rating:
            rating = "up" if result.score >= 0.75 else "down" if result.score <= 0.4 else "neutral"
        if rating in {"thumbs_up", "like", "liked", "good", "positive"}:
            rating = "up"
        elif rating in {"thumbs_down", "dislike", "bad", "negative"}:
            rating = "down"
        elif rating not in {"up", "down", "neutral"}:
            rating = "neutral"

        case = "good case" if rating == "up" else "bad case" if rating == "down" else ""
        evolve_raw = raw.get("evolve")
        if evolve_raw is None:
            evolve = bool(case and getattr(result, "feedback_source", "") in {"human", "hybrid"})
        else:
            evolve = bool(evolve_raw)
        return {
            "rating": rating,
            "comment": comment,
            "case": case,
            "evolve": evolve,
        }

    def _summarize_experience(
        self,
        prompt: str,
        passed: list[str],
        failed: list[str],
        rationale: str,
        feedback_meta: dict[str, object] | None = None,
    ) -> str:
        feedback_meta = feedback_meta or {}
        human_line = ""
        if feedback_meta.get("case"):
            human_line = (
                f"\nHuman feedback: {feedback_meta.get('case')} "
                f"(rating={feedback_meta.get('rating')}, "
                f"evolve={feedback_meta.get('evolve')}). "
                f"Comment: {feedback_meta.get('comment') or 'none'}"
            )
        try:
            msg = (
                f"Summarize in ≤80 words. Focus on what worked, failed, and the key lesson.\n\n"
                f"Prompt: {prompt}\nPassed: {', '.join(passed) or 'none'}\n"
                f"Failed: {', '.join(failed) or 'none'}\nAgent rationale: {rationale}"
                f"{human_line}"
            )
            return self._verifier.complete(msg, max_tokens=200)
        except Exception as exc:
            return f"Summary unavailable: {exc}"

    def _run_skill_evolution(self, prompt: str) -> None:
        if not self.config.enable_skill_evolution:
            return
        if not self._memory.attempts:
            return
        human_feedback_seen = any(a.feedback_case for a in self._memory.attempts)
        if human_feedback_seen and not any(a.evolve_requested for a in self._memory.attempts):
            print("[SkillEvolver] Human feedback received; user did not request evolution.")
            return

        def _complete(text: str, max_tokens: int) -> str:
            verifier = self._verifier
            if verifier is None or not hasattr(verifier, "complete"):
                raise RuntimeError("no text completion backend available")
            return verifier.complete(text, max_tokens=max_tokens)

        self._emit_status("evolving_skills", 0, "Reviewing run for skill evolution…")
        evolver = SkillEvolver(
            self._agent.skill_manager,
            complete=_complete if self._verifier is not None else None,
            min_confidence=self.config.skill_evolution_min_confidence,
        )
        try:
            result = evolver.maybe_evolve(
                prompt=prompt,
                memory=self._memory,
                evolution_log=self._evolution_log.format(),
                confirm=self._confirm_skill_evolution,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[SkillEvolver] skipped: {exc}")
            return
        if result.proposal is None:
            print("[SkillEvolver] No reusable skill update proposed.")
        elif result.applied:
            print(f"[SkillEvolver] {result.message}")
            try:
                self._agent.skill_manager.reload()
                if self._sync:
                    self._sync.reload_skills()
            except Exception:
                pass
        else:
            print(f"[SkillEvolver] {result.message}")

    def _confirm_skill_evolution(self, proposal: SkillEvolutionProposal) -> bool:
        if self.config.skill_evolution_auto_apply:
            return True
        if self._sync and hasattr(self._sync, "request_skill_evolution"):
            try:
                # Make the wait explicit in the UI — otherwise the panel keeps
                # showing the previous "Verifying image…" status while the run is
                # actually parked here waiting for the reviewer to respond.
                self._emit_status(
                    "awaiting_skill_approval",
                    0,
                    f"Awaiting your approval to save skill “{proposal.name}”…",
                )
                reply = self._sync.request_skill_evolution(
                    proposal.to_dict(),
                    target_ws=getattr(self, "_sync_ws", None),
                    timeout=600.0,
                )
                if isinstance(reply, dict):
                    approved = bool(reply.get("approved"))
                    if approved and isinstance(reply.get("proposal"), dict):
                        # Fold any human edits from the review modal back into the
                        # proposal that maybe_evolve() is about to write.
                        self._apply_proposal_edits(proposal, reply["proposal"])
                    return approved
                return bool(reply)
            except Exception as exc:  # noqa: BLE001
                print(f"[SkillEvolver] UI approval unavailable: {exc}")
        print("\n[SkillEvolver] ── Proposed Skill Evolution ──")
        print(proposal.format_for_human())
        if not sys.stdin.isatty():
            print("[SkillEvolver] Non-interactive terminal; proposal not applied.")
            return False
        answer = input("\nApply this skill evolution? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    @staticmethod
    def _apply_proposal_edits(
        proposal: SkillEvolutionProposal, edited: dict
    ) -> None:
        """Apply human edits from the review modal onto *proposal* in place.

        Only non-empty values overwrite the originals so a reviewer can't blank
        out a required field (name/description/body) by accident — those keep
        the agent's draft. ``maybe_evolve`` writes this same object after we
        return, so mutating in place is enough.
        """
        from .skill_evolver import _slugify

        name = str(edited.get("name") or "").strip()
        if name:
            proposal.name = _slugify(name) or proposal.name
        for field in ("description", "body", "rationale"):
            value = edited.get(field)
            if isinstance(value, str) and value.strip():
                setattr(proposal, field, value.strip())

    def _print_summary(self, best_score: float) -> None:
        print("\n[ClawHarness] ── Evolution Summary ──")
        print(self._evolution_log.format())
        print(f"[ClawHarness] Best score: {best_score:.2f}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def evolution_log(self) -> EvolutionLog:
        return self._evolution_log

    @property
    def memory(self) -> ClawMemory:
        return self._memory

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_workflow_file(cls, path: str, config: HarnessConfig) -> ClawHarness:
        """
        Load a workflow from a JSON file.

        Handles:
        - API-format dict (keys are numeric strings with ``class_type``)
        - Prompt-keyed save (``{"prompt": {...}}``)
        - UI-format with ``nodes`` list (attempts sibling ``*_api.json`` first)
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, dict) and "prompt" in data and isinstance(data["prompt"], dict):
            data = data["prompt"]
        elif isinstance(data, dict) and "nodes" in data and isinstance(data["nodes"], list):
            api_data = _try_sibling_api(path)
            if api_data is not None:
                print("[ClawHarness] Using sibling API-format workflow.")
                data = api_data
            else:
                print("[ClawHarness] ⚠  UI-format workflow; converting (widget names approximate).")
                data = _ui_to_api(data)

        return cls(base_workflow=data, config=config)

    @classmethod
    def from_workflow_dict(cls, workflow: dict, config: HarnessConfig) -> ClawHarness:
        return cls(base_workflow=workflow, config=config)


# ---------------------------------------------------------------------------
# UI → API conversion helpers
# ---------------------------------------------------------------------------


def _try_sibling_api(ui_path: str) -> dict | None:
    stem = Path(ui_path).stem
    parent = Path(ui_path).parent
    alt_stem = stem.removesuffix("_2512")
    for candidate in [parent / f"{stem}_api.json", parent / f"{alt_stem}_api.json"]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as fh:
                return json.load(fh)
    return None


def _ui_to_api(ui_data: dict) -> dict:
    """
    Best-effort conversion of ComfyUI UI-format to API format.
    Widget values are stored under ``__widget_N`` placeholder keys.
    """
    link_map: dict[int, list] = {}
    for lk in ui_data.get("links", []):
        if len(lk) >= 3:
            link_map[lk[0]] = [lk[1], lk[2]]

    api: dict[str, dict] = {}
    for node in ui_data.get("nodes", []):
        nid = str(node["id"])
        class_type = node.get("type", "Unknown")
        inputs: dict = {}

        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                src = link_map[link_id]
                inputs[inp.get("name", "input")] = [str(src[0]), src[1]]

        for i, val in enumerate(node.get("widgets_values", [])):
            inputs[f"__widget_{i}"] = val

        api[nid] = {
            "class_type": class_type,
            "_meta": {"title": node.get("title", class_type)},
            "inputs": inputs,
        }
    return api
