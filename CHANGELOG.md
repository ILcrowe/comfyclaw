# Changelog

All notable changes to ComfyClaw are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added

- `grok-cli` agent backend using the official Grok Build CLI and a cached
  `grok login` OAuth subscription session. ComfyClaw dispatches JSON-envelope
  tool requests, resumes each panel session independently, and exposes CLI
  installation/auth status in the Agents tab. API-key auth is blocked.
- Post-generation feedback dialog in the ComfyUI panel with thumbs-up /
  thumbs-down rating, optional comment, and opt-in skill evolution.
- Post-run `SkillEvolver` path that distills human-labeled good cases into
  reusable workflow tactics and bad cases into repair / avoidance lessons.
- Skill evolution approval protocol over the sync WebSocket, plus CLI flags
  to disable or auto-apply post-run skill updates.

### Changed

- Per-run memory now records human feedback metadata so later agent iterations
  and skill proposals can distinguish good cases from bad cases.

---

## [0.1.0] — 2026-05-11

First public, camera-ready release accompanying the paper
*An Agentic Harness for Skill-Evolving Image Generation Workflows*
(Li, Liu, Chen, Wu, Liu, Zhou, Xie, Wu, Sun, 2026).

### Added

#### Core harness
- Agent → ComfyUI → verifier loop with topology accumulation
  (`evolve_from_best=True`), configurable iteration cap, success threshold,
  and per-iteration scoreboard events.
- 16 agent tools spanning inspection, validation, basic graph edits, LoRA
  injection, ControlNet branches, regional attention, hires-fix and inpaint
  passes, plus the `set_prompt` shortcut and the skill-loader `read_skill`.
- Three verifier modes — VLM-only (`auto`), human-only, and VLM + human
  override (`copilot`) — selectable per-run from CLI flags or the panel
  toggle.
- Per-run memory (`ClawMemory`) tracking workflow snapshots, scores,
  passed / failed checks, and bounded image cache for cross-iteration
  context.
- `ComfyClient` HTTP + polling client for ComfyUI's `/prompt` and
  `/history` endpoints.

#### Multi-provider LLM support via LiteLLM
- Any of the 100+ providers supported by LiteLLM can drive the agent or
  the verifier (Anthropic, OpenAI, Google Gemini, Azure, Groq, local
  Ollama, vLLM, …).
- Independent agent and verifier model selection via `--model` and
  `--verifier-model` (env: `COMFYCLAW_MODEL`, `COMFYCLAW_VERIFIER_MODEL`).
- Default model string set to `anthropic/claude-sonnet-4-5`.

#### Pluggable agent backends
- LiteLLM driver (default) plus three CLI-spawn backends — `claude-code`
  (Anthropic `claude`), `codex` (OpenAI `codex`), `gemini-cli` (Google
  `gemini`) — all sharing a strict JSON-envelope stdio protocol.
- Automatic fall-back to LiteLLM with a warning (and a red chip in the
  panel) when the requested CLI binary is missing on `$PATH`.

#### Agent Skills (progressive disclosure)
- 19 built-in skills under `comfyclaw/skills/` covering style + quality
  rewrites, model-specific recipes (Qwen-Image-2512, DreamShaper-LCM),
  topology recipes (LoRA, ControlNet, regional, hires-fix, inpaint),
  reasoning helpers (counting, negative prompts, guardrails), the
  `workflow-builder` mega-skill for from-scratch construction, and the
  Anthropic-derived `skill-creator` meta-skill (Apache-2.0).
- Three discovery roots (`builtin`, user `~/.comfyclaw/skills/`, optional
  `--skills-dir`) with persistent enable/disable state in
  `~/.comfyclaw/skills_state.json`.
- Import via folder, `.zip` (safe single-top-dir extraction), or
  `git clone --depth=1`, all surfaced through the Skills tab in the
  ComfyClaw panel.

#### ComfyUI plugin (ComfyClaw-Sync)
- Bundled `custom_node/` shipped inside the wheel and installed via
  `comfyclaw install-node`.
- Tabbed panel — Generate / Skills / History — pinned to the top-right
  of the ComfyUI canvas; drag to reposition, click header to collapse.
- Live, node-by-node graph diff: agent mutations stream from the Python
  side via per-connection WebSockets and animate into the canvas one
  node at a time. Animation speed override:
  `localStorage.setItem('comfyclaw_op_delay', '200')`.
- Iteration scoreboard cards with delta-vs-prev, verifier critique, and
  an *Accept now* button to stop early.
- 3-state run-mode toggle (Manual / Auto / Co-pilot) + an Advanced
  panel for per-run overrides of iterations, verifier mode, and the
  *Build workflow only (no image)* dry-run checkbox.
- Per-tab state isolation in `SyncServer._ConnState` — multiple ComfyUI
  tabs can run independent ComfyClaw sessions without state bleed.
- Backend availability map (`agent_backends` message) — the panel
  detects which CLI agents are installed and colours the chip
  accordingly.

#### Built-in model skills
- `qwen-image-2512` — covers the native ComfyUI FP8 pipeline
  (`UNETLoader` + `CLIPLoader` + `VAELoader` + `KSampler` +
  `EmptySD3LatentImage`), Lightning LoRA 4-/8-step mode, recommended
  aspect-ratio buckets, and per-issue iteration strategies. Auto-loaded
  when the workflow contains Qwen nodes.
- `dreamshaper8-lcm` — DreamShaper 8 LCM model configuration (LCM
  sampler, `sgm_uniform` scheduler, 4–8 steps, CFG 1.5–2.5,
  LCM-compatible hires-fix). Auto-loaded when the active model name
  contains `"lcm"`.
- Reference base workflows under `examples/workflows/`:
  `sd15_dreamshaper_lcm.json`, `qwen_image_2512.json`, and the matching
  UI-format export `qwen_image_2512.ui.json`.

#### Agentic error recovery
- **Queue-error repair**: HTTP 4xx rejections from `/prompt` are fed
  back to the agent verbatim, with up to `--max-repair-attempts`
  (default 2) chances to inspect and fix the topology.
- **Execution-error repair**: ComfyUI run-time errors (wrong types,
  invalid connections, missing inputs) trigger the same repair loop.
- **Infrastructure-fault detection**: `BrokenPipeError` /
  `[Errno 32] Broken pipe` from tqdm's stderr flush is classified as
  transient and retried once without invoking the agent.
- Structured `_build_repair_feedback` helper produces actionable
  feedback: verbatim error, fix steps, common root causes, and
  previous verifier feedback for context.

#### CLI and environment
- Sub-commands: `run`, `serve`, `dry-run`, `install-node`, `node-path`.
- Layered configuration — CLI flags ▶ environment variables ▶ defaults.
- `.env` auto-loading via `python-dotenv` (optional dependency).
- Comprehensive `.env.example` covering all env vars, including
  per-backend binary overrides (`COMFYCLAW_CLAUDE_BIN`,
  `COMFYCLAW_CODEX_BIN`, `COMFYCLAW_GEMINI_BIN`) and the user-skill
  directory override.

#### Tests, CI, packaging
- 227 offline tests covering workflow mutation, agent dispatch, harness
  loop (including the repair branches), verifier (VLM + human + hybrid),
  skills registry (load, detect, import, state persistence), memory,
  per-connection sync server, and agent-backend selection. Full suite
  runs in well under one second.
- Pre-commit hooks (ruff + format on commit, pytest + uv build on push).
- GitHub Actions CI covering ruff, mypy (advisory), pytest on Python
  3.10 – 3.13 across Ubuntu and macOS, and a `uv build` wheel-contents
  check.
- Release workflow on `v*.*.*` tags that builds the wheel, extracts the
  CHANGELOG section for the tag, and publishes a GitHub Release.

### Fixed

- VAE output-slot bug in `_add_hires_fix` and `_add_inpaint_pass`. The
  hardcoded slot `0` is the MODEL output on `CheckpointLoaderSimple`
  (VAE is slot `2`). The fix dynamically copies the `vae` connection
  from an existing `VAEDecode` node, so both `CheckpointLoaderSimple`
  (slot 2) and standalone `VAELoader` (slot 0) are handled.
- `SyncServer` default bind host changed from `127.0.0.1` to `0.0.0.0`
  so the WebSocket server is reachable over remote tunnels and
  container port-forwards.
- The ComfyUI plugin's default WebSocket URL now follows
  `window.location.hostname` instead of a hardcoded `127.0.0.1`.

### Project meta

- GPL-3.0 license (matching ComfyUI, since ComfyClaw is a plugin /
  derivative work), with attribution to the paper authors and a third-party
  notice for the bundled `skill-creator` skill (Apache-2.0).
- `CITATION.cff` shipped at the repository root for one-click citation
  in GitHub.
- Camera-ready documentation: `docs/ARCHITECTURE.md` mapping paper
  sections to source files, `docs/REPRODUCING.md` with exact
  reproducibility instructions for the headline experiments and the
  ablation grid.
- Example workflows and the live-sync demo relocated to
  `examples/workflows/` and `examples/demo_incremental.py` respectively.
