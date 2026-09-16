"""
comfyclaw CLI — entry point installed as the ``comfyclaw`` script.

Configuration precedence (highest → lowest):
  1. Explicit CLI flags
  2. Environment variables  (can be loaded from a .env file)
  3. Built-in defaults

All sensitive configuration (API key, paths) is read from environment
variables only — never hardcoded.  Copy ``.env.example`` to ``.env`` and
fill in your values, or export the variables directly.

Sub-commands
------------
run          Run the full agent–generate–verify loop.
dry-run      Run the agent only (no ComfyUI execution needed).
install-node Symlink the ComfyClaw-Sync custom node into ComfyUI.
node-path    Print the path to the bundled custom node directory.

Environment variables
---------------------
Provider API keys (set the one matching your chosen --model provider):
  ANTHROPIC_API_KEY    Anthropic Claude  (default provider)
  OPENAI_API_KEY       OpenAI GPT-4o / o-series
  GEMINI_API_KEY       Google Gemini
  GROQ_API_KEY         Groq
  (none needed)        Local Ollama

COMFYUI_DIR              Path to ComfyUI installation (install-node).
COMFYUI_ADDR             host:port of a running ComfyUI server.
COMFYCLAW_MODEL          LiteLLM model string for the agent
                         (default: anthropic/claude-sonnet-4-5).
COMFYCLAW_VERIFIER_MODEL LiteLLM model for the vision verifier
                         (default: same as COMFYCLAW_MODEL).
COMFYCLAW_MAX_ITERATIONS Max agent–generate–verify cycles.
COMFYCLAW_THRESHOLD      Stop early when verifier score ≥ this.
COMFYCLAW_SCORE_WEIGHTS  Comma-separated "req_w,detail_w" (sum=1).
COMFYCLAW_EVOLVE_FROM_BEST  "true"/"false" for topology accumulation.
COMFYCLAW_SYNC_PORT      WebSocket port (0 = disable).
COMFYCLAW_SKILLS_DIR     Custom skills directory path.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path

from .model_bundles import (
    MODEL_BUNDLES,
    MODEL_DEST_SUBDIRS,
    bundle_status,
    download_model_file,
    download_model_from_url,
    model_target,
    probe_openai_compatible,
)

# Backward-compatible names used by tests and downstream scripts.
_MODEL_BUNDLES = MODEL_BUNDLES
_bundle_status = bundle_status
_download_model_from_url = download_model_from_url
_download_model_file = download_model_file
_model_target = model_target
_probe_openai_compatible = probe_openai_compatible

# ─────────────────────────────────────────────────────────────────────────────
# .env loader — runs at import time so env vars are available everywhere
# ─────────────────────────────────────────────────────────────────────────────


def _load_dotenv() -> None:
    """
    Load `.env` from the current working directory or the package root.
    Silently skips if python-dotenv is not installed or no .env exists.
    Existing environment variables are NOT overwritten.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
    except ImportError:
        return  # python-dotenv is optional

    # Look in cwd first, then package root
    cwd_env = Path.cwd() / ".env"
    pkg_env = Path(__file__).resolve().parent.parent / ".env"
    env_path = cwd_env if cwd_env.exists() else (pkg_env if pkg_env.exists() else None)
    if env_path:
        load_dotenv(env_path, override=False)


_load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Typed config helpers
# ─────────────────────────────────────────────────────────────────────────────


def _require_env(name: str, hint: str = "") -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        msg = f"Error: {name!r} is not set."
        if hint:
            msg += f"\n{hint}"
        sys.exit(msg)
    return val


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        print(f"[cli] Warning: {name}={raw!r} is not an integer, using {default}.", file=sys.stderr)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        print(f"[cli] Warning: {name}={raw!r} is not a float, using {default}.", file=sys.stderr)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_score_weights(default: tuple[float, float] = (0.6, 0.4)) -> tuple[float, float]:
    raw = os.environ.get("COMFYCLAW_SCORE_WEIGHTS", "").strip()
    if not raw:
        return default
    try:
        parts = [float(x) for x in raw.split(",")]
        if len(parts) == 2:
            return (parts[0], parts[1])
    except ValueError:
        pass
    print(
        f"[cli] Warning: COMFYCLAW_SCORE_WEIGHTS={raw!r} invalid, using {default}.", file=sys.stderr
    )
    return default


def _env_api_base() -> str:
    """Return an OpenAI-compatible API base from env, if configured."""
    for name in ("COMFYCLAW_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"):
        raw = os.environ.get(name, "").strip()
        if raw:
            return raw
    return ""


def _env_path() -> Path:
    return Path.cwd() / ".env"


# ─────────────────────────────────────────────────────────────────────────────
# Derived defaults
# ─────────────────────────────────────────────────────────────────────────────


def _api_key() -> str:
    """Return an API key from the environment.

    LiteLLM reads provider keys from env-vars automatically (ANTHROPIC_API_KEY,
    OPENAI_API_KEY, GEMINI_API_KEY, …).  We read ANTHROPIC_API_KEY here for
    backward compatibility when the model is Anthropic.  If you use a different
    provider, set that provider's env-var instead and leave ANTHROPIC_API_KEY
    unset — in that case this returns an empty string, which is fine.
    """
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def _comfyui_dir() -> Path:
    raw = os.environ.get("COMFYUI_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Documents" / "ComfyUI"


def _server_addr() -> str:
    return _env_str("COMFYUI_ADDR", "127.0.0.1:8188")


def _bundled_custom_node() -> Path:
    """Return the path to the ComfyClaw-Sync custom node bundled inside the package."""
    pkg_node = Path(__file__).resolve().parent / "custom_node"
    if pkg_node.is_dir():
        return pkg_node
    # Development / repo layout
    repo_node = Path(__file__).resolve().parent.parent.parent / "custom_nodes" / "ComfyClaw-Sync"
    return repo_node


# ─────────────────────────────────────────────────────────────────────────────
# Local LLM and ComfyUI model helpers
# ─────────────────────────────────────────────────────────────────────────────


def _update_env_file(path: Path, updates: dict[str, str]) -> None:
    """Create or update a simple KEY=value .env file while preserving comments."""
    existing = path.read_text().splitlines() if path.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out.append(line)
            continue
        key, _sep, _val = line.partition("=")
        key = key.strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    missing = [key for key in updates if key not in seen]
    if missing and out and out[-1].strip():
        out.append("")
    for key in missing:
        out.append(f"{key}={updates[key]}")
    path.write_text("\n".join(out).rstrip() + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Custom-node management
# ─────────────────────────────────────────────────────────────────────────────


def _install_node(comfyui_dir: Path) -> None:
    """Symlink the ComfyClaw-Sync custom node into ComfyUI's custom_nodes/."""
    src = _bundled_custom_node()
    dst = comfyui_dir / "custom_nodes" / "ComfyClaw-Sync"

    if dst.exists() or dst.is_symlink():
        print(f"[cli] Custom node already installed at {dst}")
        return
    if not (comfyui_dir / "custom_nodes").exists():
        print(
            f"[cli] ⚠  ComfyUI custom_nodes dir not found at {comfyui_dir}.\n"
            "       Set COMFYUI_DIR in .env or pass --comfyui-dir."
        )
        return
    if not src.exists():
        print(f"[cli] ⚠  ComfyClaw-Sync source not found at {src}.")
        return
    try:
        dst.symlink_to(src.resolve())
        print(f"[cli] ✅ Symlinked {src.resolve()} → {dst}")
        print("       Restart ComfyUI to activate the sync extension.")
    except Exception as exc:
        print(f"[cli] ❌ Symlink failed: {exc}")
        print(f"       Manual install:\n  cp -r {src.resolve()} {dst.parent}/")


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI startup helper
# ─────────────────────────────────────────────────────────────────────────────


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _is_local(host: str) -> bool:
    return host in _LOCAL_HOSTS


def _ensure_comfyui_running(addr: str) -> str:
    """Ping ComfyUI; auto-discover port or launch Desktop app if local."""
    from .client import ComfyClient

    client = ComfyClient(addr)
    if client.is_alive():
        print(f"[cli] ComfyUI is UP at http://{addr}")
        return addr

    host = addr.split(":")[0] if ":" in addr else "127.0.0.1"

    # Scan common ports on the same host
    probe_ports = [8188, 8000, 8080, 7130]
    for port in probe_ports:
        alt = f"{host}:{port}"
        if alt != addr and ComfyClient(alt).is_alive():
            print(f"[cli] ComfyUI found at http://{alt}")
            return alt

    # Remote host: nothing more we can do — just warn and proceed
    if not _is_local(host):
        print(f"[cli] ⚠  ComfyUI not responding at {addr}")
        print("[cli]    Verify ComfyUI is running on the remote host and the address is correct.")
        print(f"[cli]    Proceeding with {addr} — the agent will fail if ComfyUI is unreachable.")
        return addr

    # Local host: try to launch the Desktop app (macOS)
    print("[cli] ComfyUI not responding locally — attempting to open the app…")
    app_path = Path("/Applications/ComfyUI.app")
    if not app_path.exists():
        print(f"[cli] ⚠  ComfyUI Desktop app not found at {app_path}")
        print("[cli]    Start ComfyUI manually, then re-run this command.")
        return addr
    try:
        subprocess.Popen(
            ["open", str(app_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"[cli] Could not open ComfyUI: {exc}")
        return addr

    print("[cli] Waiting up to 60 s for ComfyUI to start…")
    probe_addrs = [f"{host}:{p}" for p in probe_ports]
    for _ in range(30):
        time.sleep(2)
        for pa in probe_addrs:
            if ComfyClient(pa).is_alive():
                print(f"[cli] ComfyUI started at http://{pa}")
                return pa
    print("[cli] ⚠  Timed out waiting for ComfyUI. Proceeding with {addr}.")
    return addr


def _output_extension(blob: bytes) -> str:
    """Pick the right file extension by sniffing magic bytes."""
    if not blob:
        return ".bin"
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if blob[:2] == b"\xff\xd8":
        return ".jpg"
    if blob[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    if blob[4:8] == b"ftyp":
        return ".mp4"
    if blob[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return ".bin"


def _save_image(image_bytes: bytes, prompt: str, output_dir: Path) -> Path:
    """Save an image *or* video blob with the right extension auto-detected."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = prompt[:40].replace(" ", "_").replace("/", "-")
    ts = int(time.time())
    ext = _output_extension(image_bytes)
    out = output_dir / f"comfyclaw_{ts}_{slug}{ext}"
    out.write_bytes(image_bytes)
    label = "Video" if ext in (".mp4", ".webm", ".gif", ".webp") else "Image"
    print(f"[cli] {label} saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_run(args: argparse.Namespace, dry: bool = False) -> None:
    from .harness import ClawHarness, HarnessConfig

    if getattr(args, "debug_no_generate", False):
        dry = True

    api_key = _api_key()
    addr = args.comfyui_addr

    if not dry:
        addr = _ensure_comfyui_running(addr)

    # CLI flags override env vars; env vars already loaded as defaults
    verifier_model = args.verifier_model.strip() or None
    # Map run_mode to verifier_mode (manual/auto/copilot -> none/vlm/hybrid).
    run_mode = getattr(args, "mode", "auto")
    derived_verifier = {"manual": "none", "auto": "vlm", "copilot": "hybrid"}.get(
        run_mode, args.verifier_mode
    )
    # If --verifier-mode was set explicitly via env or CLI default and
    # --mode wasn't, prefer the verifier flag for back-compat.
    if run_mode == "auto" and args.verifier_mode != "vlm":
        derived_verifier = args.verifier_mode
    cfg = HarnessConfig(
        api_key=api_key,
        server_address=addr,
        model=args.model,
        verifier_model=verifier_model,
        max_iterations=args.iterations if run_mode != "manual" else 1,
        success_threshold=args.threshold,
        sync_port=0 if args.no_sync else args.sync_port,
        skills_dir=args.skills_dir,
        evolve_from_best=not args.reset_each_iter,
        score_weights=_env_score_weights(),
        image_model=args.image_model or None,
        max_repair_attempts=args.max_repair_attempts,
        verifier_mode=derived_verifier,
        agent_backend=getattr(args, "agent_backend", "litellm"),
        run_mode=run_mode,
        modality=getattr(args, "modality", "image"),
        video_frames=getattr(args, "video_frames", 6),
        generation_timeout=getattr(args, "generation_timeout", 0),
        api_base=(getattr(args, "api_base", "") or "").strip() or None,
        enable_skill_evolution=not getattr(args, "no_skill_evolution", False),
        skill_evolution_min_confidence=getattr(args, "skill_evolution_min_confidence", 0.55),
        skill_evolution_auto_apply=getattr(args, "auto_apply_skill_evolution", False),
    )

    verifier_label = cfg.verifier_model or f"{cfg.model} (shared)"
    print(f"\n[cli] Workflow       : {args.workflow or '(empty — agent builds from scratch)'}")
    print(f"[cli] Prompt         : {args.prompt!r}")
    print(f"[cli] Agent model    : {cfg.model}")
    print(f"[cli] Agent backend  : {cfg.agent_backend}")
    if cfg.api_base:
        print(f"[cli] API base       : {cfg.api_base}")
    print(f"[cli] Run mode       : {cfg.run_mode}")
    print(f"[cli] Verifier mode  : {cfg.verifier_mode}")
    if cfg.verifier_mode in ("vlm", "hybrid"):
        print(f"[cli] Verifier model : {verifier_label}")
    print(f"[cli] Image model    : {cfg.image_model or '(from workflow)'}")
    print(f"[cli] Modality       : {cfg.modality}")
    if cfg.modality == "video":
        print(f"[cli] Video frames   : {cfg.video_frames}")
    print(f"[cli] Iterations     : {cfg.max_iterations}  Threshold: {cfg.success_threshold}")
    print(f"[cli] Dry-run        : {dry}")
    print(f"[cli] Sync port      : {cfg.sync_port or 'disabled'}")
    print(f"[cli] Evolve mode    : {'accumulate' if cfg.evolve_from_best else 'reset'}")
    print(f"[cli] Repair limit   : {cfg.max_repair_attempts} attempt(s) per iteration")
    print(
        f"[cli] Skill evolve   : {'on' if cfg.enable_skill_evolution else 'off'}"
        f"{' (auto-apply)' if cfg.skill_evolution_auto_apply else ''}"
    )

    if args.workflow:
        ctx = ClawHarness.from_workflow_file(args.workflow, cfg)
    else:
        ctx = ClawHarness.from_workflow_dict({}, cfg)
    with ctx as h:
        result = h.run(prompt=args.prompt, dry_run=dry)

    if result:
        out_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "comfyclaw_output"
        _save_image(result, args.prompt, out_dir)
    elif dry:
        print("\n[cli] Dry-run complete.")


def _cmd_serve(args: argparse.Namespace) -> None:
    """Persistent server mode — waits for trigger_generation from ComfyUI."""
    from .harness import ClawHarness, HarnessConfig
    from .sync_server import SyncServer

    api_key = _api_key()
    if getattr(args, "debug_no_generate", False):
        # In debug-no-generate mode the ComfyUI backend is never invoked,
        # so don't insist on having one up; use the configured address as-is.
        addr = args.comfyui_addr
        print("[serve] 🐞 Debug mode: ComfyUI generation will be skipped by default.")
    else:
        addr = _ensure_comfyui_running(args.comfyui_addr)

    sync_port = 0 if args.no_sync else args.sync_port
    if not sync_port:
        sys.exit("[cli] Error: serve mode requires sync (WebSocket). Do not use --no-sync.")

    verifier_model = args.verifier_model.strip() or None
    base_cfg = {
        "api_key": api_key,
        "server_address": addr,
        "model": args.model,
        "verifier_model": verifier_model,
        "success_threshold": args.threshold,
        # harness won't create its own SyncServer; we inject the shared one
        "sync_port": 0,
        "skills_dir": args.skills_dir,
        "evolve_from_best": not args.reset_each_iter,
        "score_weights": _env_score_weights(),
        "image_model": args.image_model or None,
        "max_repair_attempts": args.max_repair_attempts,
        "agent_backend": getattr(args, "agent_backend", "litellm"),
        "run_mode": getattr(args, "mode", "auto"),
        "modality": getattr(args, "modality", "image"),
        "video_frames": getattr(args, "video_frames", 6),
        "generation_timeout": getattr(args, "generation_timeout", 0),
        "api_base": (getattr(args, "api_base", "") or "").strip() or None,
        "enable_skill_evolution": not getattr(args, "no_skill_evolution", False),
        "skill_evolution_min_confidence": getattr(args, "skill_evolution_min_confidence", 0.55),
        "skill_evolution_auto_apply": getattr(args, "auto_apply_skill_evolution", False),
    }

    # ── Print the config block first ────────────────────────────────────
    print("\n[cli] ComfyClaw serve mode")
    print(f"[cli] Agent backend  : {base_cfg['agent_backend']}")
    print(f"[cli] Agent model    : {args.model}")
    if base_cfg["api_base"]:
        print(f"[cli] API base       : {base_cfg['api_base']}")
    print(f"[cli] Verifier model : {base_cfg['verifier_model'] or '(same as agent)'}")
    print(f"[cli] Run mode       : {base_cfg['run_mode']}")
    print(f"[cli] Modality       : {base_cfg['modality']}")
    print(f"[cli] ComfyUI        : http://{addr}")
    print(f"[cli] Image model    : {base_cfg['image_model'] or '(from workflow)'}")
    print(f"[cli] Repair limit   : {base_cfg['max_repair_attempts']} attempt(s) per iteration")
    print(f"[cli] Default iters  : {args.iterations}  (panel can override per run)")
    print(f"[cli] Threshold      : {base_cfg['success_threshold']}")
    print()

    # ── Bring the sync server up silently ───────────────────────────────
    # quiet=True suppresses the "Listening on ws://..." line; our own Ready
    # banner below carries the same information in a cleaner format.
    sync = SyncServer(
        port=sync_port,
        model=base_cfg["model"],
        api_key=api_key,
        server_address=addr,
        skills_dir=args.skills_dir,
        quiet=True,
    )
    sync.start()
    if not sync.is_running():
        print(f"[cli] Port {sync_port} appears busy — attempting to reclaim…")
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{sync_port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid.isdigit() and int(pid) != os.getpid():
                    os.kill(int(pid), 9)
                    print(f"[cli] Killed stale process {pid}")
            if pids:
                time.sleep(1)
                sync = SyncServer(port=sync_port)
                sync.start()
        except Exception:
            pass
        if not sync.is_running():
            sys.exit(
                f"[cli] Error: SyncServer failed to start on port {sync_port}. "
                f"Free the port manually: lsof -ti :{sync_port} | xargs kill"
            )

    # ── Count skills up-front (quiet) for the Ready banner ──────────────
    # The SyncServer's own registry is lazy and only warms on the first
    # trigger, so it can't tell us the count yet. Load a separate one just
    # to read the count without spamming the log.
    skill_count = 0
    try:
        from .skill_manager import SkillsRegistry

        skill_count = len(SkillsRegistry(skills_dir=args.skills_dir, quiet=True).skill_names)
    except Exception:
        pass

    # ── Emit a single, coherent "Ready" banner only after everything is up ──
    # The hostname the user types into a browser is more useful than the
    # bind address (which is 0.0.0.0 by default), so prefer the ComfyUI host.
    ws_host = addr.split(":")[0] if ":" in addr else "127.0.0.1"
    ws_url = f"ws://{ws_host}:{sync_port}"
    bar = "─" * 60
    print(bar)
    print(f"[serve] ✅ Ready. Connect ComfyUI panel to  {ws_url}")
    if skill_count:
        print(f"[serve]    {skill_count} skills loaded.")
    print(f"[serve]    Open ComfyUI at http://{addr} → look for the 🐾 panel.")
    print(bar)
    print()

    try:
        while True:
            print("[serve] ⏳ Waiting for generation trigger from ComfyUI…")
            result = sync.wait_for_trigger(timeout=0)
            if result is None:
                continue

            # ── Unpack per-connection trigger ────────────────────────────────
            trigger, source_ws = result

            prompt = trigger.get("prompt", "").strip()
            if not prompt:
                sync.send_error("No prompt provided.", target_ws=source_ws)
                continue

            mode = trigger.get("mode", "scratch")
            settings = trigger.get("settings") or {}
            session_id = str(trigger.get("session_id") or settings.get("session_id") or "")
            workflow = trigger.get("workflow") if mode == "improve" else {}
            if workflow is None:
                workflow = {}

            iterations = settings.get("iterations", args.iterations)
            run_mode = (
                settings.get("mode") or settings.get("run_mode") or getattr(args, "mode", "auto")
            ).strip() or "auto"
            # Map run_mode -> verifier_mode for back-compat callers that send
            # only verifier_mode.
            verifier_default = {
                "manual": "none",
                "auto": "vlm",
                "copilot": "hybrid",
            }.get(run_mode, args.verifier_mode)
            verifier_mode = settings.get("verifier_mode", verifier_default) or verifier_default

            # Model / API-key / base-url overrides from the ComfyUI panel
            trigger_model = settings.get("model", "").strip()
            trigger_api_key = settings.get("api_key", "").strip()
            trigger_api_base = settings.get("api_base", "").strip()
            trigger_verifier_model = settings.get("verifier_model", "").strip()
            trigger_backend = (
                settings.get("agent_backend") or settings.get("backend") or ""
            ).strip()
            conversation_history = settings.get("conversation_history")
            if not isinstance(conversation_history, list):
                conversation_history = []

            # Dry-run / debug mode: agent builds the workflow but ComfyUI is
            # never asked to execute it.  Useful for fast iteration on the
            # agent loop without burning GPU time on each generate click.
            # Resolution order: explicit panel setting > server-side default
            # (--debug-no-generate / COMFYCLAW_DEBUG_NO_GENERATE).
            server_default_dry = bool(getattr(args, "debug_no_generate", False))
            if "dry_run" in settings:
                dry_run = bool(settings.get("dry_run"))
            elif "debug_dry_run" in settings:
                dry_run = bool(settings.get("debug_dry_run"))
            else:
                dry_run = server_default_dry

            run_cfg = dict(base_cfg)
            if trigger_model:
                run_cfg["model"] = trigger_model
            if trigger_api_key:
                run_cfg["api_key"] = trigger_api_key
            if trigger_api_base:
                run_cfg["api_base"] = trigger_api_base
            if trigger_verifier_model:
                run_cfg["verifier_model"] = trigger_verifier_model
            if trigger_backend:
                run_cfg["agent_backend"] = trigger_backend
            run_cfg["agent_session_id"] = session_id

            # Never route provider API credentials into CLI backends
            # (claude-code / codex / gemini-cli / grok-cli). Those backends must use
            # their own local auth sessions to avoid stale-key confusion.
            backend_eff = str(run_cfg.get("agent_backend") or "").strip().lower().replace("_", "-")
            if backend_eff in {"claude-code", "codex", "gemini-cli", "grok-cli"}:
                run_cfg["api_key"] = ""
                run_cfg["api_base"] = ""

            run_cfg["run_mode"] = run_mode
            run_cfg["conversation_history"] = conversation_history

            # Modality may be selected per-trigger from the panel's video tab.
            trigger_modality = (settings.get("modality") or "").strip().lower()
            if trigger_modality in ("image", "video"):
                run_cfg["modality"] = trigger_modality
            trigger_video_frames = settings.get("video_frames")
            if isinstance(trigger_video_frames, int) and trigger_video_frames > 0:
                run_cfg["video_frames"] = trigger_video_frames

            # Manual mode = single round, no iteration loop.
            effective_iters = 1 if run_mode == "manual" else iterations

            cfg = HarnessConfig(
                **run_cfg,
                max_iterations=effective_iters,
                verifier_mode=verifier_mode,
            )

            mode_label = "from scratch" if mode == "scratch" else "improve current"
            node_count = len(workflow) if workflow else 0
            print(f"\n[serve] 🚀 Trigger received: {prompt[:80]!r}")
            print(
                f"[serve]    Mode: {mode_label}, Run: {run_mode}, "
                f"Iters: {effective_iters}, Verifier: {verifier_mode}, "
                f"Backend: {cfg.agent_backend}, Model: {cfg.model}, "
                f"Nodes: {node_count}, Dry-run: {dry_run}"
            )

            # All subsequent messages go to source_ws only
            sync.send_status(
                "running", iteration=0, detail="Initializing agent…", target_ws=source_ws
            )

            if mode == "scratch":
                sync.reset(target_ws=source_ws)
                sync.broadcast({}, target_ws=source_ws)

            # Auto-save a "Before" checkpoint on this connection
            if workflow:
                sync.save_checkpoint(workflow, f"Before: {prompt[:40]}", target_ws=source_ws)
                sync._send_json(
                    {
                        "type": "checkpoints_list",
                        "checkpoints": sync.list_checkpoints(target_ws=source_ws),
                    },
                    target_ws=source_ws,
                )

            try:
                harness = ClawHarness.from_workflow_dict(workflow, cfg)
                # Inject a connection-aware sync reference
                harness._sync = sync
                harness._sync_ws = source_ws  # harness uses this to route broadcasts
                harness._agent.on_change = harness._on_workflow_change

                # Bind source_ws via a default arg so ruff's B023 / closure-over-
                # loop-var doesn't flag this — the callback may outlive this
                # iteration of the serve loop if the harness retains the
                # reference across re-entries.
                def _status_cb(
                    state: str,
                    iteration: int = 0,
                    detail: str = "",
                    _ws: object = source_ws,
                ) -> None:
                    # Check cancel flag for this specific connection
                    with sync._conns_lock:
                        conn = sync._conns.get(_ws)
                    if conn and conn.cancel.is_set():
                        raise KeyboardInterrupt("cancelled by user")
                    sync.send_status(state, iteration, detail, target_ws=_ws)

                harness.on_status = _status_cb

                result_data = harness.run(prompt=prompt, dry_run=dry_run)

                # Auto-save "After" checkpoint
                with sync._conns_lock:
                    conn = sync._conns.get(source_ws)
                final_wf = copy.deepcopy(conn.workflow) if conn and conn.workflow else None
                if final_wf:
                    sync.save_checkpoint(final_wf, f"After: {prompt[:40]}", target_ws=source_ws)
                    sync._send_json(
                        {
                            "type": "checkpoints_list",
                            "checkpoints": sync.list_checkpoints(target_ws=source_ws),
                        },
                        target_ws=source_ws,
                    )

                direct_answer = getattr(harness._agent, "last_direct_answer", "")
                if isinstance(direct_answer, str) and direct_answer.strip():
                    sync.send_complete(
                        score=0.0,
                        iterations_used=0,
                        image_path="",
                        target_ws=source_ws,
                        answer=direct_answer,
                    )
                elif result_data:
                    out_dir = (
                        Path(args.output_dir)
                        if args.output_dir
                        else Path.cwd() / "comfyclaw_output"
                    )
                    saved = _save_image(result_data, prompt, out_dir)
                    sync.send_complete(
                        score=0.0,
                        iterations_used=iterations,
                        image_path=str(saved),
                        target_ws=source_ws,
                    )
                elif dry_run:
                    sync.send_status(
                        "dry_run_done",
                        iteration=effective_iters,
                        detail="Dry-run complete — workflow built, image generation skipped.",
                        target_ws=source_ws,
                    )
                    sync.send_complete(
                        score=0.0,
                        iterations_used=effective_iters,
                        image_path="",
                        target_ws=source_ws,
                    )
                else:
                    sync.send_complete(
                        score=0.0, iterations_used=0, image_path="", target_ws=source_ws
                    )

            except Exception as exc:
                print(f"[serve] ❌ Error: {exc}")
                sync.send_error(str(exc), target_ws=source_ws)

    except KeyboardInterrupt:
        print("\n[serve] Shutting down…")
    finally:
        sync.stop()


def _cmd_install_node(args: argparse.Namespace) -> None:
    comfyui_dir = Path(args.comfyui_dir).expanduser() if args.comfyui_dir else _comfyui_dir()
    _install_node(comfyui_dir)


def _cmd_node_path(_args: argparse.Namespace) -> None:
    """Print the path to the bundled ComfyClaw-Sync custom node directory."""
    print(_bundled_custom_node())


def _cmd_configure_local_llm(args: argparse.Namespace) -> None:
    """Print or write local LLM settings for OpenAI-compatible servers."""
    provider = args.provider
    model = args.model.strip()
    api_base = args.api_base.strip().rstrip("/")
    if provider == "vllm":
        model_value = model if model.startswith("openai/") else f"openai/{model}"
        default_key = "local-vllm"
    elif provider == "ollama":
        model_value = model if model.startswith("ollama/") else f"ollama/{model}"
        default_key = ""
    else:
        model_value = model
        default_key = "local"

    updates = {
        "COMFYCLAW_AGENT_BACKEND": "litellm",
        "COMFYCLAW_MODEL": model_value,
        "COMFYCLAW_API_BASE": api_base,
        "COMFYCLAW_RUN_MODE": args.run_mode,
    }
    if args.verifier_model:
        updates["COMFYCLAW_VERIFIER_MODEL"] = args.verifier_model.strip()
    elif args.run_mode == "manual":
        updates["COMFYCLAW_VERIFIER_MODEL"] = model_value
    if default_key:
        updates["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", default_key)

    print("\n[local-llm] Recommended environment")
    for key, value in updates.items():
        print(f"{key}={value}")

    if args.check:
        ok, detail, models = probe_openai_compatible(api_base)
        state = "OK" if ok else "FAILED"
        print(f"\n[local-llm] Endpoint check: {state} — {detail}")
        if (
            ok
            and provider == "vllm"
            and model not in models
            and model_value.removeprefix("openai/") not in models
        ):
            print(f"[local-llm] Warning: requested model {model!r} was not listed by /v1/models.")

    if args.write_env:
        env_file = Path(args.env_file).expanduser() if args.env_file else _env_path()
        _update_env_file(env_file, updates)
        print(f"\n[local-llm] Wrote {env_file}")
    else:
        print("\n[local-llm] Re-run with --write-env to update .env automatically.")

    if provider == "vllm":
        served = model_value.removeprefix("openai/")
        print("\n[local-llm] vLLM launch hint for shared GPU video generation:")
        print(
            "vllm serve "
            f"{served} --host 127.0.0.1 --port {api_base.rsplit(':', 1)[-1].split('/')[0] if ':' in api_base else '8000'} "
            "--enable-auto-tool-choice --tool-call-parser qwen3_xml "
            "--max-model-len 32768 --gpu-memory-utilization 0.65 --max-num-seqs 2"
        )


def _cmd_models(args: argparse.Namespace) -> None:
    """List, check, or download known ComfyUI model bundles."""
    action = args.models_action

    comfyui_dir = (
        Path(args.comfyui_dir).expanduser()
        if getattr(args, "comfyui_dir", None)
        else _comfyui_dir()
    )
    if action == "download-url":
        print(f"[models] Downloading URL into {comfyui_dir / 'models' / args.dest_subdir}")
        target = download_model_from_url(
            args.url,
            comfyui_dir=comfyui_dir,
            dest_subdir=args.dest_subdir,
            filename=args.filename or None,
        )
        print(f"[models] Done: {target}")
        print("[models] Restart ComfyUI so model dropdowns refresh.")
        return

    if action == "list":
        print("\n[models] Known bundles")
        for name, bundle in MODEL_BUNDLES.items():
            print(f"  {name:<18} {bundle.description}")
        return

    bundle = MODEL_BUNDLES[args.bundle]
    statuses = bundle_status(comfyui_dir, bundle)
    print(f"\n[models] {bundle.name}: {bundle.description}")
    print(f"[models] ComfyUI dir: {comfyui_dir}")
    for mf, target, exists in statuses:
        required = "optional" if mf.optional else "required"
        marker = "OK" if exists else "missing"
        print(f"  {marker:<7} {required:<8} {target}")
        if not exists:
            print(f"           source: hf://{mf.repo}/{mf.path}")
    for note in bundle.notes:
        print(f"[models] Note: {note}")

    missing = [(mf, target) for mf, target, exists in statuses if not exists]
    if action == "check":
        required_missing = [mf for mf, _target in missing if not mf.optional]
        if required_missing:
            sys.exit(1)
        return

    if not missing:
        print("[models] All files already present.")
        return

    selected = [(mf, target) for mf, target in missing if args.include_optional or not mf.optional]
    if not selected:
        print("[models] Only optional files are missing. Use --include-optional to download them.")
        return

    print(f"\n[models] Downloading {len(selected)} file(s). Large model files can take a while.")
    for mf, target in selected:
        print(f"[models] Downloading {mf.dest_name} → {target}")
        download_model_file(mf, target)
    print("[models] Done. Restart ComfyUI so model dropdowns refresh.")


# ─────────────────────────────────────────────────────────────────────────────
# `comfyclaw doctor` — pre-flight check
# ─────────────────────────────────────────────────────────────────────────────


_DOCTOR_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("ANTHROPIC_API_KEY", "anthropic/claude-*"),
    ("OPENAI_API_KEY", "openai/gpt-*, openai/o*"),
    ("GEMINI_API_KEY", "gemini/gemini-*"),
    ("GROQ_API_KEY", "groq/*"),
    ("AZURE_API_KEY", "azure/<deployment>"),
)


def _doctor_row(state: str, label: str, detail: str = "") -> None:
    """Print one doctor row.  ``state`` ∈ {ok, warn, fail}."""
    icons = {"ok": "✅", "warn": "⚠ ", "fail": "❌"}
    icon = icons.get(state, "•")
    line = f"  {icon}  {label}"
    if detail:
        line = f"{line:<42} {detail}"
    print(line)


def _cmd_doctor(args: argparse.Namespace) -> None:
    """Pre-flight check: env, ComfyUI, plugin, port, skills."""
    print("\n[doctor] ComfyClaw pre-flight check\n")

    critical_failures = 0
    warnings = 0

    # ── 1. Python version ────────────────────────────────────────────────
    py = sys.version_info
    py_str = f"{py.major}.{py.minor}.{py.micro}"
    if (py.major, py.minor) >= (3, 10):
        _doctor_row("ok", "Python version", py_str)
    else:
        _doctor_row("fail", "Python version", f"{py_str} (need ≥ 3.10)")
        critical_failures += 1

    # ── 2. `.env` file ───────────────────────────────────────────────────
    cwd_env = Path.cwd() / ".env"
    pkg_env = Path(__file__).resolve().parent.parent / ".env"
    found_env = cwd_env if cwd_env.exists() else (pkg_env if pkg_env.exists() else None)
    if found_env:
        _doctor_row("ok", ".env file", str(found_env))
    else:
        _doctor_row(
            "warn",
            ".env file",
            "not found — relying on shell environment only",
        )
        warnings += 1

    # ── 3. Provider API keys ─────────────────────────────────────────────
    provider_count = 0
    for var, models in _DOCTOR_PROVIDERS:
        if os.environ.get(var, "").strip():
            _doctor_row("ok", f"{var}", f"set → covers {models}")
            provider_count += 1
    if provider_count == 0:
        # Local Ollama doesn't need a key, so this is a warning, not fatal.
        _doctor_row(
            "warn",
            "Provider API key",
            "none set — only local providers (e.g. ollama/*) will work",
        )
        warnings += 1

    # ── 3b. OpenAI-compatible local endpoint ─────────────────────────────
    configured_model = _env_str("COMFYCLAW_MODEL", "")
    api_base = _env_api_base()
    if api_base and configured_model.startswith("openai/"):
        ok, detail, models = probe_openai_compatible(api_base, timeout=3)
        if ok:
            expected = configured_model.removeprefix("openai/")
            if models and expected not in models:
                _doctor_row(
                    "warn",
                    "Local LLM endpoint",
                    f"reachable at {api_base}, but {expected!r} not listed ({detail})",
                )
                warnings += 1
            else:
                _doctor_row("ok", "Local LLM endpoint", f"{api_base} → {detail}")
        else:
            _doctor_row("warn", "Local LLM endpoint", f"{api_base} probe failed: {detail}")
            warnings += 1

    # ── 4. ComfyUI reachable ─────────────────────────────────────────────
    addr = getattr(args, "comfyui_addr", None) or _server_addr()
    try:
        from .client import ComfyClient

        if ComfyClient(addr).is_alive():
            _doctor_row("ok", "ComfyUI", f"reachable at http://{addr}")
        else:
            # Scan common ports on the same host
            host = addr.split(":")[0] if ":" in addr else "127.0.0.1"
            found_alt = None
            for port in (8188, 8000, 8080, 7130):
                alt = f"{host}:{port}"
                if alt != addr and ComfyClient(alt).is_alive():
                    found_alt = alt
                    break
            if found_alt:
                _doctor_row(
                    "warn",
                    "ComfyUI",
                    f"not at {addr} but found at {found_alt} — update COMFYUI_ADDR",
                )
                warnings += 1
            else:
                _doctor_row("fail", "ComfyUI", f"not reachable at {addr}")
                critical_failures += 1
    except Exception as exc:
        _doctor_row("fail", "ComfyUI", f"probe failed: {exc}")
        critical_failures += 1

    # ── 5. Plugin installed in ComfyUI's custom_nodes/ ────────────────────
    comfyui_dir = _comfyui_dir()
    nodes_dir = comfyui_dir / "custom_nodes"
    plugin_link = nodes_dir / "ComfyClaw-Sync"
    if not nodes_dir.is_dir():
        _doctor_row(
            "warn",
            "ComfyClaw-Sync plugin",
            f"ComfyUI custom_nodes/ not found at {nodes_dir}",
        )
        warnings += 1
    elif not plugin_link.exists() and not plugin_link.is_symlink():
        _doctor_row(
            "warn",
            "ComfyClaw-Sync plugin",
            "not installed — run `comfyclaw install-node`",
        )
        warnings += 1
    else:
        bundled = _bundled_custom_node().resolve()
        if plugin_link.is_symlink():
            target = plugin_link.resolve()
            if target == bundled:
                _doctor_row("ok", "ComfyClaw-Sync plugin", "symlinked to this install")
            else:
                _doctor_row(
                    "warn",
                    "ComfyClaw-Sync plugin",
                    f"symlinked to a different install ({target})",
                )
                warnings += 1
        else:
            _doctor_row(
                "warn",
                "ComfyClaw-Sync plugin",
                f"installed as copy (not symlink) at {plugin_link}",
            )
            warnings += 1

    # ── 6. Sync port free ────────────────────────────────────────────────
    sync_port = _env_int("COMFYCLAW_SYNC_PORT", 8765)
    try:
        # Try lsof first (descriptive); fall back to a TCP probe.
        proc = subprocess.run(
            ["lsof", "-nP", "-iTCP:" + str(sync_port), "-sTCP:LISTEN", "-Fpc"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.stdout.strip():
            # Parse lsof -F output: lines like "p<pid>" + "c<command>"
            lines = proc.stdout.strip().splitlines()
            pid = next((line[1:] for line in lines if line.startswith("p")), "?")
            cmd = next((line[1:] for line in lines if line.startswith("c")), "?")
            _doctor_row(
                "warn",
                f"Sync port {sync_port}",
                f"held by {cmd} (pid {pid}) — stop it before `comfyclaw serve`",
            )
            warnings += 1
        else:
            _doctor_row("ok", f"Sync port {sync_port}", "free")
    except Exception:
        # If lsof unavailable, do a best-effort TCP probe.
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                sock.bind(("127.0.0.1", sync_port))
                _doctor_row("ok", f"Sync port {sync_port}", "free")
        except OSError:
            _doctor_row("warn", f"Sync port {sync_port}", "in use")
            warnings += 1

    # ── 7. Bundled skills loadable ───────────────────────────────────────
    try:
        from .skill_manager import SkillsRegistry

        reg = SkillsRegistry(quiet=True)
        n = len(reg.skill_names)
        _doctor_row("ok", "Bundled skills", f"{n} loaded")
    except Exception as exc:
        _doctor_row("fail", "Bundled skills", f"failed to load: {exc}")
        critical_failures += 1

    # ── 8. websockets package (needed for sync) ──────────────────────────
    try:
        import websockets  # noqa: F401

        _doctor_row("ok", "websockets package", "installed")
    except ImportError:
        _doctor_row(
            "warn",
            "websockets package",
            "missing — install with `pip install 'comfyclaw[sync]'`",
        )
        warnings += 1

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    if critical_failures == 0 and warnings == 0:
        print("[doctor] ✅ All checks passed. Ready to `comfyclaw serve`.")
        sys.exit(0)
    elif critical_failures == 0:
        print(
            f"[doctor] ⚠  {warnings} warning(s) — ComfyClaw should still work, "
            f"but fix these for the best experience."
        )
        sys.exit(0)
    else:
        print(
            f"[doctor] ❌ {critical_failures} critical issue(s), {warnings} warning(s). "
            f"Fix the critical ones before running `comfyclaw serve` or `comfyclaw run`."
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comfyclaw",
        description="ComfyClaw — agentic self-evolving ComfyUI harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Env-var configuration: copy .env.example → .env and fill in your values.\n"
            "All CLI flags override the corresponding env var.\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_run_args(p: argparse.ArgumentParser, *, prompt_required: bool = True) -> None:
        p.add_argument(
            "--comfyui-addr",
            default=_server_addr(),
            metavar="HOST:PORT",
            help=(
                "ComfyUI server address, e.g. '127.0.0.1:7130'. "
                "Default: COMFYUI_ADDR env var or 127.0.0.1:8188"
            ),
        )
        p.add_argument(
            "--workflow",
            default=None,
            metavar="PATH",
            help="Path to API-format ComfyUI workflow JSON (omit to start from scratch)",
        )
        p.add_argument(
            "--prompt",
            required=prompt_required,
            default=None,
            help="Image generation prompt"
            + ("" if prompt_required else " (optional — comes from ComfyUI panel)"),
        )
        p.add_argument(
            "--model",
            default=_env_str("COMFYCLAW_MODEL", "anthropic/claude-sonnet-4-5"),
            metavar="MODEL",
            help=(
                "LiteLLM model string for the agent, e.g. 'anthropic/claude-sonnet-4-5', "
                "'openai/gpt-5.4', 'gemini/gemini-2.0-flash', 'ollama/llama3.1'. "
                "Set the matching provider API key env-var."
            ),
        )
        p.add_argument(
            "--verifier-model",
            default=_env_str("COMFYCLAW_VERIFIER_MODEL", ""),
            metavar="MODEL",
            help=(
                "LiteLLM model string for the vision verifier (must support images). "
                "Defaults to the same value as --model. "
                "Example: --model ollama/llama3.1 --verifier-model openai/gpt-5.4"
            ),
        )
        p.add_argument(
            "--iterations", type=int, default=_env_int("COMFYCLAW_MAX_ITERATIONS", 3), metavar="N"
        )
        p.add_argument(
            "--threshold",
            type=float,
            default=_env_float("COMFYCLAW_THRESHOLD", 0.85),
            metavar="SCORE",
        )
        p.add_argument(
            "--sync-port", type=int, default=_env_int("COMFYCLAW_SYNC_PORT", 8765), metavar="PORT"
        )
        p.add_argument("--no-sync", action="store_true", help="Disable live WebSocket sync")
        p.add_argument(
            "--skills-dir", default=os.environ.get("COMFYCLAW_SKILLS_DIR") or None, metavar="DIR"
        )
        p.add_argument(
            "--reset-each-iter",
            action="store_true",
            default=not _env_bool("COMFYCLAW_EVOLVE_FROM_BEST", True),
            help="Disable topology accumulation (reset to base each iteration)",
        )
        p.add_argument(
            "--max-repair-attempts",
            type=int,
            default=_env_int("COMFYCLAW_MAX_REPAIR_ATTEMPTS", 2),
            metavar="N",
            help="Max agent repair attempts when ComfyUI rejects a workflow (default 2)",
        )
        p.add_argument(
            "--output-dir", default=None, metavar="DIR", help="Directory for saved output images"
        )
        p.add_argument(
            "--image-model",
            default=os.environ.get("COMFYCLAW_IMAGE_MODEL", "").strip() or None,
            metavar="NAME",
            help=(
                "Pin the ComfyUI checkpoint / UNET to this model name, e.g. "
                "'Qwen/Qwen-Image-2512' or 'realisticVisionV51.safetensors'. "
                "Overrides COMFYCLAW_IMAGE_MODEL env var. "
                "Leave unset to use whatever model the workflow already specifies."
            ),
        )
        p.add_argument(
            "--verifier-mode",
            default=_env_str("COMFYCLAW_VERIFIER_MODE", "vlm"),
            choices=["vlm", "human", "hybrid"],
            metavar="MODE",
            help=(
                "Verification mode: 'vlm' (default) uses a vision LLM, "
                "'human' collects feedback via ComfyUI panel or terminal, "
                "'hybrid' runs VLM first then lets a human accept or override."
            ),
        )
        p.add_argument(
            "--mode",
            default=_env_str("COMFYCLAW_RUN_MODE", "auto"),
            choices=["manual", "auto", "copilot"],
            metavar="MODE",
            help=(
                "Run mode: 'manual' (single round, no verifier), 'auto' (VLM "
                "verifier + iterations), 'copilot' (VLM scores then asks human)."
            ),
        )
        p.add_argument(
            "--agent-backend",
            default=_env_str("COMFYCLAW_AGENT_BACKEND", "litellm"),
            choices=["litellm", "claude-code", "codex", "gemini-cli", "grok-cli"],
            metavar="BACKEND",
            help=(
                "Agent driver: 'litellm' (default — any LiteLLM provider), "
                "'claude-code' (uses `claude` CLI), 'codex' (uses `codex` CLI), "
                "'gemini-cli' (uses `gemini` CLI), 'grok-cli' (uses `grok` CLI). "
                "Grok requires a subscription login and does not fall back to an API backend."
            ),
        )
        p.add_argument(
            "--api-base",
            default=_env_api_base(),
            metavar="URL",
            help=(
                "OpenAI-compatible API base URL for LiteLLM, e.g. "
                "http://127.0.0.1:18000/v1. Defaults to COMFYCLAW_API_BASE, "
                "OPENAI_API_BASE, or OPENAI_BASE_URL."
            ),
        )
        p.add_argument(
            "--debug-no-generate",
            action="store_true",
            default=_env_bool("COMFYCLAW_DEBUG_NO_GENERATE", False),
            help=(
                "Debug mode: build the workflow but skip ComfyUI execution. "
                "Useful for fast iteration on the agent loop without burning "
                "GPU time. In serve mode this is the default for every "
                "trigger unless the panel explicitly overrides it."
            ),
        )
        p.add_argument(
            "--modality",
            default=_env_str("COMFYCLAW_MODALITY", "image"),
            choices=["image", "video"],
            metavar="MODE",
            help=(
                "Output modality: 'image' (default) or 'video'. Video mode "
                "collects animated outputs (mp4 / webm / animated WEBP / GIF) "
                "from ComfyUI and scores them with a frame-sampling verifier. "
                "The run-video / serve-video subcommands set this to 'video' "
                "by default."
            ),
        )
        p.add_argument(
            "--video-frames",
            type=int,
            default=_env_int("COMFYCLAW_VIDEO_FRAMES", 6),
            metavar="N",
            help=(
                "Number of frames to sample per clip for the video verifier. "
                "Default 6 (first, last, and 4 evenly-spaced). Only used when "
                "--modality=video."
            ),
        )
        p.add_argument(
            "--generation-timeout",
            type=int,
            default=_env_int("COMFYCLAW_GENERATION_TIMEOUT", 0),
            metavar="SECONDS",
            help=(
                "Seconds to wait for ComfyUI generation. Default 600 for image, "
                "2400 for video. Overrides COMFYCLAW_GENERATION_TIMEOUT."
            ),
        )
        p.add_argument(
            "--no-skill-evolution",
            action="store_true",
            default=not _env_bool("COMFYCLAW_ENABLE_SKILL_EVOLUTION", True),
            help=(
                "Disable post-run skill self-evolution. By default ComfyClaw reflects "
                "after each verified run and asks before writing any skill."
            ),
        )
        p.add_argument(
            "--skill-evolution-min-confidence",
            type=float,
            default=_env_float("COMFYCLAW_SKILL_EVOLUTION_MIN_CONFIDENCE", 0.55),
            metavar="N",
            help="Minimum proposal confidence required before asking the human. Default 0.55.",
        )
        p.add_argument(
            "--auto-apply-skill-evolution",
            action="store_true",
            default=_env_bool("COMFYCLAW_AUTO_APPLY_SKILL_EVOLUTION", False),
            help=(
                "Apply proposed skill changes without human confirmation. Intended for "
                "trusted offline experiments only."
            ),
        )

    run_p = sub.add_parser("run", help="Run the full agent–generate–verify loop")
    _add_run_args(run_p)
    run_p.set_defaults(func=lambda a: _cmd_run(a, dry=False))

    dry_p = sub.add_parser("dry-run", help="Run agent only (no ComfyUI execution)")
    _add_run_args(dry_p)
    dry_p.set_defaults(func=lambda a: _cmd_run(a, dry=True))

    serve_p = sub.add_parser(
        "serve",
        help="Start persistent server — listen for generation triggers from ComfyUI",
    )
    _add_run_args(serve_p, prompt_required=False)
    serve_p.set_defaults(func=_cmd_serve)

    # Video parallels — same flags but modality defaults to "video".
    def _force_video(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:
            if getattr(action, "dest", "") == "modality":
                action.default = "video"
                break

    run_video_p = sub.add_parser(
        "run-video",
        help="One-shot video-mode agent–generate–verify loop",
    )
    _add_run_args(run_video_p)
    _force_video(run_video_p)
    run_video_p.set_defaults(func=lambda a: _cmd_run(a, dry=False))

    serve_video_p = sub.add_parser(
        "serve-video",
        help="Persistent video-mode server — listens for triggers from the ComfyUI Video tab",
    )
    _add_run_args(serve_video_p, prompt_required=False)
    _force_video(serve_video_p)
    serve_video_p.set_defaults(func=_cmd_serve)

    inst_p = sub.add_parser("install-node", help="Symlink ComfyClaw-Sync custom node into ComfyUI")
    inst_p.add_argument(
        "--comfyui-dir",
        default=None,
        metavar="DIR",
        help="ComfyUI installation directory (or set COMFYUI_DIR in .env)",
    )
    inst_p.set_defaults(func=_cmd_install_node)

    np_p = sub.add_parser("node-path", help="Print path to the bundled ComfyClaw-Sync plugin")
    np_p.set_defaults(func=_cmd_node_path)

    local_p = sub.add_parser(
        "configure-local-llm",
        help="Generate/check .env settings for local vLLM/Ollama/OpenAI-compatible LLMs",
    )
    local_p.add_argument(
        "--provider",
        choices=["vllm", "ollama", "openai-compatible"],
        default="vllm",
        help="Local provider style. Default: vllm",
    )
    local_p.add_argument(
        "--model",
        required=True,
        help="Model id, e.g. Qwen/Qwen3.6-27B for vLLM or llama3.1 for Ollama.",
    )
    local_p.add_argument(
        "--api-base",
        default="http://127.0.0.1:18000/v1",
        help="OpenAI-compatible API base URL. Default: http://127.0.0.1:18000/v1",
    )
    local_p.add_argument(
        "--verifier-model",
        default="",
        help="Optional VLM verifier model. Leave empty in manual mode.",
    )
    local_p.add_argument(
        "--run-mode",
        choices=["manual", "auto", "copilot"],
        default="manual",
        help="Default run mode to write. Use manual when the local model is text-only.",
    )
    local_p.add_argument(
        "--write-env",
        action="store_true",
        help="Write the recommended settings into .env.",
    )
    local_p.add_argument(
        "--env-file",
        default="",
        help="Optional .env path to update. Defaults to ./.env.",
    )
    local_p.add_argument(
        "--check",
        action="store_true",
        help="Probe <api-base>/models and report available model ids.",
    )
    local_p.set_defaults(func=_cmd_configure_local_llm)

    models_p = sub.add_parser(
        "models",
        help="List, check, or download known ComfyUI image/video model bundles",
    )
    models_sub = models_p.add_subparsers(dest="models_action", required=True)
    models_sub.add_parser("list", help="List known bundles").set_defaults(func=_cmd_models)

    def _add_model_bundle_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("bundle", choices=sorted(MODEL_BUNDLES), help="Model bundle name")
        p.add_argument(
            "--comfyui-dir",
            default=None,
            metavar="DIR",
            help="ComfyUI installation directory (or set COMFYUI_DIR in .env)",
        )

    check_p = models_sub.add_parser("check", help="Check whether a bundle is installed")
    _add_model_bundle_args(check_p)
    check_p.set_defaults(func=_cmd_models)

    download_p = models_sub.add_parser("download", help="Download a bundle into ComfyUI/models")
    _add_model_bundle_args(download_p)
    download_p.add_argument(
        "--include-optional",
        action="store_true",
        help="Also download optional files such as Lightning LoRAs.",
    )
    download_p.set_defaults(func=_cmd_models)

    url_p = models_sub.add_parser(
        "download-url",
        help="Download a pasted model URL into ComfyUI/models",
    )
    url_p.add_argument("url", help="Hugging Face file URL or direct http(s) URL")
    url_p.add_argument(
        "--dest-subdir",
        default="checkpoints",
        choices=MODEL_DEST_SUBDIRS,
        help="ComfyUI models subdirectory for the downloaded file.",
    )
    url_p.add_argument(
        "--filename",
        default="",
        help="Optional destination filename. Defaults to the URL filename.",
    )
    url_p.add_argument(
        "--comfyui-dir",
        default=None,
        metavar="DIR",
        help="ComfyUI installation directory (or set COMFYUI_DIR in .env)",
    )
    url_p.set_defaults(func=_cmd_models)

    doctor_p = sub.add_parser(
        "doctor",
        help="Pre-flight check (env, ComfyUI, plugin, port, skills)",
        description=(
            "Run a pre-flight check of your install. Exits 0 if there are no "
            "critical failures; non-zero otherwise. Warnings are reported but "
            "do not fail the check."
        ),
    )
    doctor_p.add_argument(
        "--comfyui-addr",
        default=_server_addr(),
        metavar="HOST:PORT",
        help=("ComfyUI server address to probe. Default: COMFYUI_ADDR env var or 127.0.0.1:8188"),
    )
    doctor_p.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
