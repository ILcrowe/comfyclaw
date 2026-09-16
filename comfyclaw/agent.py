"""
ClawAgent — LLM agent that evolves ComfyUI workflow topology.

The agent uses LiteLLM's unified completion API (OpenAI-compatible tool-use
format) to call structured workflow operations in a loop until it calls
``finalize_workflow``.  Any provider supported by LiteLLM can be used —
Anthropic, OpenAI, Google Gemini, local Ollama, etc.

Tool categories
---------------
Inspection  : inspect_workflow, query_available_models
Prompt      : set_prompt  (auto-resolves sampler→encoder links; no node ID needed)
Basic edit  : set_param, add_node, connect_nodes, delete_node
Models      : download_model_weights
LoRA        : add_lora_loader
ControlNet  : add_controlnet
Regional    : add_regional_attention
Refinement  : add_hires_fix, add_inpaint_pass
Control     : report_evolution_strategy, finalize_workflow
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .agent_backends import AgentBackend, ToolCall, get_backend
from .model_bundles import MODEL_DEST_SUBDIRS, download_model_from_url
from .skill_manager import SkillsRegistry
from .workflow import WorkflowManager

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are ComfyClaw, an expert ComfyUI workflow engineer.  Your job is to BUILD
and GROW ComfyUI workflow topologies — constructing complete pipelines from
scratch when the workflow is empty, and evolving existing ones in response to
the verifier's region-level feedback.

Unified input behavior
----------------------
The user has one Cursor-style input box that can contain either conversation
or a workflow request. Your first responsibility is to infer the user's intent
from the whole message and current workflow context.

If the message is conversational, asks for information, asks for diagnosis, or
does not clearly ask you to change/build/run the ComfyUI workflow, call
answer_user with a concise answer and stop. Do not inspect, validate, mutate,
or finalize the workflow for conversational turns.

If the message asks you to create a new result, run generation, or modify the
current workflow/output, use the workflow tools below and finish with
finalize_workflow.

Iteration strategy
------------------
Only after you have decided the user wants workflow work:
1. Call report_evolution_strategy first: state your plan and the top issue.
2. Call inspect_workflow to see the current topology.
3. **If the workflow is empty** (no nodes):
   a. Call read_skill("workflow-builder") to load architecture recipes.
   b. Call query_available_models("checkpoints") and query_available_models("diffusion_models")
      to discover available models — NEVER guess filenames.
   c. Match the model filename to an architecture (SD 1.5, SDXL, Flux, Qwen, etc.)
      using the patterns in the workflow-builder skill.
   d. Build the full pipeline node-by-node using add_node, following the matching recipe.
   e. Use ONLY exact filenames from query results.
   f. Set detailed prompts on the CLIPTextEncode nodes.
   g. Call validate_workflow to catch wiring errors before submitting.
   h. Call finalize_workflow (it auto-validates and blocks if errors remain).
4. **If the workflow already has nodes**, follow the evolution strategy:
   a. Call set_prompt — craft a detailed, professional positive prompt AND a strong
      negative prompt based on the user's goal (see "Prompt engineering" below).
      Do this EVERY iteration, even if you also plan structural changes.
   b. If a relevant skill is listed in <available_skills>, call read_skill to load
      its full instructions BEFORE applying that upgrade.
   c. Call query_available_models BEFORE adding any LoRA or ControlNet node.
   d. Apply structural upgrades (LoRA / ControlNet / regional / hires / inpaint).
   e. Tune sampler parameters (steps, CFG, seed) as needed.
   f. Call validate_workflow to catch wiring errors.
   g. Call finalize_workflow when done (it auto-validates).

Prompt engineering
------------------
When building or modifying a generation workflow, use set_prompt to write a
professional-quality positive prompt and a strong negative prompt. Do not do
this for conversational turns.

Positive prompt — structure:
  [subject & scene], [style], [lighting], [camera/lens], [quality boosters]
  • Expand every meaningful concept: vague nouns → vivid adjectives + nouns.
  • Add artistic / photographic style: "cinematic", "concept art", "photorealistic",
    "watercolor painting", "isometric", etc.
  • Add lighting: "golden hour", "dramatic rim lighting", "neon glow", "soft diffuse".
  • Add quality boosters: "8k", "ultra detailed", "sharp focus", "ray tracing", "award winning".
  • If the image has multiple subjects/regions, describe each clearly.

Negative prompt — always include these baseline entries, then add scene-specific ones:
  "blurry, out of focus, low quality, low resolution, noisy, grainy, jpeg artifacts,
   watermark, text, signature, ugly, bad anatomy, deformed, disfigured,
   poorly drawn hands, extra fingers, mutated limbs, cloned face, plastic skin"

Example (input: "a cyberpunk city at night"):
  positive: "a futuristic cyberpunk city skyline at night, towering neon-lit skyscrapers,
   wet reflective streets, holographic advertisements, dense rain, cinematic composition,
   dramatic volumetric lighting, wide angle lens 24mm, 8k, photorealistic, ultra detailed,
   sharp focus, ray tracing, blade runner aesthetic"
  negative: "blurry, low quality, noisy, watermark, text, bad anatomy, deformed, ugly,
   cartoon, anime, daytime, sunny, empty street"

Using skills (progressive disclosure)
--------------------------------------
Skills are specialised instruction sets for specific upgrade patterns.  At startup
you can see each skill's name and description in <available_skills>.  When you decide
to apply a skill, call read_skill("<skill-name>") to load the full instructions.  Only
load skills you actually intend to use — each read consumes context.

Decision heuristics
-------------------
  Workflow is EMPTY (no nodes)          → read_skill("workflow-builder") FIRST.
                                         Then query_available_models to pick arch.
  Required model is missing / empty list → read_skill("model-downloader"), then call
                                         download_model_weights with a trusted URL.
  Workflow contains QwenImageModelLoader → read_skill("qwen-image-2512") FIRST.
                                         Qwen has NO KSampler/ControlNet/LoRA —
                                         all tuning is on RH_QwenImageGenerator.
  Active model name contains "lcm"     → read_skill("dreamshaper8-lcm") FIRST, before
                                         any sampler tuning — LCM needs different
                                         steps/cfg/sampler than standard SD models.
  Flat / low-depth background          → read_skill("controlnet-control"), add depth
  Blurry edges / lost structure        → read_skill("controlnet-control"), add canny
  Wrong human pose / body              → read_skill("controlnet-control"), add pose
  Plasticky skin / poor texture        → read_skill("lora-enhancement"), detail LoRA
  Wrong anatomy (hands, fingers)       → read_skill("lora-enhancement"), anatomy LoRA
  Style inconsistency                  → read_skill("lora-enhancement"), style LoRA
  Subject and background bleed         → read_skill("regional-control")
  Low resolution / soft fine detail    → read_skill("hires-fix")
  Localised artifact in one area       → add_inpaint_pass
  User asks for photorealistic image   → read_skill("photorealistic")
  User asks for high quality / sharp   → read_skill("high-quality")
  User asks for creative / artistic    → read_skill("creative")
  Prompt is flat / needs artistic depth→ read_skill("prompt-artist")
  Multiple objects with spatial layout → read_skill("spatial")
  Text / sign / label in the image     → read_skill("text-rendering")

Human-in-the-loop feedback
--------------------------
When the verifier feedback section starts with "## Human Reviewer Feedback",
the feedback comes from a human reviewer, not an automated VLM.
Human feedback expresses subjective preferences — style, mood, composition,
color palette, artistic direction.  Prioritize these over structural/technical
changes.  Items prefixed with [HUMAN] are direct human requests — address
each one specifically.  Do not second-guess or override human preferences.

Node parameter constraints (DO NOT violate)
-------------------------------------------
  UNETLoader weight_dtype:  "default" | "fp8_e4m3fn" | "fp8_e4m3fn_fast" | "fp8_e5m2"
                            Never use "fp16" or "fp32" — causes HTTP 400.
  Apple MPS cannot run FP8 models. If you see a Float8_e4m3fn MPS error,
  set weight_dtype to "default" and do not attempt further dtype changes.
  LoRA class is "LoraLoader" (not LoRALoader).
  ControlNet apply class is "ControlNetApplyAdvanced".

Model download protocol
-----------------------
If query_available_models returns an empty list, or a required checkpoint / UNET /
VAE / text encoder / LoRA is missing, you may call download_model_weights.
This tool asks the user for approval before downloading. Do not use guessed URLs:
use URLs from a skill, a known model bundle, or a URL the user supplied.
After a successful download, tell the user that ComfyUI may need a restart for
loader dropdowns to refresh, then continue with exact local filenames when possible.
"""

_PINNED_MODEL_SECTION = """\

Pinned image model
------------------
The image-generation model for this session is LOCKED to: {model_name}

  • DO NOT change the ckpt_name / unet_name of any CheckpointLoaderSimple,
    CheckpointLoader, or UNETLoader node.
  • DO NOT delete or replace existing loader nodes.
  • You MAY add LoRA loaders on top of the pinned model — LoraLoader wires
    between the primary loader and the sampler, leaving the base model intact.
  • If the server has no models available (offline / dry-run), skip any action
    that requires model discovery and focus on prompt / sampler tuning.
"""


def _build_system_prompt(
    pinned_image_model: str | None,
    available_skills_xml: str = "",
) -> str:
    """Return the full system prompt with optional pinned-model and skills sections."""
    prompt = _SYSTEM_PROMPT_BASE
    if available_skills_xml:
        prompt += f"\n{available_skills_xml}\n"
    if pinned_image_model:
        prompt += _PINNED_MODEL_SECTION.format(model_name=pinned_image_model)
    return prompt


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def _tool(name: str, description: str, parameters: dict) -> dict:
    """Wrap a tool definition in OpenAI / LiteLLM function-calling format."""
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


_TOOLS: list[dict] = [
    _tool(
        "inspect_workflow",
        "Return a human-readable summary of all nodes, IDs, class types, and key inputs.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _tool(
        "query_available_models",
        (
            "Query the ComfyUI server for available models of a given type. "
            "Call this BEFORE adding LoRA or ControlNet nodes."
        ),
        {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": "loras | controlnets | checkpoints | unets | vae | upscale_models | clip_vision | clip",
                }
            },
            "required": ["model_type"],
        },
    ),
    _tool(
        "download_model_weights",
        (
            "Ask the user for approval, then download model weights into ComfyUI/models. "
            "Use this when query_available_models shows required weights are missing."
        ),
        {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Hugging Face resolve/blob/raw URL or direct http(s) model URL.",
                },
                "dest_subdir": {
                    "type": "string",
                    "enum": list(MODEL_DEST_SUBDIRS),
                    "description": "ComfyUI models subdirectory, e.g. checkpoints, diffusion_models, vae, loras.",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional destination filename. Leave empty to use the URL filename.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short user-facing reason why this model is needed.",
                },
            },
            "required": ["url", "dest_subdir", "reason"],
        },
    ),
    _tool(
        "set_param",
        "Set a scalar input on a specific node.",
        {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "param_name": {"type": "string"},
                "value": {},
            },
            "required": ["node_id", "param_name", "value"],
        },
    ),
    _tool(
        "add_node",
        "Append a new node to the workflow. Returns the new node ID.",
        {
            "type": "object",
            "properties": {
                "class_type": {"type": "string"},
                "nickname": {"type": "string"},
                "inputs": {"type": "object"},
            },
            "required": ["class_type"],
        },
    ),
    _tool(
        "connect_nodes",
        "Wire src_node output slot to dst_node input name.",
        {
            "type": "object",
            "properties": {
                "src_node_id": {"type": "string"},
                "src_output_index": {"type": "integer"},
                "dst_node_id": {"type": "string"},
                "dst_input_name": {"type": "string"},
            },
            "required": ["src_node_id", "src_output_index", "dst_node_id", "dst_input_name"],
        },
    ),
    _tool(
        "delete_node",
        "Remove a node and clean up its connections.",
        {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    ),
    _tool(
        "add_lora_loader",
        (
            "Insert a LoraLoader between the model/clip source and all downstream consumers. "
            "Call query_available_models('loras') first."
        ),
        {
            "type": "object",
            "properties": {
                "lora_name": {"type": "string"},
                "strength_model": {"type": "number"},
                "strength_clip": {"type": "number"},
                "model_node_id": {"type": "string"},
                "clip_node_id": {"type": "string"},
            },
            "required": ["lora_name", "model_node_id", "clip_node_id"],
        },
    ),
    _tool(
        "add_controlnet",
        (
            "Add ControlNetLoader + optional preprocessor + ControlNetApplyAdvanced. "
            "Call query_available_models('controlnets') first."
        ),
        {
            "type": "object",
            "properties": {
                "controlnet_name": {"type": "string"},
                "preprocessor_class": {"type": "string"},
                "image_node_id": {"type": "string"},
                "positive_node_id": {"type": "string"},
                "negative_node_id": {"type": "string"},
                "strength": {"type": "number"},
                "start_percent": {"type": "number"},
                "end_percent": {"type": "number"},
            },
            "required": [
                "controlnet_name",
                "preprocessor_class",
                "positive_node_id",
                "negative_node_id",
            ],
        },
    ),
    _tool(
        "add_regional_attention",
        (
            "Split conditioning into foreground and background regional prompts using "
            "BREAK tokens + ConditioningCombine."
        ),
        {
            "type": "object",
            "properties": {
                "positive_node_id": {"type": "string"},
                "clip_node_id": {"type": "string"},
                "foreground_prompt": {"type": "string"},
                "background_prompt": {"type": "string"},
                "foreground_weight": {"type": "number"},
            },
            "required": [
                "positive_node_id",
                "clip_node_id",
                "foreground_prompt",
                "background_prompt",
            ],
        },
    ),
    _tool(
        "add_hires_fix",
        "Add a hires-fix pass: LatentUpscaleBy + second KSampler + VAEDecode.",
        {
            "type": "object",
            "properties": {
                "upscale_method": {"type": "string"},
                "scale_by": {"type": "number"},
                "hires_steps": {"type": "integer"},
                "hires_denoise": {"type": "number"},
                "base_ksampler_node_id": {"type": "string"},
                "vae_node_id": {"type": "string"},
                "save_image_node_id": {"type": "string"},
            },
            "required": ["base_ksampler_node_id", "vae_node_id"],
        },
    ),
    _tool(
        "add_inpaint_pass",
        "Add a targeted inpaint pass for a specific region.",
        {
            "type": "object",
            "properties": {
                "region_description": {"type": "string"},
                "inpaint_prompt": {"type": "string"},
                "denoise_strength": {"type": "number"},
                "base_ksampler_node_id": {"type": "string"},
                "positive_node_id": {"type": "string"},
                "clip_node_id": {"type": "string"},
                "vae_node_id": {"type": "string"},
                "save_image_node_id": {"type": "string"},
            },
            "required": [
                "region_description",
                "inpaint_prompt",
                "base_ksampler_node_id",
                "positive_node_id",
                "clip_node_id",
                "vae_node_id",
            ],
        },
    ),
    _tool(
        "set_prompt",
        (
            "Set the positive and/or negative prompt text across the whole workflow. "
            "Automatically locates the CLIPTextEncode (or equivalent) nodes connected "
            "to every sampler — no node ID required. "
            "Call this EARLY to replace the raw user goal with a detailed, "
            "professional-quality prompt.  Pass empty string to leave a slot unchanged."
        ),
        {
            "type": "object",
            "properties": {
                "positive_text": {
                    "type": "string",
                    "description": (
                        "Expanded positive prompt.  Structure: "
                        "[subject & scene], [style], [lighting], [camera/lens], [quality boosters].  "
                        "Example: 'a futuristic city skyline at night, neon reflections on wet streets, "
                        "cyberpunk aesthetic, dramatic cinematic lighting, wide angle lens, "
                        "8k, photorealistic, ultra detailed, sharp focus, ray tracing'."
                    ),
                },
                "negative_text": {
                    "type": "string",
                    "description": (
                        "Negative prompt listing artefacts and failure modes to suppress.  "
                        "Always include baseline: 'blurry, low quality, low resolution, noisy, "
                        "watermark, text, bad anatomy, deformed, ugly'.  "
                        "Add model- or scene-specific negatives as needed."
                    ),
                },
            },
            "required": [],
        },
    ),
    _tool(
        "report_evolution_strategy",
        "Declare your evolution plan BEFORE making changes.",
        {
            "type": "object",
            "properties": {
                "strategy": {"type": "string"},
                "top_issue": {"type": "string"},
            },
            "required": ["strategy", "top_issue"],
        },
    ),
    _tool(
        "validate_workflow",
        (
            "Check the workflow graph for common errors BEFORE finalizing. "
            "Returns a list of issues (dangling refs, wrong slot indices, missing outputs). "
            "Call this after building or repairing the workflow to catch mistakes early."
        ),
        {"type": "object", "properties": {}, "required": []},
    ),
    _tool(
        "answer_user",
        (
            "Answer the user's question directly without modifying or generating the workflow. "
            "Use this when the user is asking for explanation, diagnosis, or guidance rather "
            "than asking you to create, change, or run the ComfyUI workflow."
        ),
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    ),
    _tool(
        "finalize_workflow",
        "Signal that all modifications are complete. Call validate_workflow first to catch errors.",
        {
            "type": "object",
            "properties": {"rationale": {"type": "string"}},
            "required": ["rationale"],
        },
    ),
    _tool(
        "read_skill",
        (
            "Load the full instructions for a named skill (progressive disclosure). "
            "Call this BEFORE applying a skill's technique. "
            "Available skill names are listed in <available_skills> in the system prompt."
        ),
        {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Exact skill name as shown in <available_skills>.",
                },
            },
            "required": ["skill_name"],
        },
    ),
]


_GROK_REDACTED_INPUT_KEYS = {
    "caption",
    "description",
    "negative",
    "negative_prompt",
    "positive",
    "positive_prompt",
    "prompt",
    "text",
    "text_g",
    "text_l",
}


def _summarize_workflow_for_grok(workflow: dict) -> str:
    """Summarize topology without forwarding stored prompt prose to Grok.

    Grok still receives the user's current request.  This only prevents an
    unrelated workflow's stored prompt text from contaminating structural
    operations such as adding or reconnecting a node.
    """
    if not workflow:
        return "(empty workflow)"
    lines = ["Workflow nodes (stored prompt text omitted):"]
    for nid in sorted(workflow.keys(), key=lambda value: int(value) if value.isdigit() else 0):
        node = workflow[nid]
        title = node.get("_meta", {}).get("title") or node.get("class_type", "?")
        inputs_repr: list[str] = []
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                inputs_repr.append(f"{key}=→node{value[0]}[{value[1]}]")
            elif isinstance(value, str) and (
                node.get("class_type") in ("CLIPTextEncode", "CLIPTextEncodeSDXL")
                or key.lower() in _GROK_REDACTED_INPUT_KEYS
            ):
                inputs_repr.append(f"{key}=<stored text omitted; {len(value)} chars>")
            else:
                value_repr = repr(value) if isinstance(value, str) else json.dumps(value)
                inputs_repr.append(f"{key}={value_repr}")
        lines.append(
            f"  [{nid}] {title} ({node.get('class_type', '?')})"
            + (f"  {', '.join(inputs_repr)}" if inputs_repr else "")
        )
    lines.append(
        "Stored prompt prose is intentionally unavailable in Grok subscription mode. "
        "Use only the current user request and structural metadata; do not infer omitted text."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ClawAgent:
    """
    Parameters
    ----------
    api_key         : API key for the LLM provider.  For Anthropic this is the
                      ``ANTHROPIC_API_KEY``; for OpenAI ``OPENAI_API_KEY``, etc.
                      When provided it is written into the matching env-var so
                      that LiteLLM can pick it up automatically.  You can also
                      set the env-var directly and leave this empty.
    model           : LiteLLM model string, e.g. ``"anthropic/claude-sonnet-4-5"``,
                      ``"openai/gpt-5.4"``, ``"gemini/gemini-2.0-flash"``,
                      ``"ollama/llama3.1"``.
    server_address  : ComfyUI HTTP address (used for model queries).
    skills_dir      : Path to skills/ folder; ``None`` uses built-in skills.
    on_change       : Called with the workflow dict after every mutation.
    max_tool_rounds : Safety cap on tool-call iterations.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "anthropic/claude-sonnet-4-5",
        server_address: str = "127.0.0.1:8188",
        skills_dir: str | None = None,
        on_change: Callable[[dict], None] | None = None,
        max_tool_rounds: int = 40,
        pinned_image_model: str | None = None,
        backend: AgentBackend | None = None,
        backend_name: str = "litellm",
        api_base: str | None = None,
        agent_session_id: str = "",
        model_download_callback: Callable[[dict], dict] | None = None,
    ) -> None:
        # Propagate the API key into the environment so LiteLLM picks it up.
        # For Anthropic keys (sk-ant-…) we set ANTHROPIC_API_KEY; for other
        # providers users should set the appropriate env-var themselves.
        if api_key:
            if api_key.startswith("sk-ant-"):
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            else:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        self.model = model
        self.server_address = server_address
        # SkillsRegistry layers in built-in + user roots automatically, so
        # the agent has its full skill recipe library even when callers
        # construct a harness with ``skills_dir=None``.
        self.skill_manager = SkillsRegistry(skills_dir=skills_dir)
        self.on_change = on_change
        self.on_agent_event: Callable[[str, str, str, dict | None], None] | None = None
        self.max_tool_rounds = max_tool_rounds
        self.pinned_image_model = pinned_image_model
        self.model_download_callback = model_download_callback
        self.last_direct_answer = ""

        # Backend selection: explicit instance wins, otherwise build by name.
        if backend is not None:
            self.backend = backend
        else:
            self.backend = get_backend(
                backend_name,
                model=model,
                api_key=api_key,
                api_base=api_base,
                extra={"session_id": agent_session_id},
            )
        self.backend_name = getattr(self.backend, "name", backend_name)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def plan_and_patch(
        self,
        workflow_manager: WorkflowManager,
        original_prompt: str,
        verifier_feedback: str | None = None,
        memory_summary: str | None = None,
        iteration: int = 1,
    ) -> str:
        """
        Drive the LLM tool-use loop to evolve the workflow.

        The actual conversation runs inside the configured
        :class:`~comfyclaw.agent_backends.AgentBackend` (LiteLLM by
        default; Claude Code / Codex / Gemini CLI sessions optional).

        Returns the rationale string from the ``finalize_workflow`` call.
        """
        user_content = self._build_user_message(
            original_prompt=original_prompt,
            workflow_manager=workflow_manager,
            verifier_feedback=verifier_feedback,
            memory_summary=memory_summary,
            iteration=iteration,
        )

        system_prompt = _build_system_prompt(
            self.pinned_image_model,
            available_skills_xml=self.skill_manager.build_available_skills_xml(),
        )

        def _dispatch_call(call: ToolCall) -> tuple[str, bool]:
            return self._dispatch(call.name, call.args, workflow_manager)

        self.last_direct_answer = ""
        return self.backend.run_tool_loop(
            system=system_prompt,
            user=user_content,
            tools=_TOOLS,
            dispatch=_dispatch_call,
            on_event=self.on_agent_event,
            max_rounds=self.max_tool_rounds,
        )

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, name: str, inputs: dict, wm: WorkflowManager) -> tuple[str, bool]:
        """Route a single tool call. Returns ``(result_text, should_stop)``."""
        try:
            match name:
                case "inspect_workflow":
                    if self.backend_name == "grok-cli":
                        return _summarize_workflow_for_grok(wm.workflow), False
                    return WorkflowManager.summarize(wm.workflow), False

                case "query_available_models":
                    return self._query_models(inputs["model_type"]), False

                case "download_model_weights":
                    return self._download_model_weights(inputs), False

                case "set_prompt":
                    pos = inputs.get("positive_text") or ""
                    neg = inputs.get("negative_text") or ""
                    pos_nodes, neg_nodes = wm.inject_prompt(
                        positive=pos if pos else None,
                        negative=neg if neg else None,
                    )
                    self._notify(wm)
                    parts = []
                    if pos_nodes:
                        parts.append(f"positive → nodes {pos_nodes}: {pos!r}")
                    if neg_nodes:
                        parts.append(f"negative → nodes {neg_nodes}: {neg!r}")
                    if not parts:
                        return (
                            "⚠️ set_prompt: no CLIPTextEncode nodes found connected to a sampler. Use set_param directly.",
                            False,
                        )
                    return "✅ " + "; ".join(parts), False

                case "set_param":
                    wm.set_param(str(inputs["node_id"]), inputs["param_name"], inputs["value"])
                    self._notify(wm)
                    return (
                        f"✅ Node {inputs['node_id']}: {inputs['param_name']} = {json.dumps(inputs['value'])}",
                        False,
                    )

                case "add_node":
                    nid = wm.add_node(
                        inputs["class_type"],
                        inputs.get("nickname"),
                        **(inputs.get("inputs") or {}),
                    )
                    self._notify(wm)
                    return f"✅ Added node {nid} ({inputs['class_type']})", False

                case "connect_nodes":
                    wm.connect(
                        str(inputs["src_node_id"]),
                        int(inputs.get("src_output_index", 0)),
                        str(inputs["dst_node_id"]),
                        inputs["dst_input_name"],
                    )
                    self._notify(wm)
                    return (
                        f"✅ Connected {inputs['src_node_id']}[{inputs.get('src_output_index', 0)}]"
                        f" → {inputs['dst_node_id']}.{inputs['dst_input_name']}"
                    ), False

                case "delete_node":
                    wm.delete_node(str(inputs["node_id"]))
                    self._notify(wm)
                    return f"✅ Deleted node {inputs['node_id']}", False

                case "add_lora_loader":
                    return self._add_lora(wm, inputs), False

                case "add_controlnet":
                    return self._add_controlnet(wm, inputs), False

                case "add_regional_attention":
                    return self._add_regional_attention(wm, inputs), False

                case "add_hires_fix":
                    return self._add_hires_fix(wm, inputs), False

                case "add_inpaint_pass":
                    return self._add_inpaint_pass(wm, inputs), False

                case "report_evolution_strategy":
                    print(
                        f"[ClawAgent] Strategy: {inputs['strategy']}\n"
                        f"            Top issue: {inputs['top_issue']}"
                    )
                    self._emit_event(
                        "strategy",
                        f"**Strategy:** {inputs['strategy']}\n**Top issue:** {inputs['top_issue']}",
                    )
                    return "Strategy noted.", False

                case "validate_workflow":
                    errs = WorkflowManager.validate(wm.workflow)
                    if errs:
                        msg = "⚠️ Validation found issues:\n" + "\n".join(f"  • {e}" for e in errs)
                        msg += "\n\nFix these before calling finalize_workflow."
                        print(f"[ClawAgent] ⚠️ Validation: {len(errs)} issue(s)")
                        self._emit_event("validation", f"⚠️ {len(errs)} issue(s) found")
                        return msg, False
                    node_count = len(wm.workflow)
                    print(f"[ClawAgent] ✅ Validation passed ({node_count} nodes)")
                    self._emit_event("validation", f"✅ Valid ({node_count} nodes)")
                    return f"✅ Workflow is valid ({node_count} nodes, no issues found).", False

                case "finalize_workflow":
                    errs = WorkflowManager.validate(wm.workflow)
                    if errs:
                        msg = (
                            "⚠️ Auto-validation found issues — fix before finalizing:\n"
                            + "\n".join(f"  • {e}" for e in errs)
                        )
                        print(f"[ClawAgent] ⚠️ Finalize blocked: {len(errs)} validation error(s)")
                        return msg, False
                    print(f"[ClawAgent] 🎯 {inputs.get('rationale', '')}")
                    return "Workflow finalized.", True

                case "answer_user":
                    answer = str(inputs.get("answer") or "").strip()
                    self.last_direct_answer = answer
                    self._emit_event("assistant_done", answer or "(no answer)")
                    return "Answered user without workflow changes.", True

                case "read_skill":
                    return self._read_skill(inputs["skill_name"]), False

                case _:
                    return f"❌ Unknown tool: {name}", False

        except Exception as exc:
            return f"❌ Tool error ({name}): {exc}", False

    # ------------------------------------------------------------------
    # Skill reader (progressive disclosure — stage 2)
    # ------------------------------------------------------------------

    def _read_skill(self, skill_name: str) -> str:
        """
        Load the full body of a named skill on demand.

        This is the **stage-2** of progressive disclosure: the agent calls
        ``read_skill`` from within a tool-use loop after deciding which skill
        to apply.  The body text is returned as the tool result so the agent
        can follow the instructions immediately.
        """
        try:
            body = self.skill_manager.get_body(skill_name)
        except KeyError:
            available = ", ".join(self.skill_manager.skill_names)
            return f"❌ Skill {skill_name!r} not found. Available skills: {available or '(none)'}"
        print(f"[ClawAgent] 📖 read_skill: {skill_name}")
        return f"## Instructions for skill: {skill_name}\n\n{body}"

    # ------------------------------------------------------------------
    # Topology builders
    # ------------------------------------------------------------------

    def _add_lora(self, wm: WorkflowManager, inputs: dict) -> str:
        lora_name = inputs["lora_name"]
        model_nid = str(inputs["model_node_id"])
        clip_nid = str(inputs["clip_node_id"])
        sm = float(inputs.get("strength_model", 0.8))
        sc = float(inputs.get("strength_clip", 0.8))

        lora_nid = wm.add_node(
            "LoraLoader",
            f"LoRA: {lora_name[:30]}",
            model=[model_nid, 0],
            clip=[clip_nid, 0],
            lora_name=lora_name,
            strength_model=sm,
            strength_clip=sc,
        )
        self._notify(wm)

        rewired_model, rewired_clip = [], []
        for nid, node in wm.workflow.items():
            if nid == lora_nid:
                continue
            for inp_name, inp_val in list(node.get("inputs", {}).items()):
                if isinstance(inp_val, list) and len(inp_val) == 2:
                    src_id, src_idx = str(inp_val[0]), inp_val[1]
                    if src_id == model_nid and src_idx == 0 and inp_name == "model":
                        wm.workflow[nid]["inputs"][inp_name] = [lora_nid, 0]
                        rewired_model.append(f"{nid}.{inp_name}")
                    if src_id == clip_nid and src_idx == 0 and inp_name == "clip":
                        wm.workflow[nid]["inputs"][inp_name] = [lora_nid, 1]
                        rewired_clip.append(f"{nid}.{inp_name}")

        self._notify(wm)
        return (
            f"✅ LoraLoader {lora_nid} ({lora_name}, sm={sm}, sc={sc})\n"
            f"   Re-wired model: {rewired_model}\n"
            f"   Re-wired clip:  {rewired_clip}"
        )

    def _add_controlnet(self, wm: WorkflowManager, inputs: dict) -> str:
        cn_name = inputs["controlnet_name"]
        preproc_cls = inputs.get("preprocessor_class", "")
        image_nid = str(inputs.get("image_node_id", ""))
        pos_nid = str(inputs["positive_node_id"])
        neg_nid = str(inputs["negative_node_id"])
        strength = float(inputs.get("strength", 0.7))
        start_pct = float(inputs.get("start_percent", 0.0))
        end_pct = float(inputs.get("end_percent", 1.0))

        cn_loader_nid = wm.add_node(
            "ControlNetLoader", f"CN: {cn_name[:25]}", control_net_name=cn_name
        )
        self._notify(wm)

        preproc_output = None
        preproc_nid = None
        if preproc_cls and image_nid:
            preproc_nid = wm.add_node(preproc_cls, preproc_cls, image=[image_nid, 0])
            preproc_output = [preproc_nid, 0]
            self._notify(wm)
        elif image_nid:
            preproc_output = [image_nid, 0]

        apply_inputs: dict = {
            "positive": [pos_nid, 0],
            "negative": [neg_nid, 0],
            "control_net": [cn_loader_nid, 0],
            "strength": strength,
            "start_percent": start_pct,
            "end_percent": end_pct,
        }
        if preproc_output:
            apply_inputs["image"] = preproc_output

        apply_nid = wm.add_node("ControlNetApplyAdvanced", "CN Apply", **apply_inputs)
        self._notify(wm)

        for nid, node in wm.workflow.items():
            if node.get("class_type") == "KSampler":
                for inp_name, inp_val in list(node["inputs"].items()):
                    if isinstance(inp_val, list) and len(inp_val) == 2:
                        src = str(inp_val[0])
                        if src == pos_nid and inp_name == "positive":
                            wm.workflow[nid]["inputs"]["positive"] = [apply_nid, 0]
                        if src == neg_nid and inp_name == "negative":
                            wm.workflow[nid]["inputs"]["negative"] = [apply_nid, 1]

        self._notify(wm)
        parts = [
            "✅ ControlNet branch added:",
            f"   Loader:  {cn_loader_nid} ({cn_name})",
        ]
        if preproc_cls and image_nid:
            parts.append(f"   Preproc: {preproc_nid} ({preproc_cls})")
        parts.append(
            f"   Apply:   {apply_nid} (strength={strength}, {start_pct:.1f}→{end_pct:.1f})"
        )
        return "\n".join(parts)

    def _add_regional_attention(self, wm: WorkflowManager, inputs: dict) -> str:
        pos_nid = str(inputs["positive_node_id"])
        clip_nid = str(inputs["clip_node_id"])
        fg_prompt = inputs["foreground_prompt"]
        bg_prompt = inputs["background_prompt"]
        fg_weight = float(inputs.get("foreground_weight", 1.3))

        if pos_nid in wm.workflow:
            meta = wm.workflow[pos_nid].setdefault("_meta", {})
            meta["title"] = "Regional Prompt (BREAK)"
            wm.workflow[pos_nid]["inputs"]["text"] = f"{fg_prompt} BREAK {bg_prompt}"
            self._notify(wm)

        fg_nid = wm.add_node(
            "CLIPTextEncode", "Foreground Prompt", clip=[clip_nid, 0], text=fg_prompt
        )
        self._notify(wm)
        bg_nid = wm.add_node(
            "CLIPTextEncode", "Background Prompt", clip=[clip_nid, 0], text=bg_prompt
        )
        self._notify(wm)
        avg_nid = wm.add_node(
            "ConditioningAverage",
            "FG Weight",
            conditioning_to=[fg_nid, 0],
            conditioning_from=[pos_nid, 0],
            conditioning_to_strength=fg_weight,
        )
        self._notify(wm)
        combine_nid = wm.add_node(
            "ConditioningCombine",
            "Regional Combine",
            conditioning_1=[avg_nid, 0],
            conditioning_2=[bg_nid, 0],
        )
        self._notify(wm)

        for _nid, node in wm.workflow.items():
            if node.get("class_type") == "KSampler":
                node["inputs"]["positive"] = [combine_nid, 0]

        self._notify(wm)
        return (
            f"✅ Regional attention: FG={fg_nid}, BG={bg_nid}, Avg={avg_nid}, Combine={combine_nid}"
        )

    def _add_hires_fix(self, wm: WorkflowManager, inputs: dict) -> str:
        base_ks_nid = str(inputs["base_ksampler_node_id"])
        vae_nid = str(inputs["vae_node_id"])
        method = inputs.get("upscale_method", "bicubic")
        scale_by = float(inputs.get("scale_by", 1.5))
        hires_steps = int(inputs.get("hires_steps", 15))
        hires_denoise = float(inputs.get("hires_denoise", 0.45))
        save_nid = str(inputs.get("save_image_node_id", ""))

        existing_decode = next(
            (nid for nid, n in wm.workflow.items() if n.get("class_type") == "VAEDecode"),
            None,
        )
        if existing_decode:
            vae_connection = list(wm.workflow[existing_decode]["inputs"].get("vae", [vae_nid, 0]))
        else:
            vae_connection = [vae_nid, 0]

        upscale_nid = wm.add_node(
            "LatentUpscaleBy",
            "Hires Upscale",
            samples=[base_ks_nid, 0],
            upscale_method=method,
            scale_by=scale_by,
        )
        self._notify(wm)

        base_inputs = copy.deepcopy(wm.workflow.get(base_ks_nid, {}).get("inputs", {}))
        base_inputs["latent_image"] = [upscale_nid, 0]
        base_inputs["steps"] = hires_steps
        base_inputs["denoise"] = hires_denoise
        hires_ks_nid = wm.add_node("KSampler", "KSampler (Hires)", **base_inputs)
        self._notify(wm)

        decode_nid = wm.add_node(
            "VAEDecode", "VAEDecode (Hires)", samples=[hires_ks_nid, 0], vae=vae_connection
        )
        self._notify(wm)

        target = (
            save_nid
            if save_nid in wm.workflow
            else next(
                (nid for nid, n in wm.workflow.items() if n.get("class_type") == "SaveImage"), None
            )
        )
        if target:
            wm.workflow[target]["inputs"]["images"] = [decode_nid, 0]

        self._notify(wm)
        return (
            f"✅ Hires fix: Upscale={upscale_nid} (×{scale_by}), "
            f"KSampler={hires_ks_nid} ({hires_steps}st, d={hires_denoise}), Decode={decode_nid}"
        )

    def _add_inpaint_pass(self, wm: WorkflowManager, inputs: dict) -> str:
        region = inputs["region_description"]
        prompt_text = inputs["inpaint_prompt"]
        denoise = float(inputs.get("denoise_strength", 0.55))
        base_ks_nid = str(inputs["base_ksampler_node_id"])
        clip_nid = str(inputs["clip_node_id"])
        vae_nid = str(inputs["vae_node_id"])
        save_nid = str(inputs.get("save_image_node_id", ""))

        ip_pos_nid = wm.add_node(
            "CLIPTextEncode",
            f"Inpaint Prompt ({region[:20]})",
            clip=[clip_nid, 0],
            text=prompt_text,
        )
        self._notify(wm)

        base_inp = copy.deepcopy(wm.workflow.get(base_ks_nid, {}).get("inputs", {}))
        base_inp["positive"] = [ip_pos_nid, 0]
        base_inp["latent_image"] = [base_ks_nid, 0]
        base_inp["denoise"] = denoise

        existing_decode = next(
            (nid for nid, n in wm.workflow.items() if n.get("class_type") == "VAEDecode"),
            None,
        )
        vae_connection = (
            list(wm.workflow[existing_decode]["inputs"].get("vae", [vae_nid, 0]))
            if existing_decode
            else [vae_nid, 0]
        )

        ip_ks_nid = wm.add_node("KSampler", f"KSampler Inpaint ({region[:20]})", **base_inp)
        self._notify(wm)

        ip_decode_nid = wm.add_node(
            "VAEDecode", "VAEDecode Inpaint", samples=[ip_ks_nid, 0], vae=vae_connection
        )
        self._notify(wm)

        target = (
            save_nid
            if save_nid in wm.workflow
            else next(
                (nid for nid, n in wm.workflow.items() if n.get("class_type") == "SaveImage"), None
            )
        )
        if target:
            wm.workflow[target]["inputs"]["images"] = [ip_decode_nid, 0]

        self._notify(wm)
        return (
            f"✅ Inpaint pass [{region}]: Prompt={ip_pos_nid}, "
            f"KSampler={ip_ks_nid} (d={denoise}), Decode={ip_decode_nid}"
        )

    # ------------------------------------------------------------------
    # ComfyUI model queries
    # ------------------------------------------------------------------

    def _query_models(self, model_type: str) -> str:
        type_map = {
            "loras": ("LoraLoader", "lora_name"),
            "controlnets": ("ControlNetLoader", "control_net_name"),
            "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
            "unets": ("UNETLoader", "unet_name"),
            "vae": ("VAELoader", "vae_name"),
            "upscale_models": ("UpscaleModelLoader", "model_name"),
            "clip_vision": ("CLIPVisionLoader", "clip_name"),
            "clip": ("CLIPLoader", "clip_name"),
        }
        entry = type_map.get(model_type.lower())
        if not entry:
            return f"❌ Unknown model_type {model_type!r}. Valid: {list(type_map)}"
        node_class, param = entry
        try:
            url = f"http://{self.server_address}/object_info/{urllib.parse.quote(node_class)}"
            with urllib.request.urlopen(url, timeout=8) as r:
                data = json.loads(r.read())
            models = (
                data.get(node_class, {}).get("input", {}).get("required", {}).get(param, [[]])[0]
            )
            if not models:
                return f"No {model_type} found (ComfyUI returned empty list)."
            return f"Available {model_type} ({len(models)}):\n" + "\n".join(
                f"  - {m}" for m in models
            )
        except Exception as exc:
            return f"❌ Could not query {model_type}: {exc}"

    def _download_model_weights(self, inputs: dict) -> str:
        url = str(inputs.get("url") or "").strip()
        dest_subdir = str(inputs.get("dest_subdir") or "").strip()
        filename = str(inputs.get("filename") or "").strip() or None
        reason = str(inputs.get("reason") or "").strip()
        if not url:
            return "❌ download_model_weights: url is required."
        if dest_subdir not in MODEL_DEST_SUBDIRS:
            return (
                "❌ download_model_weights: dest_subdir must be one of "
                f"{', '.join(MODEL_DEST_SUBDIRS)}."
            )
        if self.model_download_callback is None:
            return (
                "⚠️ Model download requires user approval, but no ComfyClaw panel is connected "
                "for this run. Ask the user to open the panel or run `comfyclaw models download-url`."
            )

        request = {
            "url": url,
            "dest_subdir": dest_subdir,
            "filename": filename or "",
            "reason": reason,
        }
        decision = self.model_download_callback(request)
        if not decision.get("approved"):
            why = decision.get("reason") or "User declined the download."
            return f"⚠️ Model download not approved: {why}"

        comfyui_dir = (
            Path(os.environ["COMFYUI_DIR"]).expanduser()
            if os.environ.get("COMFYUI_DIR", "").strip()
            else Path.home() / "Documents" / "ComfyUI"
        )
        self._emit_event(
            "tool_result",
            f"Downloading model weights to {comfyui_dir / 'models' / dest_subdir}...",
            "download_model_weights",
            None,
        )
        target = download_model_from_url(
            url,
            comfyui_dir=comfyui_dir,
            dest_subdir=dest_subdir,
            filename=filename,
        )
        return (
            f"✅ Downloaded model weights to {target}. "
            "Restart ComfyUI if the loader dropdown does not show the new file yet."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _notify(self, wm: WorkflowManager) -> None:
        if self.on_change:
            try:
                self.on_change(wm.to_dict())
            except Exception as exc:
                print(f"[ClawAgent] on_change error: {exc}", file=sys.stderr)

    def _emit_event(
        self,
        event_type: str,
        content: str,
        tool_name: str = "",
        tool_args: dict | None = None,
    ) -> None:
        """Fire an agent-thinking event if a listener is attached."""
        if self.on_agent_event:
            try:
                self.on_agent_event(event_type, content, tool_name, tool_args)
            except Exception:
                pass

    def _build_user_message(
        self,
        original_prompt: str,
        workflow_manager: WorkflowManager | None,
        verifier_feedback: str | None,
        memory_summary: str | None,
        iteration: int,
    ) -> str:
        parts = [
            f"## User Input\n{original_prompt}",
            f"## Iteration\n{iteration}",
            (
                "## Decision Required\n"
                "First decide whether this input is conversational or requests workflow work. "
                "For conversational input, call `answer_user` and stop. For workflow work, "
                "continue with strategy, inspection, edits, validation, and finalization."
            ),
        ]

        # Expose the current positive prompt text so the agent can see exactly
        # what the baseline looks like and craft an improved version.
        if workflow_manager:
            current_positive: str | None = None
            for node in workflow_manager.workflow.values():
                if node.get("class_type") in ("CLIPTextEncode", "CLIPTextEncodeSDXL"):
                    text_val = node.get("inputs", {}).get("text") or node.get("inputs", {}).get(
                        "text_g"
                    )
                    if isinstance(text_val, str) and text_val.strip():
                        current_positive = text_val.strip()
                        break
            if current_positive and self.backend_name != "grok-cli":
                parts.append(
                    f"## Current Positive Prompt\n{current_positive}\n\n"
                    "Use this only if you decide the user is asking for workflow work."
                )
            elif current_positive:
                parts.append(
                    "## Current Positive Prompt\n"
                    "(stored text omitted in Grok subscription mode; use the current user "
                    "request and structural metadata only)"
                )
            else:
                parts.append("## Current Positive Prompt\n(none)")

        # Detect the active checkpoint / UNET model from the workflow and surface
        # it prominently so the agent can match it against model-specific skills
        # without having to call inspect_workflow first.
        active_model: str | None = None
        if workflow_manager:
            _loader_classes = ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader")
            _model_params = ("ckpt_name", "unet_name")
            for node in workflow_manager.workflow.values():
                if node.get("class_type") in _loader_classes:
                    for param in _model_params:
                        val = node.get("inputs", {}).get(param)
                        if val and isinstance(val, str):
                            active_model = val
                            break
                if active_model:
                    break

        # Detect Qwen-Image pipeline — either via custom pipeline nodes (old plugin)
        # or via native ComfyUI UNETLoader with a qwen model filename.
        _qwen_plugin_classes = {"QwenImageModelLoader", "RH_QwenImageGenerator"}
        _qwen_unet_names = ("qwen_image",)

        def _node_is_qwen_unet(node: dict) -> bool:
            if node.get("class_type") != "UNETLoader":
                return False
            name = str(node.get("inputs", {}).get("unet_name", "")).lower()
            return any(kw in name for kw in _qwen_unet_names)

        is_qwen = workflow_manager is not None and any(
            node.get("class_type") in _qwen_plugin_classes or _node_is_qwen_unet(node)
            for node in workflow_manager.workflow.values()
        )

        if is_qwen:
            parts.append(
                "## Active Model\n`Qwen-Image-2512` (custom pipeline — no KSampler/ControlNet/LoRA)"
            )
        elif active_model:
            parts.append(f"## Active Model\n`{active_model}`")

        # Hint at relevant skills (names only — full instructions loaded via read_skill)
        # Also suggest model-specific skills based on the active checkpoint name.
        relevant = self.skill_manager.detect_relevant_skills(original_prompt)
        # Honour explicit "/skill <name>" directives typed by the user: promote
        # those to the front so the agent loads them, and emit a hard directive.
        forced_skills: list[str] = []
        valid_names = set(self.skill_manager.skill_names)
        for m in re.finditer(r"/skills?\s+([a-zA-Z0-9 ,_\-]+)", original_prompt or ""):
            for tok in re.split(r"[\s,]+", m.group(1).strip()):
                name = tok.strip().lower()
                if name in valid_names and name not in forced_skills:
                    forced_skills.append(name)
        for name in reversed(forced_skills):
            if name in relevant:
                relevant.remove(name)
            relevant.insert(0, name)
        if workflow_manager is not None and len(workflow_manager.workflow) == 0:
            if "workflow-builder" not in relevant:
                relevant.insert(0, "workflow-builder")
        if is_qwen:
            if "qwen-image-2512" not in relevant:
                relevant.insert(0, "qwen-image-2512")
        elif active_model:
            for skill_name in self.skill_manager.skill_names:
                # Simple substring match: skill name keywords appear in model filename
                skill_keywords = skill_name.replace("-", " ").split()
                model_lower = active_model.lower()
                if all(kw in model_lower for kw in skill_keywords if len(kw) > 2):
                    if skill_name not in relevant:
                        relevant.append(skill_name)
        if relevant:
            hint = ", ".join(sorted(relevant))
            forced_line = ""
            if forced_skills:
                forced_line = (
                    f"\n**The user explicitly requested these skills via /skill: "
                    f'{", ".join(forced_skills)}. You MUST call read_skill on each of '
                    "them and apply their instructions.**"
                )
            parts.append(
                f"## Suggested Skills\nThese skills may be relevant: {hint}\n"
                "Call read_skill(<name>) to load full instructions before applying.\n"
                '**If the active model is an LCM variant, read_skill("dreamshaper8-lcm") FIRST.**\n'
                '**If the workflow contains QwenImageModelLoader, read_skill("qwen-image-2512") FIRST.**'
                f"{forced_line}"
            )

        if verifier_feedback:
            if "[HUMAN]" in verifier_feedback or "[Human override]" in verifier_feedback:
                parts.append(
                    f"## Human Reviewer Feedback (previous iteration)\n{verifier_feedback}\n\n"
                    "The human reviewer has given you specific feedback about the image quality.\n"
                    "Prioritize their subjective preferences (style, mood, composition, color) "
                    "over structural or technical changes. Address their feedback directly."
                )
            else:
                parts.append(
                    f"## Verifier Feedback (previous iteration)\n{verifier_feedback}\n\n"
                    "Use the evolution_suggestions and region_issues above to decide which "
                    "structural upgrade to apply this round."
                )
        if memory_summary:
            parts.append(f"## Memory / Past Attempts\n{memory_summary}")

        if workflow_manager is not None and len(workflow_manager.workflow) == 0:
            parts.append(
                "## Empty Workflow Context\n"
                "The workflow is empty. If and only if the user is asking for generation or "
                'workflow construction, build from scratch: read_skill("workflow-builder"), '
                "query available models, add and connect nodes, set prompts, validate, then "
                "finalize_workflow. If the input is conversational, answer_user instead."
            )

        parts.append(
            "Do not call workflow tools until you have decided the input asks for workflow work."
        )
        return "\n\n".join(parts)
