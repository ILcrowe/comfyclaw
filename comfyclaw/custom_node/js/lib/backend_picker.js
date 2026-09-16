/**
 * Backend picker — chooses the Python AgentBackend driver.
 *
 *   • LiteLLM    — direct API calls (requires API key).
 *   • Claude Code — uses the `claude` CLI (must be installed + signed in).
 *   • Codex      — uses the `codex` CLI.
 *   • Gemini CLI — uses the `gemini` CLI.
 *   • Grok CLI   — uses the `grok` CLI with a subscription login.
 *
 * Availability is fed in from the server via the ``agent_backends`` WS
 * message (see ``SyncServer._dispatch`` -> ``list_agent_backends``).
 *
 * The status map per backend is now an object::
 *
 *   {
 *     state: "ok" | "needs_install" | "needs_auth" | "unsupported" | "error",
 *     binary_path: string,
 *     auth_method: string,
 *     detail: string,
 *     can_install: boolean,
 *   }
 *
 * Booleans are still accepted for backward compatibility.
 *
 * Backends in ``needs_install`` / ``needs_auth`` states are visually
 * distinct (red / amber dot) and clickable: the click fires
 * ``onAction(id, action)`` where ``action`` is ``"install"`` or
 * ``"signin"`` — the caller is expected to open the matching modal.
 */

import { escHtml } from "./util.js";
import { showToast } from "./toast.js";

const BACKENDS = [
  { id: "litellm",     label: "LiteLLM",      desc: "Direct API (Anthropic, OpenAI, Gemini, Groq, Ollama, …)",
    letter: "L", brand: "#7287fd" },
  { id: "claude-code", label: "Claude Code",  desc: "Uses the `claude` CLI.",
    letter: "C", brand: "#cc785c" },
  { id: "codex",       label: "Codex",        desc: "Uses the `codex` CLI.",
    letter: "O", brand: "#10a37f" },
  { id: "gemini-cli",  label: "Gemini CLI",   desc: "Uses the `gemini` CLI.",
    letter: "G", brand: "#4285f4" },
  { id: "grok-cli",    label: "Grok CLI",     desc: "Uses `grok login` (subscription OAuth).",
    letter: "X", brand: "#242424" },
];

function _logoHtml(meta, size = 14) {
  return `<span style="display:inline-flex;align-items:center;justify-content:center;
                width:${size}px;height:${size}px;border-radius:3px;
                background:${meta.brand};color:#fff;
                font-weight:700;font-size:${Math.max(8, Math.round(size * 0.62))}px;
                font-family:system-ui,-apple-system,sans-serif;
                line-height:1;flex-shrink:0;
                text-shadow:0 1px 0 rgba(0,0,0,0.25);">${escHtml(meta.letter)}</span>`;
}

function _readSaved() {
  return localStorage.getItem("comfyclaw_agent_backend") || "litellm";
}

function _normalizeStatus(raw) {
  // Boolean form (legacy): true -> ok, false -> unsupported.
  if (raw === true)  return { state: "ok",          detail: "", binary_path: "", auth_method: "", can_install: false };
  if (raw === false) return { state: "unsupported", detail: "", binary_path: "", auth_method: "", can_install: false };
  if (!raw || typeof raw !== "object") return null;
  return {
    state: raw.state || (raw.available ? "ok" : "unsupported"),
    detail: raw.detail || "",
    binary_path: raw.binary_path || "",
    auth_method: raw.auth_method || "",
    can_install: !!raw.can_install,
  };
}

export function createBackendPicker({ onChange, onAction } = {}) {
  const root = document.createElement("div");
  root.style.cssText = `
    display: flex; gap: 5px; margin: 4px 0 8px; flex-wrap: wrap;
  `;
  let active = _readSaved();
  /** @type {Record<string, {state: string, detail: string, binary_path: string, auth_method: string, can_install: boolean}>} */
  let statuses = { litellm: { state: "ok", detail: "", binary_path: "", auth_method: "", can_install: false } };

  function _isUsable(id) {
    const st = statuses[id];
    if (!st) return true;            // unknown = optimistically allow
    return st.state === "ok";
  }

  function _paint() {
    for (const btn of root.querySelectorAll("button[data-backend]")) {
      const id = btn.dataset.backend;
      const st = statuses[id];
      const state = st?.state;
      const isActive = id === active;
      const connected = state === "ok";
      btn.classList.toggle("cc-chip-active", isActive);
      // Binary connection model: connected = full opacity & enabled,
      // not-connected = dimmed & disabled.  Sign-in/install live in
      // Settings → Agents, not on the chip itself.
      btn.style.opacity = connected ? "1" : "0.45";
      btn.disabled = !connected;
      btn.style.cursor = connected ? "pointer" : "not-allowed";

      const meta = BACKENDS.find((b) => b.id === id);
      let tip = meta?.desc || "";
      if (!connected) {
        const why =
          state === "needs_install" ? "Not installed"  :
          state === "needs_auth"    ? "Not signed in"  :
          state === "unsupported"   ? "Unavailable"    :
          state === "error"         ? "Probe error"    : "Unavailable";
        tip = `${meta?.label || id}: ${why} — manage in Settings → Agents`;
        if (st?.detail) tip += ` (${st.detail})`;
      } else if (st?.detail) {
        tip = `${meta?.label || id}: ${st.detail}`;
      }
      btn.title = tip;

      const dot = btn.querySelector(".cc-be-dot");
      if (dot) {
        dot.style.background = connected
          ? "var(--cc-accent-green)"
          : "var(--cc-fg-faint)";
      }
    }
  }

  function set(id, fire = true) {
    if (!BACKENDS.find((b) => b.id === id)) return;
    if (active === id) return;
    if (!_isUsable(id)) {
      // Send the user to Settings → Agents instead of just toasting at
      // them — that page has the actual Install / Sign-in buttons.
      const st = statuses[id];
      const meta = BACKENDS.find((b) => b.id === id);
      const label = meta?.label || id;
      const why =
        st?.state === "needs_install" ? "not installed"  :
        st?.state === "needs_auth"    ? "not signed in"  :
        st?.state === "unsupported"   ? "unavailable"    :
        st?.state === "error"         ? "probe error"    : "unavailable";
      showToast(
        `${label} is ${why}. Open Settings → Agents to fix.`,
        st?.state === "error" ? "error" : "warning",
        5000,
      );
      return;
    }
    active = id;
    localStorage.setItem("comfyclaw_agent_backend", id);
    _paint();
    if (fire && typeof onChange === "function") onChange(id);
  }

  for (const be of BACKENDS) {
    const btn = document.createElement("button");
    btn.className = "cc-chip";
    btn.dataset.backend = be.id;
    btn.title = be.desc;
    btn.style.cssText = "display:inline-flex;align-items:center;gap:6px;";
    btn.innerHTML = `
      ${_logoHtml(be, 14)}
      <span class="cc-be-dot" style="width:7px;height:7px;border-radius:50%;
        background:var(--cc-fg-faint);display:inline-block;
        box-shadow:0 0 0 1px rgba(0,0,0,0.15) inset;"></span>
      <span>${escHtml(be.label)}</span>
    `;
    btn.addEventListener("click", () => set(be.id));
    root.appendChild(btn);
  }
  _paint();

  return {
    root,
    value: () => active,
    set,
    /** Return the raw status object for a backend (state/detail/...). */
    status: (id) => statuses[id] || null,
    /** Return the active backend's status object. */
    activeStatus: () => statuses[active] || null,
    /** Update the UI with the latest availability map from the server.
     *
     *  Accepts either the new object-valued map or the legacy boolean map.
     */
    setAvailability(map) {
      const previousActive = active;
      for (const [name, raw] of Object.entries(map || {})) {
        const norm = _normalizeStatus(raw);
        if (norm) statuses[name] = norm;
      }
      // If the currently-selected backend is no longer ok, demote to litellm.
      // This was historically silent — which is the *exact* source of the
      // "why is it asking me for an API key?" confusion.  Surface a toast
      // with the actual reason + a hint on how to recover.
      if (!_isUsable(active)) {
        const demoted = active;
        const st = statuses[demoted];
        active = "litellm";
        localStorage.setItem("comfyclaw_agent_backend", active);

        if (demoted !== previousActive) {
          // First paint after probe — no need to nag the user.
        } else if (demoted !== "litellm") {
          const meta = BACKENDS.find((b) => b.id === demoted);
          const label = meta?.label || demoted;
          const why = st?.detail || "unavailable";
          showToast(
            `${label} is unavailable (${why}). Falling back to LiteLLM. ` +
              `Open Settings → Agents to install or sign in.`,
            st?.state === "error" ? "error" : "warning",
            7000,
          );
        }
      }
      _paint();
    },
    /** Programmatically request an action ("install" | "signin") for an id.
     *  Wired to the popover's action buttons. */
    triggerAction(id, action) {
      if (typeof onAction === "function") onAction(id, action);
    },
  };
}
