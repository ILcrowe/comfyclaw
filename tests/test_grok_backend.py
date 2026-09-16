"""Grok CLI backend: OAuth-only routing and ComfyClaw-owned tool dispatch."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from comfyclaw.agent import ClawAgent, _summarize_workflow_for_grok
from comfyclaw.agent_backends.base import get_backend, probe_all
from comfyclaw.agent_backends.grok_backend import (
    GrokCLIBackend,
    _get_recorded_grok_session,
    _grok_env,
    _guard_subscription_config,
    _oauth_cache_present,
)
from comfyclaw.chat_agent import _summarize_workflow
from comfyclaw.workflow import WorkflowManager

SID = "12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def grok_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    monkeypatch.setenv("COMFYCLAW_GROK_SUBSCRIPTION_ONLY", "true")
    monkeypatch.delenv("GROK_CONFIG", raising=False)
    monkeypatch.delenv("GROK_CONFIG_PATH", raising=False)
    monkeypatch.delenv("GROK_AUTH_PROVIDER_COMMAND", raising=False)
    monkeypatch.delenv("GROK_OIDC_ISSUER", raising=False)
    monkeypatch.delenv("GROK_OIDC_CLIENT_ID", raising=False)
    monkeypatch.delenv("GROK_MODELS_BASE_URL", raising=False)
    monkeypatch.delenv("GROK_MODELS_LIST_URL", raising=False)
    monkeypatch.delenv("GROK_CLI_CHAT_PROXY_BASE_URL", raising=False)
    return tmp_path


def _oauth_login(home: Path) -> None:
    (home / "auth.json").write_text(
        json.dumps({"https://accounts.x.ai/sign-in": {"key": "fake-oauth-token"}}),
        encoding="utf-8",
    )


def test_probe_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFYCLAW_GROK_BIN", "missing-grok-test-binary")
    status = next(s for s in probe_all() if s.name == "grok-cli")
    assert status.state == "needs_install"


def test_auth_probe_distinguishes_oauth_from_api_key(
    grok_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMFYCLAW_GROK_BIN", "grok")
    with patch("comfyclaw.agent_backends.base.shutil.which", return_value="/bin/grok"):
        assert next(s for s in probe_all() if s.name == "grok-cli").state == "needs_auth"
        (grok_home / "auth.json").write_text(
            json.dumps({"xai::api_key": {"key": "fake-api-key"}}), encoding="utf-8"
        )
        assert not _oauth_cache_present()
        _oauth_login(grok_home)
        status = next(s for s in probe_all() if s.name == "grok-cli")
        assert (status.state, status.auth_method) == ("ok", "oauth")


def test_subscription_guard_rejects_model_api_credentials(grok_home: Path) -> None:
    (grok_home / "config.toml").write_text(
        '[model."grok-build"]\napi_key = "fake-secret"\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="BLOCKED"):
        _guard_subscription_config()


def test_subscription_guard_rejects_endpoint_override(grok_home: Path) -> None:
    (grok_home / "config.toml").write_text(
        '[endpoints]\nmodels_base_url = "https://example.invalid/v1"\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="BLOCKED"):
        _guard_subscription_config()


def test_subscription_child_environment_scrubs_api_keys(
    grok_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test")
    monkeypatch.setenv("GROK_CODE_XAI_API_KEY", "test")
    monkeypatch.setenv("GROK_DEPLOYMENT_KEY", "test")
    child = _grok_env()
    assert "XAI_API_KEY" not in child
    assert "GROK_CODE_XAI_API_KEY" not in child
    assert "GROK_DEPLOYMENT_KEY" not in child
    assert child["GROK_DISABLE_API_KEY_AUTH"] == "1"


def test_factory_returns_grok_backend(grok_home: Path) -> None:
    _oauth_login(grok_home)
    with patch.object(GrokCLIBackend, "is_available", return_value=True):
        backend = get_backend("grok-cli", extra={"session_id": "panel-a"})
    assert isinstance(backend, GrokCLIBackend)
    assert backend.session_key == "panel-a"


def test_factory_never_falls_back_to_api_when_grok_missing(grok_home: Path) -> None:
    with patch.object(GrokCLIBackend, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="not installed"):
            get_backend("grok-cli", api_key="test")


def test_grok_generation_does_not_construct_api_verifier(grok_home: Path) -> None:
    from comfyclaw.harness import ClawHarness, HarnessConfig

    _oauth_login(grok_home)
    with (
        patch.object(GrokCLIBackend, "is_available", return_value=True),
        patch("comfyclaw.harness.ClawVerifier", side_effect=AssertionError("API verifier used")),
    ):
        harness = ClawHarness.from_workflow_dict(
            {},
            HarnessConfig(
                sync_port=0, agent_backend="grok-cli", run_mode="auto", verifier_mode="vlm"
            ),
        )
    assert harness._verifier is None


def test_envelope_dispatch_and_session_resume(grok_home: Path) -> None:
    _oauth_login(grok_home)
    prompts = []
    outputs = [
        {
            "text": json.dumps(
                {
                    "tool_calls": [{"name": "inspect_graph", "arguments": {}}],
                    "rationale": "checking",
                    "done": False,
                }
            ),
            "sessionId": SID,
        },
        {
            "text": json.dumps({"tool_calls": [], "rationale": "graph inspected", "done": True}),
            "sessionId": SID,
        },
        {
            "text": json.dumps({"tool_calls": [], "rationale": "continued", "done": True}),
            "sessionId": SID,
        },
    ]

    def fake_run(argv, _stdin, **kwargs):
        prompts.append((argv, kwargs["env"], kwargs["encoding"]))
        return 0, json.dumps(outputs.pop(0)), ""

    dispatched = []

    def dispatch(call):
        dispatched.append(call.name)
        return "one node", False

    with patch("comfyclaw.agent_backends._stream_session.run_cli_oneshot", side_effect=fake_run):
        be = GrokCLIBackend(session_key="panel-unique-test")
        result = be.run_tool_loop("system", "inspect", [{"name": "inspect_graph"}], dispatch)
        assert result == "graph inspected"
        assert dispatched == ["inspect_graph"]
        assert _get_recorded_grok_session("panel-unique-test") == SID
        assert _get_recorded_grok_session("another-panel") == ""
        assert be.run_tool_loop("system", "continue", [], dispatch) == "continued"

    assert "--resume" not in prompts[0][0]
    assert prompts[1][0][prompts[1][0].index("--resume") + 1] == SID
    assert prompts[2][0][prompts[2][0].index("--resume") + 1] == SID
    assert prompts[0][0][prompts[0][0].index("--tools") + 1] == "todo_write"
    assert "inspect_graph" in prompts[0][0][prompts[0][0].index("-p") + 1]
    assert "never infer a tool name" in prompts[0][0][prompts[0][0].index("-p") + 1]
    assert prompts[0][0][prompts[0][0].index("--disallowed-tools") + 1] == (
        "todo_write,search_tool,use_tool,Agent"
    )
    assert "MCPTool" in prompts[0][0]
    assert "--model" not in prompts[0][0]
    assert prompts[0][1]["GROK_DISABLE_API_KEY_AUTH"] == "1"
    assert prompts[0][2] == "utf-8"


def test_grok_workflow_summaries_omit_stored_prompt_text() -> None:
    sensitive = "stored prompt that must not be forwarded"
    workflow = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": sensitive, "clip": ["2", 0]},
        },
        "2": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "safe-model.safetensors"},
        },
    }

    generation_summary = _summarize_workflow_for_grok(workflow)
    chat_summary = _summarize_workflow(workflow, omit_stored_text=True)

    assert sensitive not in generation_summary
    assert sensitive not in chat_summary
    assert "stored text omitted" in generation_summary
    assert "stored text omitted" in chat_summary
    assert "safe-model.safetensors" in generation_summary
    assert "safe-model.safetensors" in chat_summary


def test_grok_generation_message_omits_current_positive_prompt() -> None:
    sensitive = "stored prompt that must not be forwarded"
    workflow = WorkflowManager(
        {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": sensitive}}}
    )
    agent = object.__new__(ClawAgent)
    agent.backend_name = "grok-cli"
    agent.skill_manager = SimpleNamespace(detect_relevant_skills=lambda _prompt: [], skill_names=[])

    message = agent._build_user_message("Add one node", workflow, None, None, 1)

    assert sensitive not in message
    assert "stored text omitted in Grok subscription mode" in message


def test_provider_safety_block_clears_resume_session_without_retry(grok_home: Path) -> None:
    _oauth_login(grok_home)
    session_key = "panel-safety-reset-test"
    calls = []
    outputs = [
        (
            0,
            json.dumps(
                {
                    "text": json.dumps(
                        {"tool_calls": [], "rationale": "ready", "done": True}
                    ),
                    "sessionId": SID,
                }
            ),
            "",
        ),
        (
            1,
            json.dumps(
                {
                    "type": "error",
                    "message": (
                        "API error (status 403 Forbidden): permission-denied: "
                        "Content violates usage guidelines. Failed check: "
                        "SAFETY_CHECK_TYPE_CSAM"
                    ),
                }
            ),
            "",
        ),
    ]

    def fake_run(argv, _stdin, **_kwargs):
        calls.append(argv)
        return outputs.pop(0)

    with patch("comfyclaw.agent_backends._stream_session.run_cli_oneshot", side_effect=fake_run):
        backend = GrokCLIBackend(session_key=session_key)
        assert backend.run_tool_loop("system", "safe request", [], lambda _call: "") == "ready"
        assert _get_recorded_grok_session(session_key) == SID
        with pytest.raises(RuntimeError, match="blocked this request under its safety policy"):
            backend.run_tool_loop("system", "blocked request", [], lambda _call: "")

    assert len(calls) == 2
    assert "--resume" in calls[1]
    assert _get_recorded_grok_session(session_key) == ""


@pytest.mark.asyncio
async def test_panel_chat_uses_grok_without_litellm_fallback() -> None:
    from comfyclaw.chat_agent import chat_stream

    with patch.object(GrokCLIBackend, "run_tool_loop", return_value="Grok reply") as run:
        result = [
            token
            async for token in chat_stream(
                [
                    {"role": "user", "content": "old panel transcript marker"},
                    {"role": "assistant", "content": "old assistant reply"},
                    {"role": "user", "content": "Summarize this graph"},
                ],
                {},
                "",
                agent_backend="grok-cli",
                session_id="panel-chat",
                skills_registry=False,
            )
    ]
    assert result == ["Grok reply"]
    assert run.call_count == 1
    assert run.call_args.args[1] == "Summarize this graph"
    assert "old panel transcript marker" not in run.call_args.args[0]
