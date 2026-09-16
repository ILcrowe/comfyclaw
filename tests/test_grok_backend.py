"""Grok CLI backend: OAuth-only routing and ComfyClaw-owned tool dispatch."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from comfyclaw.agent_backends.base import get_backend, probe_all
from comfyclaw.agent_backends.grok_backend import (
    GrokCLIBackend,
    _get_recorded_grok_session,
    _grok_env,
    _guard_subscription_config,
    _oauth_cache_present,
)

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


@pytest.mark.asyncio
async def test_panel_chat_uses_grok_without_litellm_fallback() -> None:
    from comfyclaw.chat_agent import chat_stream

    with patch.object(GrokCLIBackend, "run_tool_loop", return_value="Grok reply") as run:
        result = [
            token
            async for token in chat_stream(
                [{"role": "user", "content": "Summarize this graph"}],
                {},
                "",
                agent_backend="grok-cli",
                session_id="panel-chat",
                skills_registry=False,
            )
        ]
    assert result == ["Grok reply"]
    assert run.call_count == 1
