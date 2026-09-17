"""Grok-specific intent gate and task-contract regression tests."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from comfyclaw.agent_backends.grok_backend import GrokCLIBackend, _build_grok_intent_gate

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
    (tmp_path / "auth.json").write_text(
        json.dumps({"https://accounts.x.ai/sign-in": {"key": "fake-oauth-token"}}),
        encoding="utf-8",
    )
    return tmp_path


def test_production_intent_gate_preserves_general_qa_path() -> None:
    gate = _build_grok_intent_gate([{"name": "answer_user"}, {"name": "set_param"}])
    assert "CONVERSATION" in gate
    assert "Use `answer_user` and stop" in gate
    assert "Do not fabricate Goal/Target/Preserve/Change fields for ordinary Q&A" in gate
    assert "does not clearly request a workflow mutation" in gate


def test_tool_free_chat_uses_rationale_instead_of_missing_answer_user_tool() -> None:
    gate = _build_grok_intent_gate([])
    assert "This invocation is tool-free chat" in gate
    assert "Return `tool_calls: []`" in gate
    assert "put the full user-facing answer in `rationale`" in gate
    assert "set `done: true`" in gate
    assert "Use `answer_user` and stop" not in gate


def test_intent_gate_defines_workflow_contract_without_inventing_preferences() -> None:
    gate = _build_grok_intent_gate([{"name": "answer_user"}, {"name": "set_param"}])
    for field in ("Goal", "Target", "Preserve", "Change", "Constraints", "Done"):
        assert field in gate
    assert "keep it `not specified`; never invent a preference" in gate
    assert "Prefer the smallest reversible interpretation" in gate
    assert "Never broaden Target or Change" in gate


def test_first_turn_injects_intent_gate_without_extra_cli_call(grok_home: Path) -> None:
    captured = []

    def fake_run(argv, _stdin, **_kwargs):
        prompt_path = Path(argv[argv.index("--prompt-file") + 1])
        captured.append((list(argv), prompt_path.read_text(encoding="utf-8")))
        return 0, json.dumps(
            {
                "text": json.dumps(
                    {"tool_calls": [], "rationale": "answered", "done": True}
                ),
                "sessionId": SID,
            }
        ), ""

    user = "## User Input\n왜 이 그래프가 느려?\n\n## Decision Required\nClassify first."
    with patch("comfyclaw.agent_backends._stream_session.run_cli_oneshot", side_effect=fake_run):
        result = GrokCLIBackend(session_key="intent-test").run_tool_loop(
            "system",
            user,
            [{"name": "answer_user", "description": "answer without mutation"}],
            lambda _call: ("ok", True),
        )

    assert result == "answered"
    assert len(captured) == 1
    argv, prompt = captured[0]
    assert "## Grok intent gate" in prompt
    assert "## User Input\n왜 이 그래프가 느려?" in prompt
    assert "Goal        = the requested end state" in prompt
    rules = argv[argv.index("--rules") + 1]
    assert "follow the conversation path described in the intent gate" in rules


def test_first_turn_keeps_literal_workflow_constraints(grok_home: Path) -> None:
    captured_prompt = ""

    def fake_run(argv, _stdin, **_kwargs):
        nonlocal captured_prompt
        prompt_path = Path(argv[argv.index("--prompt-file") + 1])
        captured_prompt = prompt_path.read_text(encoding="utf-8")
        return 0, json.dumps(
            {
                "text": json.dumps(
                    {"tool_calls": [], "rationale": "ready", "done": True}
                ),
                "sessionId": SID,
            }
        ), ""

    user = "## User Input\n재킷만 낡게 해. 얼굴은 건드리지 말고 Queue는 하지 마."
    with patch("comfyclaw.agent_backends._stream_session.run_cli_oneshot", side_effect=fake_run):
        GrokCLIBackend(session_key="constraint-test").run_tool_loop(
            "system", user, [], lambda _call: ("ok", False)
        )

    assert "재킷만 낡게 해. 얼굴은 건드리지 말고 Queue는 하지 마." in captured_prompt
    assert "Words such as `only`, `just`, `do not`, `keep`, `preserve`, `without`" in captured_prompt
