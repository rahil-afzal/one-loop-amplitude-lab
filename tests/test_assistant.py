import pytest

from amplitude_lab.assistant import BubbleClosedFormTask, run_task
from amplitude_lab.assistant.client import GeminiClient, redact
from amplitude_lab.assistant.config import GeminiConfig
from amplitude_lab.assistant.errors import (
    AssistantConfigError,
    AssistantRequestError,
    AssistantUnavailableError,
)
from amplitude_lab.assistant.mock import MOCK_PROPOSALS, ScriptedClient, mock_client

TASK = BubbleClosedFormTask()


def test_correct_proposal_verified(problem):
    outcome = run_task(mock_client("correct"), TASK, problem)
    assert outcome.status == "verified"
    assert outcome.verification["passed"]


@pytest.mark.parametrize("kind", ["wrong_sign", "wrong_branch", "explicit_ipi"])
def test_wrong_candidates_rejected_by_deterministic_verifier(problem, kind):
    outcome = run_task(mock_client(kind), TASK, problem)
    assert outcome.status == "rejected"
    assert not outcome.verification["passed"]


@pytest.mark.parametrize("kind", ["malicious", "garbage"])
def test_unsafe_candidates_rejected_before_any_evaluation(problem, kind):
    outcome = run_task(mock_client(kind), TASK, problem)
    assert outcome.status == "rejected_unparseable"
    assert outcome.verification is None


def test_malicious_payload_has_no_side_effect(problem, tmp_path, monkeypatch):
    marker = tmp_path / "pwned"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    outcome = run_task(ScriptedClient(payload), TASK, problem)
    assert outcome.status == "rejected_unparseable"
    assert not marker.exists()


def test_correct_form_is_not_taken_on_trust(problem):
    """Equivalent-looking forms with the wrong sheet must fail on the boundary points."""
    outcome = run_task(ScriptedClient("2 - log(s/mu2)"), TASK, problem)
    assert outcome.status == "rejected"


def test_every_mock_kind_has_a_defined_outcome(problem):
    expected = {
        "correct": "verified",
        "wrong_sign": "rejected",
        "wrong_branch": "rejected",
        "explicit_ipi": "rejected",
        "malicious": "rejected_unparseable",
        "garbage": "rejected_unparseable",
    }
    assert set(expected) == set(MOCK_PROPOSALS)
    for kind, status in expected.items():
        assert run_task(mock_client(kind), TASK, problem).status == status


def test_client_failure_becomes_unavailable_outcome(problem):
    class Failing:
        name = "failing"

        def generate(self, prompt, system_instruction=None):
            raise AssistantRequestError("network down")

    outcome = run_task(Failing(), TASK, problem)
    assert outcome.status == "unavailable" and "network down" in outcome.detail


def test_outcome_is_json_ready(problem):
    import json

    json.dumps(run_task(mock_client("correct"), TASK, problem).to_dict())


def test_prompt_is_sent_to_client(problem):
    client = mock_client("correct")
    run_task(client, TASK, problem)
    assert len(client.calls) == 1
    assert "closed form" in client.calls[0][0]
    assert "principal" in client.calls[0][1]


# ---- configuration and credential handling ---------------------------------------------


def test_missing_key_error_is_clear():
    config = GeminiConfig.from_env()
    with pytest.raises(AssistantConfigError, match="GEMINI_API_KEY"):
        config.require_key()


def test_client_does_not_touch_network_without_key():
    with pytest.raises(AssistantConfigError):
        GeminiClient(GeminiConfig.from_env()).smoke_test()


def test_key_never_in_repr(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-do-not-log")
    config = GeminiConfig.from_env()
    assert "secret-do-not-log" not in repr(config)
    assert "secret-do-not-log" not in str(config)
    assert "<redacted>" in repr(config)


def test_model_and_timeout_configurable(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "some-other-model")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "12.5")
    config = GeminiConfig.from_env()
    assert config.model == "some-other-model" and config.timeout_seconds == 12.5


@pytest.mark.parametrize("bad", ["abc", "0", "-3", "nan", "inf", "-inf"])
def test_bad_timeout_rejected(monkeypatch, bad):
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", bad)
    with pytest.raises(AssistantConfigError):
        GeminiConfig.from_env()


def test_redact_helper():
    assert "KEY123" not in redact("failure with KEY123 inside", "KEY123")
    assert len(redact("x" * 5000, None)) < 700


def test_sdk_exception_is_sanitized(monkeypatch):
    """Even if the SDK echoes the key in an exception, it must not reach the caller."""
    genai = pytest.importorskip("google.genai")

    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("auth failed for key=sk-live-SECRET-123")

    monkeypatch.setattr(genai, "Client", Boom)
    config = GeminiConfig(api_key="sk-live-SECRET-123")
    with pytest.raises(AssistantRequestError) as excinfo:
        GeminiClient(config).generate("hi")
    assert "sk-live-SECRET-123" not in str(excinfo.value)
    assert "<REDACTED>" in str(excinfo.value)
    assert excinfo.value.__cause__ is None  # no chained exception that could carry the key


def test_empty_response_is_an_error(monkeypatch):
    genai = pytest.importorskip("google.genai")

    class FakeModels:
        def generate_content(self, **kwargs):
            return type("R", (), {"text": "   "})()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    with pytest.raises(AssistantRequestError, match="empty"):
        GeminiClient(GeminiConfig(api_key="k")).generate("hi")


def test_request_shape_uses_documented_sdk_interface(monkeypatch):
    genai = pytest.importorskip("google.genai")

    seen = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            seen.update(kwargs)
            return type("R", (), {"text": " OK "})()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    config = GeminiConfig(api_key="k", model="m", timeout_seconds=7)
    assert GeminiClient(config).generate("p", "sys") == "OK"
    assert seen["model"] == "m" and seen["contents"] == "p"
    assert seen["config"].system_instruction == "sys"
    assert seen["client_kwargs"]["http_options"].timeout == 7000


def test_missing_sdk_is_reported(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("google"):
            raise ImportError("no google")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AssistantUnavailableError, match="google-genai"):
        GeminiClient(GeminiConfig(api_key="k")).generate("hi")


def test_dotenv_loading_is_a_noop_without_the_package(monkeypatch):
    import builtins

    from amplitude_lab.assistant import config as config_module

    monkeypatch.undo()  # drop the autouse stub so the real loader runs
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("no dotenv")
        return real_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", fake_import)
        mp.setenv("GEMINI_API_KEY", "from-os-env")
        config_module.load_dotenv_file()  # must not raise
        assert GeminiConfig.from_env(load_dotenv=True).api_key == "from-os-env"
