import pytest

from amplitude_lab.cli import main
from amplitude_lab.llm import GeminiLLM, LLMError

from .conftest import CANDIDATES


def test_gemini_needs_key_and_model():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM(None, "m")
    with pytest.raises(LLMError, match="GEMINI_MODEL"):
        GeminiLLM("k", None)


def test_key_not_in_repr_or_description():
    llm = GeminiLLM("secret-key", "m", temperature=0.3)
    assert "secret-key" not in repr(llm) and "secret-key" not in str(llm.describe())
    assert llm.describe()["temperature"] == 0.3 and llm.describe()["model"] == "m"


def test_gemini_timeout_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "300")
    assert GeminiLLM.from_env(use_dotenv=False).timeout == 300


@pytest.mark.parametrize("bad", ["abc", "0", "-3"])
def test_gemini_bad_timeout_from_env_rejected(monkeypatch, bad):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", bad)
    with pytest.raises(LLMError, match="GEMINI_TIMEOUT_SECONDS"):
        GeminiLLM.from_env(use_dotenv=False)


def test_sdk_error_is_scrubbed(monkeypatch):
    genai = pytest.importorskip("google.genai")

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("bad request, key=secret-key")

    monkeypatch.setattr(genai, "Client", Boom)
    with pytest.raises(LLMError) as exc:
        GeminiLLM("secret-key", "m").complete("sys", [{"role": "user", "text": "hi"}])
    assert "secret-key" not in str(exc.value) and exc.value.__cause__ is None


def test_request_shape(monkeypatch):
    genai = pytest.importorskip("google.genai")
    seen = {}

    class Models:
        def generate_content(self, **kw):
            seen.update(kw)
            return type("R", (), {"text": "ok"})()

    class Client:
        def __init__(self, *a, **k):
            seen["client"] = k
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    llm = GeminiLLM("k", "some-model", timeout=5)
    msgs = [{"role": "user", "text": "a"}, {"role": "assistant", "text": "b"}]
    assert llm.complete("sys", msgs) == "ok"
    assert seen["model"] == "some-model"
    assert [c.role for c in seen["contents"]] == ["user", "model"]
    assert seen["config"].system_instruction == "sys"
    assert seen["client"]["http_options"].timeout == 5000


def test_empty_response_is_an_error(monkeypatch):
    genai = pytest.importorskip("google.genai")

    class Models:
        def generate_content(self, **kw):
            return type("R", (), {"text": "  "})()

    class Client:
        def __init__(self, *a, **k):
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    with pytest.raises(LLMError, match="empty"):
        GeminiLLM("k", "m").complete("s", [{"role": "user", "text": "x"}])


def test_empty_response_reports_gemini_reason(monkeypatch):
    genai = pytest.importorskip("google.genai")

    class Models:
        def generate_content(self, **kw):
            return type(
                "R",
                (),
                {
                    "text": None,
                    "prompt_feedback": type("F", (), {"block_reason": "SAFETY"})(),
                    "candidates": [
                        type("C", (), {"finish_reason": "MAX_TOKENS"})()
                    ],
                },
            )()

    class Client:
        def __init__(self, *a, **k):
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    with pytest.raises(LLMError, match="prompt block reason: SAFETY") as exc:
        GeminiLLM("k", "m").complete("s", [{"role": "user", "text": "x"}])
    assert "candidate finish reason: MAX_TOKENS" in str(exc.value)


def test_cli_selftest(capsys):
    assert main(["selftest", "--seeds", "5"]) == 0
    assert "reference passed 5 of 5 seeds" in capsys.readouterr().out


def test_cli_verify_pass_and_fail(capsys):
    assert main(["verify", str(CANDIDATES / "correct.py"), "--trusted", "--seed", "4"]) == 0
    capsys.readouterr()
    assert (
        main(["verify", str(CANDIDATES / "no_mu.py"), "--trusted", "--details", "--seed", "4"]) == 1
    )
    out = capsys.readouterr().out
    assert "FAIL scale_covariance" in out and "worst_ratio" in out and "seed 4" in out


def test_cli_task_prints_statement(capsys):
    assert main(["task"]) == 0
    assert "massless phi^4" in capsys.readouterr().out


def test_cli_unknown_task(capsys):
    assert main(["task", "--task", "no_such_task"]) == 2


def test_cli_agent_without_key(capsys):
    assert main(["agent", "--model", "x"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
