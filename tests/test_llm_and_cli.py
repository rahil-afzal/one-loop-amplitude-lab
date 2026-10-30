import sys
import types

import pytest

from amplitude_lab.llm import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MAX_TOKENS,
    GeminiLLM,
    LLMError,
    OpenAICompatibleLLM,
    llm_from_env,
)


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


def test_gemini_default_timeout_is_unchanged(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    assert GeminiLLM.from_env(use_dotenv=False).timeout == 120


@pytest.mark.parametrize("bad", ["abc", "0", "-3", "nan", "inf", "-inf"])
def test_gemini_bad_timeout_from_env_rejected(monkeypatch, bad):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", bad)
    with pytest.raises(LLMError, match="LLM_TIMEOUT_SECONDS"):
        GeminiLLM.from_env(use_dotenv=False)


def test_compatible_provider_config_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "gateway-secret")
    monkeypatch.setenv("LLM_MODEL", "qwen/qwen-3.7-plus")
    monkeypatch.setenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "180")
    client = llm_from_env(use_dotenv=False)
    assert isinstance(client, OpenAICompatibleLLM)
    assert client.model == "qwen/qwen-3.7-plus"
    assert client.base_url == "https://tokens.bd/v1"
    assert client.timeout == 180
    assert client.stream is False
    assert client.max_tokens == DEFAULT_OPENAI_MAX_TOKENS
    assert "gateway-secret" not in repr(client)
    assert "gateway-secret" not in str(client.describe())


def test_explicit_timeout_overrides_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "90")

    client = OpenAICompatibleLLM.from_env(timeout=45, use_dotenv=False)

    assert client.timeout == 45


def test_invalid_explicit_timeout_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "model")

    with pytest.raises(LLMError, match="finite positive number"):
        OpenAICompatibleLLM.from_env(timeout=0, use_dotenv=False)


def test_openai_max_tokens_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "provider/model")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "6000")

    client = OpenAICompatibleLLM.from_env(use_dotenv=False)

    assert client.max_tokens == 6000
    assert client.describe()["max_tokens"] == 6000


@pytest.mark.parametrize("bad", ["abc", "0", "-2", "1.5"])
def test_invalid_openai_max_tokens_are_rejected(monkeypatch, bad):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "provider/model")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", bad)

    with pytest.raises(LLMError, match="OPENAI_MAX_TOKENS must be a positive integer"):
        OpenAICompatibleLLM.from_env(use_dotenv=False)


def test_compatible_provider_requires_api_key_and_model(monkeypatch):
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAICompatibleLLM.from_env(use_dotenv=False)
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    with pytest.raises(LLMError, match="LLM_MODEL"):
        OpenAICompatibleLLM.from_env(use_dotenv=False)


def test_compatible_provider_request_shape(monkeypatch):
    seen = {}

    class Completions:
        def create(self, **kwargs):
            seen["request"] = kwargs
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="  derived result  "),
                        finish_reason="stop",
                    )
                ]
            )

    class Client:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client))
    client = OpenAICompatibleLLM(
        "secret-token", "deepseek/deepseek-4.1-flash", temperature=0.1, timeout=180
    )
    result = client.complete(
        "derive carefully",
        [{"role": "user", "text": "derive"}, {"role": "assistant", "text": "audit"}],
    )
    assert result == "derived result"
    assert seen["client"] == {
        "api_key": "secret-token",
        "base_url": "https://tokens.bd/v1",
        "timeout": 180,
        "max_retries": 0,
    }
    assert seen["request"] == {
        "model": "deepseek/deepseek-4.1-flash",
        "messages": [
            {"role": "system", "content": "derive carefully"},
            {"role": "user", "content": "derive"},
            {"role": "assistant", "content": "audit"},
        ],
        "temperature": 0.1,
        "max_tokens": DEFAULT_OPENAI_MAX_TOKENS,
        "stream": False,
    }


def test_compatible_provider_streaming_joins_content_and_closes_stream(monkeypatch):
    seen = {}

    class Completions:
        def create(self, **kwargs):
            seen["request"] = kwargs
            return iter(
                [
                    types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(content="derived "),
                                finish_reason=None,
                            )
                        ]
                    ),
                    types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(content="result"),
                                finish_reason=None,
                            )
                        ]
                    ),
                    types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                delta=types.SimpleNamespace(content=None),
                                finish_reason="stop",
                            )
                        ]
                    ),
                ]
            )

    class Client:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client))
    client = OpenAICompatibleLLM("secret-token", "provider/model", stream=True)

    result = client.complete("system", [{"role": "user", "text": "prompt"}])

    assert result == "derived result"
    assert seen["request"]["stream"] is True
    assert client.describe()["stream"] is True


def test_compatible_provider_streaming_uses_finish_reason_for_empty_response(monkeypatch):
    chunks = iter(
        [
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(content=None),
                        finish_reason="length",
                    )
                ]
            )
        ]
    )

    class Completions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return chunks

    class Client:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client))
    client = OpenAICompatibleLLM("key", "provider/model", stream=True)
    with pytest.raises(LLMError, match="finish reason: length"):
        client.complete("system", [{"role": "user", "text": "prompt"}])


def test_compatible_provider_stream_setting_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "provider/model")
    monkeypatch.setenv("OPENAI_STREAM", "true")

    assert OpenAICompatibleLLM.from_env(use_dotenv=False).stream is True

    monkeypatch.setenv("OPENAI_STREAM", "sometimes")
    with pytest.raises(LLMError, match="OPENAI_STREAM must be a boolean"):
        OpenAICompatibleLLM.from_env(use_dotenv=False)


def test_compatible_provider_redacts_key_from_sdk_error(monkeypatch):
    class Completions:
        def create(self, **kwargs):
            raise RuntimeError("request failed with token secret-token")

    class Client:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client))
    client = OpenAICompatibleLLM("secret-token", "qwen-3.7-plus")
    with pytest.raises(LLMError) as exc:
        client.complete("system", [{"role": "user", "text": "prompt"}])
    assert "RuntimeError" in str(exc.value)
    assert "secret-token" not in str(exc.value)


def test_compatible_provider_reports_finish_reason(monkeypatch):
    class Completions:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=None),
                        finish_reason="length",
                    )
                ]
            )

    class Client:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client))
    with pytest.raises(LLMError, match="finish reason: length"):
        OpenAICompatibleLLM("key", "qwen-3.7-plus").complete(
            "system", [{"role": "user", "text": "prompt"}]
        )


def test_compatible_provider_rejects_truncated_nonempty_response(monkeypatch):
    class Completions:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="incomplete partial answer"),
                        finish_reason="length",
                    )
                ]
            )

    class Client:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client))
    with pytest.raises(LLMError, match="response may be incomplete"):
        OpenAICompatibleLLM("key", "provider/model").complete(
            "system", [{"role": "user", "text": "prompt"}]
        )


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
                    "candidates": [type("C", (), {"finish_reason": "MAX_TOKENS"})()],
                },
            )()

    class Client:
        def __init__(self, *a, **k):
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    with pytest.raises(LLMError, match="prompt block reason: SAFETY") as exc:
        GeminiLLM("k", "m").complete("s", [{"role": "user", "text": "x"}])
    assert "candidate finish reason: MAX_TOKENS" in str(exc.value)
