import math
import os


class LLMError(RuntimeError):
    pass


DEFAULT_OPENAI_BASE_URL = "https://tokens.bd/v1"
DEFAULT_LLM_TIMEOUT = 180.0
DEFAULT_OPENAI_MAX_TOKENS = 128000
DEFAULT_OPENAI_MAX_RETRIES = 2
DEFAULT_DERIVATION_MODEL = "qwen/qwen-3.7-plus"
DEFAULT_AUDIT_MODEL = "claude-haiku-5-5"


class ScriptedLLM:
    """Replays fixed replies. For tests and dry runs."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.systems = []

    def describe(self):
        return {"name": self.name}

    def complete(self, system, messages):
        self.systems.append(system)
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise LLMError("scripted replies exhausted")
        return self.replies.pop(0)

    def complete_stream(self, system, messages, on_token):
        response = self.complete(system, messages)
        for start in range(0, len(response), 12):
            on_token(response[start : start + 12])
        return response


def _redact(text, secret):
    if secret:
        text = text.replace(secret, "<REDACTED>")
    return text[:500]


def _load_dotenv():
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True), override=False)
    except ImportError:
        pass


def _timeout_from_env(default=DEFAULT_LLM_TIMEOUT, timeout=None):
    raw_timeout = timeout
    if raw_timeout is None:
        raw_timeout = os.environ.get(
            "LLM_TIMEOUT_SECONDS",
            os.environ.get("GEMINI_TIMEOUT_SECONDS", str(default)),
        )
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise LLMError("LLM_TIMEOUT_SECONDS must be a number.") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise LLMError("LLM_TIMEOUT_SECONDS must be a finite positive number.")
    return timeout


def _bool_from_env(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise LLMError(f"{name} must be a boolean (true/false).")


def _max_tokens_from_env(default=DEFAULT_OPENAI_MAX_TOKENS):
    raw_tokens = os.environ.get("OPENAI_MAX_TOKENS", str(default))
    try:
        max_tokens = int(raw_tokens)
    except ValueError as exc:
        raise LLMError("OPENAI_MAX_TOKENS must be a positive integer.") from exc
    if max_tokens <= 0:
        raise LLMError("OPENAI_MAX_TOKENS must be a positive integer.")
    return max_tokens


class OpenAICompatibleLLM:
    """Client for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        api_key,
        model,
        base_url=DEFAULT_OPENAI_BASE_URL,
        temperature=0.2,
        timeout=DEFAULT_LLM_TIMEOUT,
        stream=False,
        max_tokens=DEFAULT_OPENAI_MAX_TOKENS,
        reasoning_effort=None,
    ):
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        if not model:
            raise LLMError("LLM_MODEL is not set; choose a model ID from the API provider.")
        if not base_url or not base_url.startswith(("https://", "http://")):
            raise LLMError("OPENAI_BASE_URL must be an http:// or https:// URL.")
        self._key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.stream = stream
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.name = f"openai-compatible:{model}"

    @classmethod
    def from_env(cls, model=None, temperature=0.2, timeout=None, use_dotenv=True):
        if use_dotenv:
            _load_dotenv()
        return cls(
            os.environ.get("OPENAI_API_KEY"),
            model or os.environ.get("LLM_MODEL"),
            base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            temperature=temperature,
            timeout=_timeout_from_env(timeout=timeout),
            stream=_bool_from_env("OPENAI_STREAM"),
            max_tokens=_max_tokens_from_env(),
        )

    @classmethod
    def from_role_env(cls, role, default_model, temperature=0.2, timeout=None, use_dotenv=True):
        if role not in ("derivation", "audit"):
            raise ValueError(f"Unknown LLM role {role!r}.")
        if use_dotenv:
            _load_dotenv()
        prefix = role.upper()
        api_key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = os.environ.get(f"{prefix}_MODEL") or default_model
        base_url = os.environ.get(f"{prefix}_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL
        )
        streaming = _bool_from_env("STREAMING_ENABLED", _bool_from_env("OPENAI_STREAM", True))
        return cls(
            api_key,
            model,
            base_url=base_url,
            temperature=temperature,
            timeout=_timeout_from_env(timeout=timeout),
            stream=streaming,
            max_tokens=_max_tokens_from_env(),
            reasoning_effort=os.environ.get(f"{prefix}_REASONING_EFFORT") or None,
        )

    def __repr__(self):
        return (
            f"OpenAICompatibleLLM(model={self.model!r}, base_url={self.base_url!r}, "
            "api_key=<redacted>)"
        )

    def describe(self):
        description = {
            "name": self.name,
            "provider": "openai-compatible",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout_s": self.timeout,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            description["reasoning_effort"] = self.reasoning_effort
        return description

    def complete(self, system, messages):
        return self.complete_stream(system, messages, lambda token: token)

    def complete_stream(self, system, messages, on_token):
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError(
                "The OpenAI SDK is not installed. Run: python -m pip install "
                "-e '.[openai-compatible]'."
            ) from None

        finish_reason = None
        try:
            client = OpenAI(
                api_key=self._key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=DEFAULT_OPENAI_MAX_RETRIES,
            )
            request = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}]
                + [{"role": message["role"], "content": message["text"]} for message in messages],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": self.stream,
            }
            if self.reasoning_effort:
                request["reasoning_effort"] = self.reasoning_effort
            response = client.chat.completions.create(
                **request,
            )
            if self.stream:
                content_parts = []
                finish_reason = None
                try:
                    for chunk in response:
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        choice = choices[0]
                        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                        delta = getattr(choice, "delta", None)
                        content = getattr(delta, "content", None)
                        if isinstance(content, str):
                            content_parts.append(content)
                            on_token(content)
                        elif isinstance(content, list):
                            text_parts = [
                                getattr(part, "text", "")
                                for part in content
                                if getattr(part, "type", None) == "text"
                            ]
                            content_parts.extend(text_parts)
                            for text in text_parts:
                                on_token(text)
                finally:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                content = "".join(content_parts)
            else:
                choices = getattr(response, "choices", None) or []
                choice = choices[0] if choices else None
                finish_reason = getattr(choice, "finish_reason", None)
                message = getattr(choice, "message", None)
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    content = "".join(
                        getattr(part, "text", "")
                        for part in content
                        if getattr(part, "type", None) == "text"
                    )
        except Exception as exc:
            message = _redact(f"{type(exc).__name__}: {exc}", self._key)
            if (
                type(exc).__name__ in {
                    "APIError",
                    "APIConnectionError",
                    "APITimeoutError",
                    "InternalServerError",
                    "RateLimitError",
                }
                or "upstream ai provider encountered an error" in message.lower()
            ):
                message += (
                    " The client retries transient provider errors up to "
                    f"{DEFAULT_OPENAI_MAX_RETRIES} times. If this persists, check the "
                    "provider's status, model availability, and configured endpoint, then retry."
                )
            raise LLMError(message) from None

        if finish_reason == "length":
            raise LLMError(
                "Chat API reached its completion token limit "
                f"(finish reason: length, OPENAI_MAX_TOKENS={self.max_tokens}); "
                "the response may be incomplete. Increase OPENAI_MAX_TOKENS if the "
                "model/provider allows it."
            )
        if not isinstance(content, str) or not content.strip():
            detail = f" (finish reason: {finish_reason})" if finish_reason else ""
            raise LLMError(
                "Chat API returned an empty assistant message"
                + detail
                + "; check model ID and endpoint compatibility."
            )
        if not self.stream:
            on_token(content)
        return content.strip()


def llm_from_env(provider=None, model=None, temperature=0.2, timeout=None, use_dotenv=True):
    """Create the configured API client; OpenAI-compatible is the default provider."""
    if use_dotenv:
        _load_dotenv()
    selected = (provider or os.environ.get("LLM_PROVIDER", "openai-compatible")).lower()
    if selected in ("openai-compatible", "openai_compatible"):
        return OpenAICompatibleLLM.from_env(
            model=model, temperature=temperature, timeout=timeout, use_dotenv=False
        )
    if selected == "gemini":
        return GeminiLLM.from_env(
            model=model, temperature=temperature, timeout=timeout, use_dotenv=False
        )
    raise LLMError(f"Unsupported LLM_PROVIDER {selected!r}; use 'openai-compatible' or 'gemini'.")


def role_llm_from_env(role, default_model, temperature=0.2, timeout=None, use_dotenv=True):
    """Create an independently configured OpenAI-compatible derivation or audit client."""
    return OpenAICompatibleLLM.from_role_env(
        role=role,
        default_model=default_model,
        temperature=temperature,
        timeout=timeout,
        use_dotenv=use_dotenv,
    )


def _response_reason(response):
    reasons = []
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)
    if block_reason is not None:
        label = getattr(block_reason, "name", None) or str(
            getattr(block_reason, "value", block_reason)
        )
        if "unspecified" not in label.lower() and label not in ("0", ""):
            reasons.append(f"prompt block reason: {label}")

    for candidate in getattr(response, "candidates", None) or []:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is None:
            continue
        label = getattr(finish_reason, "name", None) or str(
            getattr(finish_reason, "value", finish_reason)
        )
        if "unspecified" not in label.lower() and label not in ("0", ""):
            reasons.append(f"candidate finish reason: {label}")
    return "; ".join(reasons)


class GeminiLLM:
    """Thin wrapper around the google-genai client (pip install -e '.[gemini]').

    The call pattern follows the google-genai docs, but it has only been exercised against a
    stub here, not the live API. Check the model name against the current model list.
    """

    def __init__(self, api_key, model, temperature=0.7, timeout=120.0, stream=False):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set (environment or .env)")
        if not model:
            raise LLMError("GEMINI_MODEL is not set; pick a model from the Gemini docs")
        self._key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.stream = stream
        self.name = f"gemini:{model}"

    @classmethod
    def from_env(cls, model=None, temperature=0.7, timeout=None, use_dotenv=True):
        if use_dotenv:
            _load_dotenv()
        timeout = _timeout_from_env(default=120.0, timeout=timeout)
        return cls(
            os.environ.get("GEMINI_API_KEY"),
            model or os.environ.get("GEMINI_MODEL"),
            temperature=temperature,
            timeout=timeout,
            stream=_bool_from_env("GEMINI_STREAM"),
        )

    def __repr__(self):
        return f"GeminiLLM(model={self.model!r})"

    def describe(self):
        return {
            "name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "timeout_s": self.timeout,
            "stream": self.stream,
        }

    def complete(self, system, messages):
        return self.complete_stream(system, messages, lambda token: token)

    def complete_stream(self, system, messages, on_token):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise LLMError("google-genai is not installed: pip install -e '.[gemini]'") from None
        try:
            client = genai.Client(
                api_key=self._key,
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
            )
            contents = [
                types.Content(
                    role="user" if m["role"] == "user" else "model",
                    parts=[types.Part(text=m["text"])],
                )
                for m in messages
            ]
            config = types.GenerateContentConfig(
                system_instruction=system, temperature=self.temperature
            )
            if self.stream:
                parts = []
                response = client.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                )
                for chunk in response:
                    text_part = getattr(chunk, "text", None)
                    if text_part:
                        parts.append(text_part)
                        on_token(text_part)
                text = "".join(parts)
            else:
                response = client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                text = response.text
        except Exception as exc:
            # the message can echo request details, so scrub the key and drop the chain
            raise LLMError(_redact(f"{type(exc).__name__}: {exc}", self._key)) from None
        if not text or not text.strip():
            reason = _response_reason(response)
            detail = f" ({reason})" if reason else ""
            raise LLMError(
                "Gemini returned no text (empty response)"
                + detail
                + "; check model access, safety filtering and generation limits."
            )
        if not self.stream:
            on_token(text)
        return text
