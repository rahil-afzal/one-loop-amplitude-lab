import math
import os


class LLMError(RuntimeError):
    pass


DEFAULT_OPENAI_BASE_URL = "https://tokens.bd/v1"
DEFAULT_LLM_TIMEOUT = 180.0
DEFAULT_OPENAI_MAX_TOKENS = 2048


class ScriptedLLM:
    """Replays fixed replies. For tests and dry runs."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def describe(self):
        return {"name": self.name}

    def complete(self, system, messages):
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise LLMError("scripted replies exhausted")
        return self.replies.pop(0)


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
        )

    def __repr__(self):
        return (
            f"OpenAICompatibleLLM(model={self.model!r}, base_url={self.base_url!r}, "
            "api_key=<redacted>)"
        )

    def describe(self):
        return {
            "name": self.name,
            "provider": "openai-compatible",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout_s": self.timeout,
            "stream": self.stream,
        }

    def complete(self, system, messages):
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
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}]
                + [{"role": message["role"], "content": message["text"]} for message in messages],
                temperature=self.temperature,
                max_tokens=DEFAULT_OPENAI_MAX_TOKENS,
                stream=self.stream,
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
                        elif isinstance(content, list):
                            content_parts.extend(
                                getattr(part, "text", "")
                                for part in content
                                if getattr(part, "type", None) == "text"
                            )
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
            raise LLMError(_redact(f"{type(exc).__name__}: {exc}", self._key)) from None

        if not isinstance(content, str) or not content.strip():
            detail = f" (finish reason: {finish_reason})" if finish_reason else ""
            raise LLMError(
                "Chat API returned an empty assistant message"
                + detail
                + "; check model ID and endpoint compatibility."
            )
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

    def __init__(self, api_key, model, temperature=0.7, timeout=120.0):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set (environment or .env)")
        if not model:
            raise LLMError("GEMINI_MODEL is not set; pick a model from the Gemini docs")
        self._key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
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
        )

    def __repr__(self):
        return f"GeminiLLM(model={self.model!r})"

    def describe(self):
        return {"name": self.name, "model": self.model, "temperature": self.temperature,
                "timeout_s": self.timeout}  # fmt: skip

    def complete(self, system, messages):
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
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=self.temperature
                ),
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
        return text
