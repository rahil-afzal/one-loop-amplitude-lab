import math
import os


class LLMError(RuntimeError):
    pass


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
    def from_env(cls, model=None, temperature=0.7, use_dotenv=True):
        if use_dotenv:
            try:
                from dotenv import find_dotenv, load_dotenv

                load_dotenv(find_dotenv(usecwd=True))
            except ImportError:
                pass
        try:
            timeout = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "120"))
        except ValueError as exc:
            raise LLMError("GEMINI_TIMEOUT_SECONDS must be a number.") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise LLMError("GEMINI_TIMEOUT_SECONDS must be a finite positive number.")
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
