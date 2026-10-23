"""Assistant client interface and the Gemini implementation.

SDK surface (``from google import genai``; ``genai.Client(api_key=...)``;
``client.models.generate_content(model=, contents=, config=types.GenerateContentConfig)``;
``types.HttpOptions(timeout=<milliseconds>)``) was inspected against google-genai 2.33.0.
Model IDs and live behaviour were NOT verified: check https://ai.google.dev/gemini-api/docs.

The SDK is imported lazily. There are no automatic retries. Error messages are sanitized.
"""

from __future__ import annotations

from typing import Protocol

from .config import GeminiConfig
from .errors import AssistantRequestError, AssistantUnavailableError


class AssistantClient(Protocol):
    name: str

    def generate(self, prompt: str, system_instruction: str | None = None) -> str: ...


def redact(text: str, secret: str | None) -> str:
    if secret:
        text = text.replace(secret, "<REDACTED>")
    return text if len(text) <= 600 else text[:600] + "..."


class GeminiClient:
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.name = f"gemini:{config.model}"

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        key = self.config.require_key()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise AssistantUnavailableError(
                "google-genai is not installed. Run: python -m pip install -e '.[gemini]'. "
                "The physics engine and offline tests do not need it."
            ) from exc

        try:
            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=int(self.config.timeout_seconds * 1000)),
            )
            kwargs: dict = {"temperature": self.config.temperature}
            if system_instruction:
                kwargs["system_instruction"] = system_instruction
            response = client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=types.GenerateContentConfig(**kwargs),
            )
            text = getattr(response, "text", None)
        except Exception as exc:
            raise AssistantRequestError(
                "Gemini request failed: "
                + redact(f"{type(exc).__name__}: {exc}", key)
                + ". No automatic retry was attempted; check model ID, quota and network."
            ) from None
        if not text or not str(text).strip():
            raise AssistantRequestError("Gemini returned an empty or blocked response.")
        return str(text).strip()

    def smoke_test(self) -> str:
        return self.generate(
            "Reply with exactly the word OK. Do not add anything else.",
            "You are a connectivity test. Reply with only OK.",
        )
