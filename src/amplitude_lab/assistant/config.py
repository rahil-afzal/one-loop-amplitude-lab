"""Gemini configuration. Credentials come from the environment and are never printed.

Loading a local ``.env`` is opt-in (``load_dotenv=True``) and needs python-dotenv.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from .errors import AssistantConfigError

DEFAULT_MODEL = "gemini-3.8-flash"  # NOT live-verified: check the official model list.


def load_dotenv_file() -> None:
    """Load ``.env`` from the working directory if python-dotenv is installed.

    Without python-dotenv this is a no-op: an operating-system ``GEMINI_API_KEY`` still works.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


@dataclass
class GeminiConfig:
    api_key: str | None = field(default=None, repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 60.0
    temperature: float = 0.2

    def __repr__(self) -> str:
        return (
            f"GeminiConfig(model={self.model!r}, timeout_seconds={self.timeout_seconds!r}, "
            "api_key=<redacted>)"
        )

    @classmethod
    def from_env(cls, *, load_dotenv: bool = False) -> GeminiConfig:
        if load_dotenv:
            load_dotenv_file()
        try:
            timeout = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "300"))
        except ValueError as exc:
            raise AssistantConfigError("GEMINI_TIMEOUT_SECONDS must be a number.") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise AssistantConfigError("GEMINI_TIMEOUT_SECONDS must be a finite positive number.")
        return cls(
            api_key=os.environ.get("GEMINI_API_KEY") or None,
            model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
            timeout_seconds=timeout,
        )

    def require_key(self) -> str:
        if not self.api_key:
            raise AssistantConfigError(
                "GEMINI_API_KEY is not set. Set it as an environment variable, or in a local "
                ".env file (needs: python -m pip install -e '.[gemini]'; never commit it). "
                "Offline calculations do not need a key."
            )
        return self.api_key
