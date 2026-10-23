"""Real network test. Opt in with RUN_GEMINI_SMOKE=1 and a real GEMINI_API_KEY."""

import os

import pytest

from amplitude_lab.assistant.client import GeminiClient
from amplitude_lab.assistant.config import GeminiConfig

pytestmark = pytest.mark.gemini


@pytest.mark.skipif(
    os.environ.get("RUN_GEMINI_SMOKE") != "1",
    reason="Opt-in: set RUN_GEMINI_SMOKE=1 and GEMINI_API_KEY to run.",
)
def test_real_gemini_smoke(monkeypatch):
    # The autouse fixture clears the key; the opt-in test needs the real environment.
    monkeypatch.undo()
    config = GeminiConfig.from_env(load_dotenv=True)
    config.require_key()
    assert GeminiClient(config).smoke_test().strip().upper().startswith("OK")
