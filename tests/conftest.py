import sys

import pytest

from amplitude_lab.benchmarks import resolve


@pytest.fixture
def problem():
    return resolve("yhg_dim4")


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    # tests must never pick up a real key, from the environment or from a local .env
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_AUDIT_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setitem(sys.modules, "dotenv", None)
