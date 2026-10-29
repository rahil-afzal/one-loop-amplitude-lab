import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "examples" / "candidates"


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    # tests must never pick up a real key, from the environment or from a local .env
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setitem(sys.modules, "dotenv", None)
