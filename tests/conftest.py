import pytest

from amplitude_lab.benchmarks import pure_phi4, yhg_dim4


@pytest.fixture
def problem():
    return yhg_dim4()


@pytest.fixture
def pure_problem():
    return pure_phi4()


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Offline tests must never see a real key or a local .env file."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr("amplitude_lab.assistant.config.load_dotenv_file", lambda: None)
