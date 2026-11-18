import pytest

from amplitude_lab.assistant import BubbleClosedFormTask, run_task
from amplitude_lab.assistant.errors import (
    AssistantRequestError,
)
from amplitude_lab.assistant.mock import MOCK_PROPOSALS, ScriptedClient, mock_client

TASK = BubbleClosedFormTask()


def test_correct_proposal_verified(problem):
    outcome = run_task(mock_client("correct"), TASK, problem)
    assert outcome.status == "verified"
    assert outcome.verification["passed"]


@pytest.mark.parametrize("kind", ["wrong_sign", "wrong_branch", "explicit_ipi"])
def test_wrong_candidates_rejected_by_deterministic_verifier(problem, kind):
    outcome = run_task(mock_client(kind), TASK, problem)
    assert outcome.status == "rejected"
    assert not outcome.verification["passed"]


@pytest.mark.parametrize("kind", ["malicious", "garbage"])
def test_unsafe_candidates_rejected_before_any_evaluation(problem, kind):
    outcome = run_task(mock_client(kind), TASK, problem)
    assert outcome.status == "rejected_unparseable"
    assert outcome.verification is None


def test_malicious_payload_has_no_side_effect(problem, tmp_path, monkeypatch):
    marker = tmp_path / "pwned"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    outcome = run_task(ScriptedClient(payload), TASK, problem)
    assert outcome.status == "rejected_unparseable"
    assert not marker.exists()


def test_correct_form_is_not_taken_on_trust(problem):
    """Equivalent-looking forms with the wrong sheet must fail on the boundary points."""
    outcome = run_task(ScriptedClient("2 - log(s/mu2)"), TASK, problem)
    assert outcome.status == "rejected"


def test_every_mock_kind_has_a_defined_outcome(problem):
    expected = {
        "correct": "verified",
        "wrong_sign": "rejected",
        "wrong_branch": "rejected",
        "explicit_ipi": "rejected",
        "malicious": "rejected_unparseable",
        "garbage": "rejected_unparseable",
    }
    assert set(expected) == set(MOCK_PROPOSALS)
    for kind, status in expected.items():
        assert run_task(mock_client(kind), TASK, problem).status == status


def test_client_failure_becomes_unavailable_outcome(problem):
    class Failing:
        name = "failing"

        def generate(self, prompt, system_instruction=None):
            raise AssistantRequestError("network down")

    outcome = run_task(Failing(), TASK, problem)
    assert outcome.status == "unavailable" and "network down" in outcome.detail


def test_outcome_is_json_ready(problem):
    import json

    json.dumps(run_task(mock_client("correct"), TASK, problem).to_dict())


def test_prompt_is_sent_to_client(problem):
    client = mock_client("correct")
    run_task(client, TASK, problem)
    assert len(client.calls) == 1
    assert "closed form" in client.calls[0][0]
    assert "principal" in client.calls[0][1]


