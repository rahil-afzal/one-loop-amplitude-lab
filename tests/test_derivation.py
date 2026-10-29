import json
from pathlib import Path

from amplitude_lab.cli import main
from amplitude_lab.derivation import AUDIT_REQUEST, run_derivation
from amplitude_lab.llm import ScriptedLLM
from amplitude_lab.task import Task


def test_derivation_requests_answer_then_audit_and_saves_report(tmp_path):
    task = Task.load("phi4_dim4")
    derivation = "# Result\n\nThe boxed amplitude is \\(A(s,t,u)\\)."
    audit = "PASS: crossing. UNCERTAIN: independent reference comparison."
    llm = ScriptedLLM([derivation, audit])

    result = run_derivation(task, llm, runs_dir=tmp_path)

    assert result.status == "completed"
    assert result.error == ""
    assert result.elapsed_seconds >= 0
    assert result.derivation_seconds >= 0 and result.audit_seconds >= 0
    assert len(llm.calls) == 2
    assert llm.calls[0][0]["text"] == task.derivation_prompt
    assert llm.calls[1][-1]["text"] == AUDIT_REQUEST
    assert any(
        message["role"] == "assistant" and message["text"] == derivation
        for message in llm.calls[1]
    )

    run_dir = Path(result.run_dir)
    assert (run_dir / "prompt.md").read_text(encoding="utf-8") == task.derivation_prompt
    report = (run_dir / "result.md").read_text(encoding="utf-8")
    assert derivation in report and audit in report
    assert "not an independent proof" in report
    assert json.loads((run_dir / "outcome.json").read_text())["status"] == "completed"
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert [event["stage"] for event in events] == ["derivation", "audit"]


def test_derivation_failure_preserves_partial_answer(tmp_path):
    llm = ScriptedLLM(["A partial derivation.", ""])
    result = run_derivation(Task.load("phi4_dim4"), llm, runs_dir=tmp_path)

    assert result.status == "failed"
    assert "empty response" in result.error
    report = (Path(result.run_dir) / "result.md").read_text()
    assert "A partial derivation." in report
    assert "Workflow error" in report


def test_task_command_prints_analytical_prompt(capsys):
    assert main(["task", "--analytical"]) == 0
    assert "Analytical amplitude derivation" in capsys.readouterr().out


def test_derive_command_uses_two_pass_flow(monkeypatch, tmp_path, capsys):
    class FakeLLM:
        name = "gemini:test"

        @classmethod
        def from_env(cls, **kwargs):
            return cls()

    monkeypatch.setattr("amplitude_lab.cli.GeminiLLM", FakeLLM)

    class FakeResult:
        status = "completed"
        elapsed_seconds = 3
        run_dir = str(tmp_path / "run")
        error = ""

    calls = {}

    def fake_run_derivation(task, llm, runs_dir, echo):
        calls["task"] = task
        calls["llm"] = llm
        calls["runs_dir"] = runs_dir
        return FakeResult()

    monkeypatch.setattr("amplitude_lab.cli.run_derivation", fake_run_derivation)
    assert main(["derive", "--runs-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "separate model audit" in output
    assert "result.md" in output
    assert calls["task"].derivation_prompt
    assert calls["runs_dir"] == str(tmp_path)
