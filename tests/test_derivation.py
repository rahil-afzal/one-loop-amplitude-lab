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
    llm = ScriptedLLM([derivation])
    auditor = ScriptedLLM([audit])

    result = run_derivation(task, llm, runs_dir=tmp_path, audit_llm=auditor)

    assert result.status == "completed"
    assert result.error == ""
    assert result.audit_model == "scripted"
    assert result.elapsed_seconds >= 0
    assert result.derivation_seconds >= 0 and result.audit_seconds >= 0
    assert len(llm.calls) == len(auditor.calls) == 1
    assert llm.calls[0][0]["text"] == task.derivation_prompt
    assert auditor.calls[0][-1]["text"] == AUDIT_REQUEST
    assert any(
        message["role"] == "assistant" and message["text"] == derivation
        for message in auditor.calls[0]
    )

    run_dir = Path(result.run_dir)
    assert (run_dir / "prompt.md").read_text(encoding="utf-8") == task.derivation_prompt
    report = (run_dir / "result.md").read_text(encoding="utf-8")
    assert derivation in report and audit in report
    assert "not an independent proof" in report
    assert json.loads((run_dir / "outcome.json").read_text())["status"] == "completed"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["audit_llm"]["name"] == "scripted"
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

    class FakeResult:
        status = "completed"
        elapsed_seconds = 3
        run_dir = str(tmp_path / "run")
        error = ""

    calls = {}

    def fake_llm_from_env(**kwargs):
        calls.setdefault("models", []).append(kwargs["model"])
        calls["provider"] = kwargs["provider"]
        return FakeLLM()

    monkeypatch.setattr("amplitude_lab.cli.llm_from_env", fake_llm_from_env)

    def fake_run_derivation(task, llm, runs_dir, echo, audit_llm):
        calls["task"] = task
        calls["llm"] = llm
        calls["audit_llm"] = audit_llm
        calls["runs_dir"] = runs_dir
        return FakeResult()

    monkeypatch.setattr("amplitude_lab.cli.run_derivation", fake_run_derivation)
    assert main(
        [
            "derive",
            "--model",
            "qwen-api-id",
            "--audit-model",
            "deepseek-api-id",
            "--runs-dir",
            str(tmp_path),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "audit model" in output
    assert "result.md" in output
    assert calls["task"].derivation_prompt
    assert calls["runs_dir"] == str(tmp_path)
    assert calls["provider"] is None
    assert calls["models"] == ["qwen-api-id", "deepseek-api-id"]
    assert calls["audit_llm"] is not calls["llm"]
