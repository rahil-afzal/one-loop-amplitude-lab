import json
from pathlib import Path

from amplitude_lab.cli import main
from amplitude_lab.derivation import (
    AUDIT_REQUEST,
    DERIVATION_CONTEXT,
    DERIVATION_STEPS,
    STEP_CONTEXT,
    run_derivation,
)
from amplitude_lab.llm import ScriptedLLM
from amplitude_lab.task import Task


def test_derivation_requests_answer_then_audit_and_saves_report(tmp_path):
    task = Task.load("phi4_dim4")
    steps = [
        "Vertex: +i c4; propagator: i/(p^2+i0).",
        "Tree amplitude: A_tree = c4.",
        "Bubble integral: iB(q^2) = integral ...",
        "Feynman parameter: shifted denominator ...",
        "Loop integral: Gamma(epsilon) ...",
        "Expansion: pole plus logarithm ...",
        "Subtracted bubble: B(q^2) = ...",
        "Symmetry factor: 1/2.",
        "Channels: s, t, and u terms ...",
        "Crossing sum: A = ...",
        "Branches: ln(-s-i0) = ...",
        "Crossing check: symmetric.",
        "Imaginary part check: Im A = ...",
        "Scale check: derivative = ...",
        "# Result\n\nThe boxed amplitude is \\(A(s,t,u)\\).",
    ]
    derivation = steps[-1]
    audit = "PASS: crossing. UNCERTAIN: independent reference comparison."
    llm = ScriptedLLM(steps)
    auditor = ScriptedLLM([audit])

    result = run_derivation(task, llm, runs_dir=tmp_path, audit_llm=auditor)

    assert result.status == "completed"
    assert result.error == ""
    assert result.audit_model == "scripted"
    assert result.elapsed_seconds >= 0
    assert result.derivation_seconds >= 0 and result.audit_seconds >= 0
    assert len(llm.calls) == len(steps)
    assert len(auditor.calls) == 1
    assert task.derivation_prompt in llm.calls[0][0]["text"]
    assert len(llm.calls[0]) == 1
    assert DERIVATION_CONTEXT in llm.calls[1][0]["text"]
    assert task.derivation_prompt not in llm.calls[1][0]["text"]
    for step_number in range(2, len(steps) + 1):
        prompt = llm.calls[step_number - 1][0]["text"]
        assert all(steps[prior_index] in prompt for prior_index in STEP_CONTEXT[step_number])
        assert all(
            steps[prior_index] not in prompt
            for prior_index in range(step_number - 1)
            if prior_index not in STEP_CONTEXT[step_number]
        )
        assert len(llm.calls[step_number - 1]) == 1
    assert auditor.calls[0][-1]["text"] == AUDIT_REQUEST
    audited_work = next(
        message["text"] for message in auditor.calls[0] if message["role"] == "assistant"
    )
    assert derivation in audited_work
    assert steps[0] in audited_work
    assert all(step in audited_work for step in steps)

    run_dir = Path(result.run_dir)
    assert (run_dir / "prompt.md").read_text(encoding="utf-8") == task.derivation_prompt
    report = (run_dir / "result.md").read_text(encoding="utf-8")
    assert all(step in report for step in steps)
    assert derivation in report and audit in report
    assert "not an independent proof" in report
    assert json.loads((run_dir / "outcome.json").read_text())["status"] == "completed"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["audit_llm"]["name"] == "scripted"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [event["stage"] for event in events] == [
        *(f"derivation_step_{index}" for index in range(1, len(steps) + 1)),
        "audit",
    ]


def test_derivation_failure_preserves_partial_answer(tmp_path):
    llm = ScriptedLLM(["A partial derivation.", ""])
    result = run_derivation(Task.load("phi4_dim4"), llm, runs_dir=tmp_path)

    assert result.status == "failed"
    assert "empty response" in result.error
    report = (Path(result.run_dir) / "result.md").read_text()
    assert "A partial derivation." in report
    assert "Workflow error" in report
    assert "Completed step: conventions" in llm.calls[1][0]["text"]
    events = [
        json.loads(line)
        for line in (Path(result.run_dir) / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["stage"] == "derivation_step_2"


def test_task_command_prints_analytical_prompt(capsys):
    assert main(["task", "--analytical"]) == 0
    prompt = capsys.readouterr().out
    assert "Analytical amplitude derivation" in prompt
    assert "derive them from the Lagrangian" in prompt
    assert "the target amplitude" in prompt
    assert "propagator factors of i" in prompt
    assert "the vertex is +i c4" not in prompt


def test_derivation_steps_require_explicit_sign_and_integral_bookkeeping():
    instructions = "\n".join(instruction for _, instruction in DERIVATION_STEPS)

    assert "derive" in instructions
    assert "denominator-only" in instructions
    assert "both propagator i factors" in instructions
    assert "First show the sum for iA" in instructions
    assert "Do not insert a memorized beta function" in instructions
    assert "resolve the conflict from the stated conventions" in instructions


def test_derive_uses_fixed_models_and_tunable_timeout(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LLM_MODEL", "environment-model-is-ignored")
    monkeypatch.setenv("LLM_AUDIT_MODEL", "environment-audit-is-ignored")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "90")

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
        calls.setdefault("clients", []).append(kwargs)
        return FakeLLM()

    monkeypatch.setattr("amplitude_lab.cli.llm_from_env", fake_llm_from_env)

    def fake_run_derivation(task, llm, runs_dir, echo, audit_llm):
        calls["task"] = task
        calls["llm"] = llm
        calls["audit_llm"] = audit_llm
        calls["runs_dir"] = runs_dir
        return FakeResult()

    monkeypatch.setattr("amplitude_lab.cli.run_derivation", fake_run_derivation)
    assert (
        main(
            [
                "derive",
                "--timeout-seconds",
                "45",
                "--runs-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "audit model" in output
    assert "result.md" in output
    assert calls["task"].derivation_prompt
    assert calls["runs_dir"] == str(tmp_path)
    assert [client["provider"] for client in calls["clients"]] == [
        "openai-compatible",
        "openai-compatible",
    ]
    assert [client["model"] for client in calls["clients"]] == [
        "qwen/qwen-3.8-max",
        "qwen/qwen-3.8-max",
    ]
    assert [client["timeout"] for client in calls["clients"]] == [45.0, 45.0]
    assert calls["audit_llm"] is not calls["llm"]


def test_derive_uses_timeout_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "70")
    clients = []

    class FakeLLM:
        def __init__(self, name):
            self.name = name

    class FakeResult:
        status = "completed"
        elapsed_seconds = 3
        run_dir = str(tmp_path / "run")
        error = ""

    def fake_llm_from_env(**kwargs):
        clients.append(kwargs)
        return FakeLLM(kwargs["model"] or "configured-primary")

    monkeypatch.setattr("amplitude_lab.cli.llm_from_env", fake_llm_from_env)
    monkeypatch.setattr(
        "amplitude_lab.cli.run_derivation",
        lambda task, llm, runs_dir, echo, audit_llm: FakeResult(),
    )

    assert main(["derive", "--runs-dir", str(tmp_path)]) == 0
    assert [client["model"] for client in clients] == [
        "qwen/qwen-3.8-max",
        "qwen/qwen-3.8-max",
    ]
    assert [client["timeout"] for client in clients] == [None, None]
