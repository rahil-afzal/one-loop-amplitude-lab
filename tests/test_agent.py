import json
import re

from amplitude_lab import __version__
from amplitude_lab.agent import parse_reply, run_agent
from amplitude_lab.llm import ScriptedLLM
from amplitude_lab.task import Task

from .conftest import CANDIDATES


def block(kind, code):
    return f"some thoughts\n```{kind}\n{code}\n```\n"


def code_of(name):
    return (CANDIDATES / f"{name}.py").read_text(encoding="utf-8")


def small_task(**kw):
    t = Task.load("phi4_dim4")
    t.stages = []
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def only_run(tmp_path):
    return next(tmp_path.iterdir())


def test_parse_reply():
    assert parse_reply(block("run", "print(1)")) == ("run", "print(1)\n", 1)
    assert parse_reply(block("submit", "x = 1"))[0] == "submit"
    assert parse_reply("no block here") is None
    assert parse_reply("```python\nprint(1)\n```") is None


def test_parse_reply_counts_extra_blocks_and_keeps_inner_backticks():
    two = "```run\nprint(1)\n```\nthen\n```submit\nx=1\n```"
    assert parse_reply(two) == ("run", "print(1)\n", 2)
    code = "s = '```'\nprint(s)\n"  # a fence inside a line is not a closing fence
    assert parse_reply(f"```run\n{code}```") == ("run", code, 1)


def test_task_statement():
    t = Task.load("phi4_dim4")
    assert "10 submissions" in t.statement and "{max_attempts}" not in t.statement
    assert "stage-specific instructions" in t.statement
    for name in ("crossing", "unitarity", "dispersion", "rg_beta", "reference_match"):
        assert name in t.statement
    assert "s = (p1+p2)^2" in t.statement  # the invariants are defined


def test_staged_agent_uses_separate_prompts_and_advances_after_runs(tmp_path, monkeypatch):
    task = Task.load("phi4_dim4")

    class PassingReport:
        passed = True
        structural_ok = True

        def to_dict(self):
            return {"passed": self.passed, "structural_ok": self.structural_ok}

        def feedback(self, attempt, max_attempts):
            return f"submission {attempt} passed"

    monkeypatch.setattr("amplitude_lab.agent.verify", lambda candidate, suite: PassingReport())
    llm = ScriptedLLM(
        [
            block("run", "print('branch check')"),
            block("run", "print('s-channel check')"),
            block("submit", code_of("correct")),
        ]
    )
    result = run_agent(task, llm, runs_dir=tmp_path)
    assert result.solved and result.turns == 3 and result.n_run_blocks == 2
    first_prompt = llm.calls[0][0]["text"]
    second_prompt = llm.calls[1][-1]["text"]
    third_prompt = llm.calls[2][-1]["text"]
    assert "Stage 1/3: Check conventions" in first_prompt
    assert "Stage 2/3: Check one channel" in second_prompt
    assert "Stage 3/3: Assemble and submit" in third_prompt
    assert any(
        "output:" in message["text"] for message in llm.calls[1] if message["role"] == "user"
    )
    events = rows(only_run(tmp_path) / "events.jsonl")
    assert [event["kind"] for event in events] == ["run", "run", "submit"]
    assert [event["stage"] for event in events] == [1, 2, 3]
    meta = json.loads((only_run(tmp_path) / "meta.json").read_text())
    assert [stage["name"] for stage in meta["stages"]] == [
        "Check conventions",
        "Check one channel",
        "Assemble and submit",
    ]


def test_staged_agent_retries_a_failed_calculation_before_advancing(tmp_path, monkeypatch):
    class PassingReport:
        passed = True
        structural_ok = True

        def to_dict(self):
            return {"passed": self.passed, "structural_ok": self.structural_ok}

        def feedback(self, attempt, max_attempts):
            return f"submission {attempt} passed"

    monkeypatch.setattr("amplitude_lab.agent.verify", lambda candidate, suite: PassingReport())
    task = Task.load("phi4_dim4")
    llm = ScriptedLLM(
        [
            block("run", "raise RuntimeError('retry this stage')"),
            block("run", "print('branch check')"),
            block("run", "print('s-channel check')"),
            block("submit", code_of("correct")),
        ]
    )
    result = run_agent(task, llm, runs_dir=tmp_path)
    assert result.solved and result.turns == 4
    assert "did not complete successfully" in llm.calls[1][-1]["text"]
    assert "Stage 1/3 (Check conventions)" in llm.calls[1][-1]["text"]
    assert "Stage 2/3: Check one channel" in llm.calls[2][-1]["text"]


def test_wrong_then_right(tmp_path):
    llm = ScriptedLLM(
        [
            block("run", "print(1 + 1)"),
            block("submit", code_of("wrong_factor")),
            block("submit", code_of("correct")),
        ]
    )
    result = run_agent(small_task(), llm, runs_dir=tmp_path)
    assert result.solved and result.attempts == 2 and result.turns == 3
    assert result.n_run_blocks == 1 and result.first_solved_attempt == 2
    assert result.structural_solved and result.first_structural_attempt == 2

    seen = llm.calls[2]
    assert seen[2]["text"].startswith("output:\n2")
    assert any("FAIL unitarity" in m["text"] for m in seen)

    run = only_run(tmp_path)
    attempts = rows(run / "attempts.jsonl")
    assert [a["passed"] for a in attempts] == [False, True]
    assert all(isinstance(a["seed"], int) for a in attempts)
    assert attempts[0]["seed"] != attempts[1]["seed"]  # fresh points every submission
    assert (run / "attempt_01.py").exists() and (run / "attempt_02.py").exists()
    assert json.loads((run / "outcome.json").read_text())["solved"] is True
    assert "agent (turn 1)" in (run / "transcript.md").read_text()


def test_agent_echoes_progress_and_elapsed_time(tmp_path, monkeypatch):
    class PassingReport:
        passed = True
        structural_ok = True

        def to_dict(self):
            return {"passed": self.passed, "structural_ok": self.structural_ok}

        def feedback(self, attempt, max_attempts):
            return f"submission {attempt} passed"

    monkeypatch.setattr("amplitude_lab.agent.verify", lambda candidate, suite: PassingReport())
    progress = []
    result = run_agent(
        small_task(),
        ScriptedLLM([block("submit", code_of("correct"))]),
        runs_dir=tmp_path,
        echo=progress.append,
    )
    assert result.elapsed_seconds >= 0
    assert any("Sending prompt to model" in line and "elapsed:" in line for line in progress)
    assert any("Model response received" in line for line in progress)
    assert any("Checking submission" in line for line in progress)
    assert any("Submission 1 passed" in line for line in progress)


def test_run_directory_records_what_was_run(tmp_path):
    task = small_task()
    run_agent(task, ScriptedLLM([block("submit", code_of("correct"))]), runs_dir=tmp_path)
    run = only_run(tmp_path)
    meta = json.loads((run / "meta.json").read_text())
    assert meta["task"] == "phi4_dim4" and meta["suite"] == "phi4_dim4"
    assert meta["statement_sha256"] == task.statement_sha256
    assert meta["llm"]["name"] == "scripted"
    assert meta["limits"]["max_attempts"] == 10
    assert meta["environment"]["amplitude_lab"] == __version__
    assert {"python", "sympy", "mpmath", "platform"} <= set(meta["environment"])
    assert (run / "statement.md").read_text() == task.statement
    events = rows(run / "events.jsonl")
    assert [e["kind"] for e in events] == ["submit"]
    assert events[0]["passed"] is True and "verify_s" in events[0]


def test_events_count_tool_use(tmp_path):
    llm = ScriptedLLM(
        [block("run", "print(1)"), "just talking", block("run", "print(2)"), block("submit", "x")]
    )
    result = run_agent(small_task(max_attempts=1), llm, runs_dir=tmp_path)
    assert (result.n_run_blocks, result.n_no_block, result.attempts) == (2, 1, 1)
    kinds = [e["kind"] for e in rows(only_run(tmp_path) / "events.jsonl")]
    assert kinds == ["run", "no_block", "run", "submit"]


def test_structural_and_reference_outcomes_are_reported_separately(tmp_path):
    llm = ScriptedLLM([block("submit", code_of("wrong_constant"))] * 2)
    result = run_agent(small_task(max_attempts=2), llm, runs_dir=tmp_path)
    assert not result.solved and result.structural_solved
    assert result.first_structural_attempt == 1 and result.first_solved_attempt == 0


def test_stops_when_submissions_run_out(tmp_path):
    llm = ScriptedLLM([block("submit", code_of("wrong_factor"))] * 3)
    result = run_agent(small_task(max_attempts=2), llm, runs_dir=tmp_path)
    assert not result.solved and result.attempts == 2 and result.reason == "out of submissions"
    assert not result.structural_solved
    assert len(llm.calls) == 2


def test_stops_at_turn_limit(tmp_path):
    llm = ScriptedLLM([block("run", "print('hi')")] * 5)
    result = run_agent(small_task(max_turns=3), llm, runs_dir=tmp_path)
    assert not result.solved and result.turns == 3 and result.attempts == 0


def test_reply_without_a_block_gets_a_nudge(tmp_path):
    llm = ScriptedLLM(["I would rather talk about it.", block("submit", code_of("correct"))])
    assert run_agent(small_task(), llm, runs_dir=tmp_path).solved
    assert "No `run` or `submit` block" in llm.calls[1][-1]["text"]


def test_untagged_python_block_is_named_in_the_nudge(tmp_path):
    llm = ScriptedLLM(["```python\nprint(1)\n```", "x"])
    run_agent(small_task(max_turns=2), llm, runs_dir=tmp_path)
    assert "```python" in llm.calls[1][-1]["text"]


def test_second_block_is_reported_as_unused(tmp_path):
    two = "```run\nprint(1)\n```\n```submit\nx = 1\n```"
    llm = ScriptedLLM([two, "x"])
    run_agent(small_task(max_turns=2), llm, runs_dir=tmp_path)
    assert "Only the first block was used" in llm.calls[1][-1]["text"]


def test_forbidden_import_in_run_block(tmp_path):
    llm = ScriptedLLM([block("run", "import os\nprint(os.listdir('.'))"), "x"])
    run_agent(small_task(max_turns=2), llm, runs_dir=tmp_path)
    assert "not allowed" in llm.calls[1][-1]["text"]


def test_llm_failure_is_recorded(tmp_path):
    result = run_agent(small_task(), ScriptedLLM([]), runs_dir=tmp_path)
    assert not result.solved and result.reason.startswith("llm error")
    assert rows(only_run(tmp_path) / "events.jsonl")[0]["kind"] == "llm_error"


def test_nothing_numeric_reaches_the_agent_even_from_a_leaky_candidate(tmp_path):
    """End to end: a candidate that tries to smuggle the test points out through its
    exception text must not get them into any message the model sees."""
    llm = ScriptedLLM([block("submit", code_of("leaks_points")), "x"])
    run_agent(small_task(max_turns=2), llm, runs_dir=tmp_path)
    to_agent = " ".join(m["text"] for m in llm.calls[1] if m["role"] == "user")
    assert "ValueError" in to_agent
    assert not re.search(r"\d+\.\d+", to_agent)
    seed = rows(only_run(tmp_path) / "attempts.jsonl")[0]["seed"]
    assert str(seed) not in to_agent


def test_the_answer_key_cannot_be_used_by_a_submission(tmp_path):
    llm = ScriptedLLM([block("submit", code_of("reads_reference"))] * 2)
    result = run_agent(small_task(max_attempts=2), llm, runs_dir=tmp_path)
    assert not result.solved and not result.structural_solved
