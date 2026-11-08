import json
from pathlib import Path

from amplitude_lab.workflows.autonomous import (
    AUDIT_SYSTEM,
    DERIVATION_SYSTEM,
    PlanError,
    _audit_response,
    _format_audit_response,
    _plan_overview,
    parse_plan,
    run_autonomous_derivation,
)
from amplitude_lab.workflows.llm import LLMError, ScriptedLLM
from amplitude_lab.workflows.task import Task

GOOD = (
    "Derivation and checks.\n"
    "```final\n"
    "A_tree = c4\n"
    "A_loop = c4**2/(32*pi**2) * "
    "(6 - log(-s/mu2) - log(-t/mu2) - log(-u/mu2))\n"
    "beta_c4 = -3*c4**2/(16*pi**2)\n"
    "```"
)
BAD = GOOD.replace("6 -", "3 -")


def plan(*rows):
    return json.dumps(
        {
            "interpretation": "The task requests a massless scalar 2-to-2 amplitude.",
            "lagrangian": "$\\mathcal{L}=\\frac12(\\partial\\phi)^2+\\frac{c_4}{4!}\\phi^4$",
            "strategy": (
                "Derive rules, enumerate contributing diagrams, then evaluate the amplitude."
            ),
            "assumptions": ["All external particles are on shell."],
            "steps": [
                {
                    "id": step_id,
                    "goal": (
                        f"Establish and justify the Lagrangian; {goal}"
                        if index == 0 and "lagrangian" not in goal.casefold()
                        else goal
                    ),
                    "depends_on": dependencies,
                }
                for index, (step_id, goal, dependencies) in enumerate(rows)
            ]
        }
    )


def audit(verdict="ACCEPT", target="final", diagnostic="The equations are consistent."):
    return json.dumps({"verdict": verdict, "target_step_id": target, "diagnostic": diagnostic})


def execute(tmp_path, derivation_replies, audit_replies, retries=3, on_event=None):
    return run_autonomous_derivation(
        Task.load("phi4_dim4"),
        ScriptedLLM(derivation_replies),
        ScriptedLLM(audit_replies),
        runs_dir=tmp_path,
        max_step_retries=retries,
        on_event=on_event,
    )


def test_plan_requires_a_valid_dependency_order():
    steps = parse_plan(plan(("rules", "derive rules", []), ("final", "derive answer", ["rules"])))
    assert [step.id for step in steps] == ["rules", "final"]
    try:
        parse_plan(plan(("first", "derive", ["later"]), ("later", "derive", [])))
    except ValueError as exc:
        assert "earlier steps" in str(exc)
    else:
        raise AssertionError("forward dependency was accepted")


def test_derivation_prompt_requests_checkable_intermediate_explanations():
    assert "essential intermediate equations" in DERIVATION_SYSTEM
    assert "Avoid repeating the problem" in DERIVATION_SYSTEM
    assert "dimensional or limiting-case sanity check" in DERIVATION_SYSTEM
    assert "notation contract in the accepted checkpoints" in DERIVATION_SYSTEM
    assert "checkpoint" in DERIVATION_SYSTEM
    assert "Revision notes" in DERIVATION_SYSTEM
    assert "Do not provide private chain-of-thought" in DERIVATION_SYSTEM


def test_audit_prompt_and_response_use_scoped_text_format():
    assert "Audit only this step's stated goal and deliverables" in AUDIT_SYSTEM
    assert "VERDICT: ACCEPT|REJECT" in AUDIT_SYSTEM
    response = (
        "VERDICT: REJECT\n"
        "PREVIOUS: none\n"
        "BLOCKING: sign | eq. (3) | sign conflicts with convention | re-derive sign\n"
        "MINOR: none\n"
        "INVALIDATE: current\n"
    )
    verdict, target, diagnostic = _audit_response(response, {"current"}, "current")
    assert verdict == "REJECT"
    assert target == "current"
    assert "sign conflicts with convention" in diagnostic


def test_audit_text_format_rejects_acceptance_with_blocking_issue():
    response = (
        "VERDICT: ACCEPT\n"
        "PREVIOUS: none\n"
        "BLOCKING: sign | eq. (3) | inconsistent | re-derive\n"
        "MINOR: none\n"
        "INVALIDATE: none\n"
    )
    verdict, target, _ = _audit_response(response, {"current"}, "current")
    assert verdict == "UNCERTAIN"
    assert target == "current"


def test_plan_requires_agent_defined_lagrangian():
    payload = json.loads(plan(("final", "derive final amplitude", [])))
    del payload["lagrangian"]
    try:
        _plan_overview(json.dumps(payload))
    except PlanError as exc:
        assert "explicit Lagrangian" in str(exc)
    else:
        raise AssertionError("planner response without a Lagrangian was accepted")


def test_first_plan_step_must_establish_lagrangian(tmp_path):
    payload = json.loads(plan(("rules", "derive Feynman rules", [])))
    payload["steps"][0]["goal"] = "Derive Feynman rules"
    result = execute(tmp_path, [json.dumps(payload)], [])
    assert result.status == "failed"
    assert "first plan step must establish" in result.error


def test_inconsistent_audit_target_cannot_accept_a_step():
    from amplitude_lab.workflows.autonomous import _audit_response

    verdict, target, diagnostic = _audit_response(
        audit("ACCEPT", "earlier"),
        {"earlier", "current"},
        "current",
    )
    assert verdict == "UNCERTAIN"
    assert target == "current"
    assert "re-derive" in diagnostic


def test_audit_json_is_rendered_as_readable_markdown():
    response = (
        '```json\n{"verdict":"REJECT","target_step_id":"rules",'
        '"diagnostic":"The sign in $S_0$ is inconsistent. Re-derive it."}\n```'
    )

    rendered = _format_audit_response(response)

    assert "**Verdict:** REJECT" in rendered
    assert "**Review target:** `rules`" in rendered
    assert "**Auditor's explanation:**" in rendered
    assert "The sign in $S_0$ is inconsistent. Re-derive it." in rendered
    assert "```json" not in rendered


def test_success_streams_and_saves_audited_deterministic_result(tmp_path):
    events = []
    result = execute(
        tmp_path,
        [
            plan(
                ("rules", "derive Feynman rules", []),
                ("final", "derive final amplitude", ["rules"]),
            ),
            "The rules follow by differentiating the interaction and kinetic terms.",
            GOOD,
        ],
        [audit("ACCEPT", "rules"), audit("ACCEPT", "final")],
        on_event=events.append,
    )

    assert result.status == "completed_verified"
    assert result.verification["passed"] is True
    assert result.interpretation.startswith("The task requests")
    assert result.lagrangian.startswith("$\\mathcal{L}")
    assert result.strategy.startswith("Derive rules")
    assert result.assumptions == ["All external particles are on shell."]
    assert "A_loop =" in result.final_amplitude
    assert [step["step_id"] for step in result.accepted_steps] == ["rules", "final"]
    assert any(event["type"] == "token" for event in events)
    assert any(event["state"] == "awaiting_audit" for event in events)
    assert any(event["type"] == "verification.finished" for event in events)
    saved = Path(result.run_dir)
    assert not (saved / "events.jsonl").exists()
    assert json.loads((saved / "outcome.json").read_text())["status"] == "completed_verified"
    report = (saved / "result.md").read_text()
    assert "## Problem interpretation" in report
    assert "## Selected approach" in report
    assert "All external particles are on shell." in report
    assert "Attempts and audit history" in report
    assert "Deterministic verification" in report
    assert "**Verdict:** ACCEPT" in report
    assert "```json" not in report
    response = (saved / "response.md").read_text()
    assert "## Proposed Lagrangian and conventions" in response
    assert "## Derivation" in response
    assert "## Result" in response
    assert "A_loop =" in response
    assert (saved / "steps.md").is_file()


def test_step_prompts_reuse_compact_context_and_only_relevant_checkpoints(tmp_path):
    task = Task(
        "context_reuse",
        "UNIQUE_RAW_REQUIREMENT: derive the scalar amplitude with the stated convention.",
    )
    derivation_llm = ScriptedLLM(
        [
            plan(
                ("rules", "derive the Lagrangian and Feynman rules", []),
                ("unrelated", "derive an independent auxiliary result", []),
                ("final", "derive the final amplitude", ["rules"]),
            ),
            "Lagrangian and rules checkpoint.",
            "Independent auxiliary checkpoint.",
            GOOD,
        ]
    )
    audit_llm = ScriptedLLM(
        [
            audit("ACCEPT", "rules"),
            audit("ACCEPT", "unrelated"),
            audit("ACCEPT", "final"),
        ]
    )

    result = run_autonomous_derivation(
        task, derivation_llm, audit_llm, runs_dir=tmp_path
    )

    assert result.status == "completed_unverified"
    first_step_prompt = json.loads(derivation_llm.calls[1][0]["text"])
    later_step_prompt = json.loads(derivation_llm.calls[3][0]["text"])
    assert first_step_prompt["original_problem"] == task.derivation_prompt
    assert "original_problem" not in later_step_prompt
    assert "UNIQUE_RAW_REQUIREMENT" not in derivation_llm.calls[3][0]["text"]

    final_audit_prompt = json.loads(audit_llm.calls[2][0]["text"])
    assert "UNIQUE_RAW_REQUIREMENT" not in audit_llm.calls[2][0]["text"]
    assert [row["step_id"] for row in final_audit_prompt["accepted_checkpoints"]] == [
        "rules"
    ]


def test_rejection_returns_diagnostic_and_derivation_corrects_it(tmp_path):
    result = execute(
        tmp_path,
        [plan(("final", "derive final amplitude", [])), BAD, GOOD],
        [
            audit("REJECT", "final", "Recheck the finite constant from the parameter integral."),
            audit("ACCEPT", "final"),
        ],
        retries=1,
    )
    assert result.status == "completed_verified"
    assert [attempt.status for attempt in result.attempts] == ["rejected", "accepted"]
    assert not (Path(result.run_dir) / "events.jsonl").exists()
    steps = (Path(result.run_dir) / "steps.md").read_text()
    assert "Recheck the finite constant" in steps
    assert "rejected" in steps


def test_earlier_checkpoint_invalidates_dependent_work_and_backtracks(tmp_path):
    result = execute(
        tmp_path,
        [
            plan(
                ("rules", "derive rules", []),
                ("diagrams", "enumerate diagrams", ["rules"]),
                ("final", "derive final amplitude", ["diagrams"]),
            ),
            "Incorrect initial convention.",
            "Diagram count based on the initial convention.",
            GOOD,
            "Corrected convention derived from the Lagrangian.",
            "Recomputed diagrams from corrected rules.",
            GOOD,
        ],
        [
            audit("ACCEPT", "rules"),
            audit("ACCEPT", "diagrams"),
            audit("REJECT", "rules", "The vertex sign conflicts with the stated Lagrangian."),
            audit("ACCEPT", "rules"),
            audit("ACCEPT", "diagrams"),
            audit("ACCEPT", "final"),
        ],
    )
    assert result.status == "completed_verified"
    assert len(result.backtracking) == 1
    assert result.backtracking[0].target_step_id == "rules"
    assert result.backtracking[0].invalidated_steps == ["rules", "diagrams", "final"]
    assert [row["content"] for row in result.accepted_steps] == [
        "Corrected convention derived from the Lagrangian.",
        "Recomputed diagrams from corrected rules.",
        GOOD,
    ]


def test_retry_exhaustion_does_not_accept_unresolved_step(tmp_path):
    result = execute(
        tmp_path,
        [plan(("final", "derive final amplitude", [])), BAD, BAD],
        [
            audit("REJECT", "final", "Unresolved coefficient mismatch."),
            audit("REJECT", "final", "The same coefficient mismatch remains."),
        ],
        retries=1,
    )
    assert result.status == "retry_exhausted"
    assert not result.accepted_steps
    assert len(result.attempts) == 2
    assert result.error


def test_repeated_backtracking_obeys_retry_limit(tmp_path):
    result = execute(
        tmp_path,
        [
            plan(
                ("rules", "derive rules", []),
                ("final", "derive final amplitude", ["rules"]),
            ),
            "Initial rules.",
            GOOD,
            "Revised rules.",
            GOOD,
        ],
        [
            audit("ACCEPT", "rules"),
            audit("REJECT", "rules", "The earlier vertex sign is inconsistent."),
            audit("ACCEPT", "rules"),
            audit("REJECT", "rules", "The revised vertex sign is still inconsistent."),
        ],
        retries=1,
    )
    assert result.status == "retry_exhausted"
    assert "rules" in result.error
    assert len(result.backtracking) == 2
    assert not result.accepted_steps


def test_deterministic_failure_rejects_auditor_accepted_final_and_retries(tmp_path):
    result = execute(
        tmp_path,
        [plan(("final", "derive final amplitude", [])), BAD, GOOD],
        [audit("ACCEPT", "final"), audit("ACCEPT", "final")],
        retries=1,
    )
    assert result.status == "completed_verified"
    assert result.attempts[0].status == "deterministic_rejection"
    assert result.attempts[0].deterministic_checks["passed"] is False
    assert result.attempts[1].deterministic_checks["passed"] is True


def test_stream_callback_failure_is_reported_and_run_is_persisted(tmp_path):
    class BrokenStream:
        name = "broken-stream"
        stream = True

        def complete_stream(self, _system, _messages, on_token):
            on_token("partial")
            raise LLMError("provider stream disconnected")

    events = []

    def broken_sink(event):
        events.append(event)
        if event["type"] == "token":
            raise RuntimeError("UI disconnected")

    result = run_autonomous_derivation(
        Task.load("phi4_dim4"),
        BrokenStream(),
        ScriptedLLM([]),
        runs_dir=tmp_path,
        on_event=broken_sink,
    )
    assert result.status == "failed"
    assert "event stream callback failed" in result.error
    assert (Path(result.run_dir) / "outcome.json").is_file()


def test_provider_stream_failure_preserves_partial_attempt(tmp_path):
    class BrokenStream:
        name = "broken-provider"
        stream = True

        def __init__(self):
            self.calls = 0

        def complete_stream(self, _system, _messages, on_token):
            self.calls += 1
            if self.calls == 1:
                return plan(("final", "derive final amplitude", []))
            on_token("partial derivation")
            raise LLMError("provider stream disconnected")

    derivation = BrokenStream()
    result = run_autonomous_derivation(
        Task.load("phi4_dim4"),
        derivation,
        ScriptedLLM([]),
        runs_dir=tmp_path,
    )
    assert result.status == "failed"
    assert "provider stream disconnected" in result.error
    assert result.attempts[0].status == "generation_failed"
    assert result.attempts[0].proposal == "partial derivation"
