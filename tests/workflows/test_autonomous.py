import json
from pathlib import Path

import amplitude_lab.workflows.autonomous as autonomous
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


def nested_plan(*rows):
    return json.dumps(
        {
            "interpretation": "A nested checkpoint test.",
            "lagrangian": "$L = \\frac12(\\partial\\phi)^2$",
            "strategy": "Derive and audit each bounded substep.",
            "assumptions": [],
            "steps": [
                {
                    "id": step_id,
                    "goal": parent_goal,
                    "depends_on": parent_dependencies,
                    "substeps": substeps,
                }
                for step_id, parent_goal, parent_dependencies, substeps in rows
            ],
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


def test_nested_plan_validates_child_schema_and_legacy_rows_remain_permissive():
    valid = nested_plan(
        (
            "rules",
            "Derive the rules",
            [],
            [
                {"id": "lagrangian", "goal": "Establish the Lagrangian"},
                {"id": "sign", "goal": "Fix the sign", "depends_on": ["lagrangian"]},
            ],
        )
    )
    steps = parse_plan(valid)
    assert steps[0].explicit_substeps
    assert [child.id for child in steps[0].substeps] == ["lagrangian", "sign"]
    assert steps[0].substeps[1].depends_on == ("lagrangian",)

    legacy = json.loads(plan(("rules", "derive rules", [])))
    legacy["steps"][0]["unrelated_legacy_field"] = True
    assert not parse_plan(json.dumps(legacy))[0].explicit_substeps

    malformed = json.loads(valid)
    malformed["steps"][0]["substeps"][1]["extra"] = "rejected"
    try:
        parse_plan(json.dumps(malformed))
    except PlanError as exc:
        assert "exact child schema" in str(exc)
    else:
        raise AssertionError("unknown nested fields were accepted")

    malformed = json.loads(valid)
    malformed["steps"][0]["substeps"][0]["depends_on"] = ["sign"]
    try:
        parse_plan(json.dumps(malformed))
    except PlanError as exc:
        assert "earlier substeps" in str(exc)
    else:
        raise AssertionError("a forward child dependency was accepted")


def test_rejected_substep_retries_locally_and_consumes_dependencies_once(tmp_path):
    task = Task("substep_retry", "Derive a simple scalar expression.")
    derivation = ScriptedLLM(
        [
            nested_plan(
                (
                    "rules",
                    "Derive the rule",
                    [],
                    [
                        {"id": "lagrangian", "goal": "Establish the Lagrangian"},
                        {
                            "id": "factor",
                            "goal": "Derive the finite factor",
                            "depends_on": ["lagrangian"],
                        },
                    ],
                )
            ),
            "Lagrangian checkpoint.",
            "An incorrect factor.",
            "The corrected factor.",
        ]
    )
    auditor = ScriptedLLM(
        [
            audit("ACCEPT", "rules"),
            audit("REJECT", "current", "Re-derive the finite factor."),
            audit("ACCEPT", "rules"),
        ]
    )
    result = run_autonomous_derivation(
        task, derivation, auditor, runs_dir=tmp_path, max_substep_retries=1
    )
    assert result.status == "completed_unverified"
    assert len(derivation.calls) == 4
    assert [item.substep_id for item in result.attempts] == [
        "lagrangian",
        "factor",
        "factor",
    ]
    assert result.accepted_steps[0]["content"] == (
        "Lagrangian checkpoint.\n\nThe corrected factor."
    )
    final_audit_prompt = json.loads(auditor.calls[-1][0]["text"])
    assert "original_problem" not in final_audit_prompt
    assert [item["substep_id"] for item in final_audit_prompt["accepted_checkpoints"]] == [
        "lagrangian"
    ]


def test_valid_inserted_substep_gates_only_its_rejected_target(tmp_path):
    task = Task("insert_substep", "Derive a scalar expression.")
    events = []
    derivation = ScriptedLLM(
        [
            nested_plan(
                (
                    "rules",
                    "Derive the result",
                    [],
                    [
                        {"id": "lagrangian", "goal": "Establish the Lagrangian"},
                        {
                            "id": "answer",
                            "goal": "Derive the final expression",
                            "depends_on": ["lagrangian"],
                        },
                    ],
                )
            ),
            "Lagrangian checkpoint.",
            "Incomplete answer.",
            "Derivation of the omitted factor.",
            "Corrected answer using the factor.",
        ]
    )
    auditor = ScriptedLLM(
        [
            audit("ACCEPT", "rules"),
            (
                "VERDICT: REJECT\n"
                "BLOCKING: missing | omitted factor | derive it\n"
                "INVALIDATE: current\n"
                'INSERT_SUBSTEP: {"id":"factor","goal":"Derive the omitted factor",'
                '"depends_on":["lagrangian"]}\n'
            ),
            audit("ACCEPT", "rules"),
            audit("ACCEPT", "rules"),
        ]
    )
    result = run_autonomous_derivation(
        task, derivation, auditor, runs_dir=tmp_path, on_event=events.append
    )
    assert result.status == "completed_unverified"
    assert [attempt.substep_id for attempt in result.attempts] == [
        "lagrangian",
        "answer",
        "factor",
        "answer",
    ]
    answer = next(step for step in result.plan if step["id"] == "rules")
    inserted = next(child for child in answer["substeps"] if child["id"] == "factor")
    assert inserted["depends_on"] == ("lagrangian",)
    update = next(event for event in events if event["type"] == "plan.updated")
    assert update["reason"] == "substep_inserted"
    assert any(child["id"] == "factor" for child in update["steps"][0]["substeps"])
    outcome = json.loads((Path(result.run_dir) / "outcome.json").read_text())
    assert "child_statuses" in outcome


def test_child_invalidation_preserves_independent_sibling_checkpoints(tmp_path):
    task = Task("child_invalidation", "Derive dependent scalar results.")
    derivation = ScriptedLLM(
        [
            nested_plan(
                (
                    "rules",
                    "Establish the rule set",
                    [],
                    [
                        {"id": "lagrangian", "goal": "Establish the Lagrangian"},
                        {
                            "id": "sign",
                            "goal": "Fix the interaction sign",
                            "depends_on": ["lagrangian"],
                        },
                        {"id": "aux", "goal": "Derive an independent check"},
                    ],
                ),
                (
                    "final",
                    "Derive the requested result",
                    ["rules"],
                    [{"id": "answer", "goal": "Derive the final expression"}],
                ),
            ),
            "Lagrangian.",
            "Independent auxiliary result.",
            "Initial sign.",
            "Initial final expression.",
            "Corrected sign.",
            "Recomputed final expression.",
        ]
    )
    auditor = ScriptedLLM(
        [
            audit("ACCEPT", "rules"),
            audit("ACCEPT", "rules"),
            audit("ACCEPT", "rules"),
            (
                "VERDICT: REJECT\n"
                "BLOCKING: sign | interaction sign | re-derive it\n"
                "INVALIDATE: rules.sign\n"
            ),
            audit("ACCEPT", "rules"),
            audit("ACCEPT", "final"),
        ]
    )
    result = run_autonomous_derivation(task, derivation, auditor, runs_dir=tmp_path)
    assert result.status == "completed_unverified"
    assert [attempt.substep_id for attempt in result.attempts].count("lagrangian") == 1
    assert [attempt.substep_id for attempt in result.attempts].count("aux") == 1
    assert [attempt.substep_id for attempt in result.attempts].count("sign") == 2
    assert [attempt.substep_id for attempt in result.attempts].count("answer") == 2
    assert result.backtracking[0].invalidated_steps == ["rules.sign", "final.answer"]
    assert result.child_statuses["rules.aux"]["status"] == "accepted"


def test_parent_target_from_nested_audit_retries_parent_prerequisite(tmp_path):
    task = Task("parent_target", "Derive dependent scalar results.")
    derivation = ScriptedLLM(
        [
            nested_plan(
                (
                    "rules",
                    "Establish the rule set",
                    [],
                    [
                        {"id": "lagrangian", "goal": "Establish the Lagrangian"},
                        {
                            "id": "sign",
                            "goal": "Fix the interaction sign",
                            "depends_on": ["lagrangian"],
                        },
                    ],
                )
            ),
            "Initial Lagrangian.",
            "Initial sign.",
            "Corrected Lagrangian.",
            "Corrected sign.",
        ]
    )
    auditor = ScriptedLLM(
        [
            audit("ACCEPT", "rules"),
            (
                "VERDICT: REJECT\n"
                "BLOCKING: convention | sign | re-check the parent derivation\n"
                "INVALIDATE: rules\n"
            ),
            audit("ACCEPT", "rules"),
            audit("ACCEPT", "rules"),
        ]
    )

    result = run_autonomous_derivation(
        task, derivation, auditor, runs_dir=tmp_path, max_substep_retries=1
    )

    assert result.status == "completed_unverified"
    assert [attempt.substep_id for attempt in result.attempts] == [
        "lagrangian",
        "sign",
        "lagrangian",
        "sign",
    ]
    assert result.backtracking[0].target_step_id == "rules"
    assert result.backtracking[0].invalidated_steps == [
        "rules.lagrangian",
        "rules.sign",
    ]


def test_repeated_open_issue_does_not_trigger_plan_replacement(tmp_path):
    task = Task("local_only_retry", "Derive a simple expression.")
    derivation = ScriptedLLM(
        [
            plan(("final", "derive final expression", [])),
            "First proposal.",
            "Second proposal.",
        ]
    )
    auditor = ScriptedLLM(
        [
            "VERDICT: REJECT\nBLOCKING: sign | wrong sign | re-derive\nINVALIDATE: none\n",
            (
                "VERDICT: REJECT\nPREVIOUS: B1 STILL OPEN: still inconsistent\n"
                "BLOCKING: B1 | sign | still wrong | re-derive\nINVALIDATE: none\n"
            ),
        ]
    )
    result = run_autonomous_derivation(
        task, derivation, auditor, runs_dir=tmp_path, max_step_retries=1, max_replans=2
    )
    assert result.status == "retry_exhausted"
    assert len(derivation.calls) == 3  # One planner call; no implicit replanning.
    assert "Checkpoint 'final'/'main' exhausted" in result.error


def test_explicit_rejected_plan_target_replans_once_and_preserves_accepted_child(tmp_path):
    task = Task("explicit_replan", "Derive dependent scalar results.")
    events = []
    initial_plan = nested_plan(
        (
            "rules",
            "Establish rules",
            [],
            [
                {"id": "lagrangian", "goal": "Establish the Lagrangian"},
                {"id": "factor", "goal": "Derive a factor", "depends_on": ["lagrangian"]},
            ],
        )
    )
    revised_plan = nested_plan(
        (
            "rules",
            "Establish rules",
            [],
            [
                {"id": "lagrangian", "goal": "Establish the Lagrangian"},
                {
                    "id": "factor",
                    "goal": "Derive the corrected factor",
                    "depends_on": ["lagrangian"],
                },
            ],
        )
    )
    derivation = ScriptedLLM(
        [
            initial_plan,
            "Lagrangian checkpoint.",
            "Initial factor.",
            revised_plan,
            "Corrected factor.",
        ]
    )
    auditor = ScriptedLLM(
        [
            audit("ACCEPT", "rules"),
            (
                "VERDICT: REJECT\nBLOCKING: factor | wrong factor | re-derive\n"
                "INVALIDATE: PLAN\n"
            ),
            audit("ACCEPT", "rules"),
        ]
    )
    result = run_autonomous_derivation(
        task,
        derivation,
        auditor,
        runs_dir=tmp_path,
        max_replans=1,
        max_substep_retries=1,
        on_event=events.append,
    )
    assert result.status == "completed_unverified"
    assert len(derivation.calls) == 5
    assert [attempt.substep_id for attempt in result.attempts].count("lagrangian") == 1
    assert [attempt.substep_id for attempt in result.attempts].count("factor") == 2
    assert any(
        event["type"] == "plan.updated" and event["reason"] == "replan"
        for event in events
    )
    update = next(
        event
        for event in events
        if event["type"] == "plan.updated" and event["reason"] == "replan"
    )
    assert update["steps"][0]["substeps"][1]["goal"] == "Derive the corrected factor"


def test_snapshot_replace_failure_stops_before_derivation(monkeypatch, tmp_path):
    task = Task(
        "snapshot_failure",
        "Derive a simple expression.",
    )
    derivation = ScriptedLLM(
        [
            nested_plan(
                (
                    "rules",
                    "Derive the result",
                    [],
                    [{"id": "lagrangian", "goal": "Establish the Lagrangian"}],
                )
            ),
            "Must not be requested.",
        ]
    )

    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(autonomous.os, "replace", fail_replace)
    result = run_autonomous_derivation(
        task, derivation, ScriptedLLM([]), runs_dir=tmp_path
    )
    assert result.status == "failed"
    assert result.error_code == "snapshot_write_failed"
    assert len(derivation.calls) == 1
    assert not (Path(result.run_dir) / "checkpoints.json").exists()
    assert json.loads((Path(result.run_dir) / "outcome.json").read_text())["status"] == "failed"


def test_failed_transition_snapshot_keeps_last_committed_graph(monkeypatch, tmp_path):
    task = Task(
        "snapshot_accept_failure",
        "Derive a simple expression.",
    )
    derivation = ScriptedLLM(
        [
            nested_plan(
                (
                    "rules",
                    "Derive the result",
                    [],
                    [{"id": "lagrangian", "goal": "Establish the Lagrangian"}],
                )
            ),
            "Accepted output must not become committed.",
            "Must not be requested after the failed commit.",
        ]
    )
    auditor = ScriptedLLM([audit("ACCEPT", "rules")])
    real_replace = autonomous.os.replace
    replace_count = 0

    def fail_acceptance_replace(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 6:
            raise OSError("injected checkpoint transition replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(autonomous.os, "replace", fail_acceptance_replace)
    result = run_autonomous_derivation(
        task, derivation, auditor, runs_dir=tmp_path
    )
    assert result.status == "failed"
    assert result.error_code == "snapshot_write_failed"
    assert len(derivation.calls) == 2
    assert len(auditor.calls) == 1
    assert not result.accepted_steps
    assert replace_count == 6
    snapshot = json.loads((Path(result.run_dir) / "checkpoints.json").read_text())
    assert snapshot["accepted"] == {}
    assert snapshot["status"] == "running"


def test_synthetic_main_id_collision_does_not_commit_an_insertion(tmp_path):
    task = Task("insert_collision", "Derive a simple expression.")
    derivation = ScriptedLLM(
        [
            plan(("final", "Establish the Lagrangian and derive the result", [])),
            "Initial proposal.",
            "Corrected proposal.",
        ]
    )
    insertion = (
        "VERDICT: REJECT\nBLOCKING: missing | derive an omitted prerequisite\n"
        "INVALIDATE: current\n"
        'INSERT_SUBSTEP: {"id":"main","goal":"Derive the missing prerequisite",'
        '"depends_on":[]}\n'
    )
    auditor = ScriptedLLM([insertion, insertion, insertion, audit("ACCEPT", "final")])
    result = run_autonomous_derivation(
        task, derivation, auditor, runs_dir=tmp_path, max_step_retries=1
    )
    assert result.status == "completed_unverified"
    assert "substeps" not in result.plan[0]
    snapshot = json.loads((Path(result.run_dir) / "checkpoints.json").read_text())
    assert snapshot["insertion_count"] == 0
    assert snapshot["retry_counts"]["final.main"] == 1


def test_unready_queue_terminates_with_scheduler_stalled(monkeypatch, tmp_path):
    task = Task("scheduler_stall", "Derive a simple expression.")
    derivation = ScriptedLLM([plan(("final", "Establish the Lagrangian", []))])
    monkeypatch.setattr(
        autonomous,
        "_node_dependencies",
        lambda _node, _nodes, _steps: ["missing-dependency"],
    )
    result = run_autonomous_derivation(
        task, derivation, ScriptedLLM([]), runs_dir=tmp_path
    )
    assert result.status == "failed"
    assert result.error_code == "scheduler_stalled"
    assert len(derivation.calls) == 1


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
    assert "Derive Feynman rules" in result.error
    assert (Path(result.run_dir) / "plan_raw.md").exists()


def test_first_plan_step_accepts_action_phrasing(tmp_path):
    payload = json.loads(plan(("rules", "derive Feynman rules", [])))
    payload["steps"][0]["goal"] = "Write the action S = int(1/2 (dphi)^2 - c4/4! phi^4) and fix conventions"
    result = execute(tmp_path, [json.dumps(payload)], [])
    assert "first plan step must establish" not in (result.error or "")


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
    assert result.status == "completed_verified", result.error
    assert result.attempts[0].status == "deterministic_rejection"
    assert result.attempts[0].deterministic_checks["passed"] is False
    assert result.attempts[1].deterministic_checks["passed"] is True


def test_deterministic_rejection_recomputes_descendants_only(monkeypatch, tmp_path):
    task = Task.load("phi4_dim4")
    derivation = ScriptedLLM(
        [
            nested_plan(
                (
                    "rules",
                    "Derive the result",
                    [],
                    [
                        {"id": "lagrangian", "goal": "Establish the Lagrangian"},
                        {
                            "id": "formula",
                            "goal": "Derive the final formula",
                            "depends_on": ["lagrangian"],
                        },
                        {
                            "id": "dependent",
                            "goal": "Check the formula",
                            "depends_on": ["formula"],
                        },
                        {"id": "aux", "goal": "Derive an independent check"},
                    ],
                )
            ),
            "Lagrangian checkpoint.",
            "Independent auxiliary checkpoint.",
            "Initial formula.\n```final\nA_loop = 1\n```",
            "Check based on the initial formula.",
            "Corrected formula.\n```final\nA_loop = 2\n```",
            "Check based on the corrected formula.",
        ]
    )
    auditor = ScriptedLLM([audit("ACCEPT", "rules") for _ in range(6)])
    reports = iter(
        [
            {"passed": False, "failed_names": ["loop_value"]},
            {"passed": True, "failed_names": []},
        ]
    )

    class VerificationReport:
        def __init__(self, value):
            self.passed = value["passed"]
            self.failed_names = value["failed_names"]

        def to_dict(self):
            return {"passed": self.passed, "checks": []}

    monkeypatch.setattr(
        autonomous,
        "_hygiene_problem",
        lambda _proposal, _is_last_step: "",
    )
    monkeypatch.setattr(
        autonomous,
        "verify_final_answer",
        lambda _content: VerificationReport(next(reports)),
    )

    result = run_autonomous_derivation(
        task,
        derivation,
        auditor,
        runs_dir=tmp_path,
        max_substep_retries=1,
    )

    assert result.status == "completed_verified"
    assert [attempt.substep_id for attempt in result.attempts].count("formula") == 2
    assert [attempt.substep_id for attempt in result.attempts].count("dependent") == 2
    assert [attempt.substep_id for attempt in result.attempts].count("lagrangian") == 1
    assert [attempt.substep_id for attempt in result.attempts].count("aux") == 1
    assert result.child_statuses["rules.aux"]["status"] == "accepted"


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
