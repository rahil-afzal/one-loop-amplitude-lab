import json
from pathlib import Path

from amplitude_lab.feedback import parse_verdict, run_feedback_loop
from amplitude_lab.llm import ScriptedLLM
from amplitude_lab.task import Task

GOOD = "c4**2/(32*pi**2) * (6 - log(-s/mu2) - log(-t/mu2) - log(-u/mu2))"
BAD = GOOD.replace("6 -", "3 -")


def derivation(loop, beta="-3*c4**2/(16*pi**2)"):
    return f"work\n```final\nA_tree = c4\nA_loop = {loop}\nbeta_c4 = {beta}\n```"


FAIL_AUDIT = "VERDICT: FAIL\nFIRST_ERROR: 5\nERROR: constant\nHINT: recheck Gamma expansion"
PASS_AUDIT = "VERDICT: PASS\nFIRST_ERROR: none\nERROR: none\nHINT: none"


def test_parse_verdict():
    assert parse_verdict(FAIL_AUDIT) == "FAIL"
    assert parse_verdict("garbage") == "UNCERTAIN"


def test_loop_revises_after_feedback_and_records_verification(tmp_path):
    llm = ScriptedLLM([derivation(BAD), derivation(GOOD)])
    auditor = ScriptedLLM([FAIL_AUDIT, PASS_AUDIT])
    result = run_feedback_loop(Task.load("phi4_dim4"), llm, auditor, runs_dir=tmp_path)

    assert result.status == "audit_passed"
    assert [r.verification_passed for r in result.rounds] == [False, True]
    assert result.rounds[0].failed_checks[0] == "loop_physical_region"
    assert result.final_verification_passed
    # the revision prompt carries the audit hint but not the deterministic verdict
    revision_prompt = llm.calls[1][0]["text"]
    assert "recheck Gamma expansion" in revision_prompt
    assert "loop_physical_region" not in revision_prompt
    # the audit prompt must not ask for or contain a reference solution
    assert "no full solution" in auditor.calls[0][0]["text"]
    run_dir = Path(result.run_dir)
    assert json.loads((run_dir / "outcome.json").read_text())["status"] == "audit_passed"
    assert (run_dir / "round2_verification.json").exists()


def test_audit_pass_does_not_override_failed_verification(tmp_path):
    llm = ScriptedLLM([derivation(BAD)])
    auditor = ScriptedLLM([PASS_AUDIT])
    result = run_feedback_loop(Task.load("phi4_dim4"), llm, auditor, runs_dir=tmp_path)
    assert result.status == "audit_passed"
    assert not result.final_verification_passed


def test_max_rounds_and_llm_failure(tmp_path):
    llm = ScriptedLLM([derivation(BAD), derivation(BAD)])
    auditor = ScriptedLLM([FAIL_AUDIT, FAIL_AUDIT])
    result = run_feedback_loop(
        Task.load("phi4_dim4"), llm, auditor, runs_dir=tmp_path, max_rounds=2
    )
    assert result.status == "max_rounds" and len(result.rounds) == 2

    failed = run_feedback_loop(
        Task.load("phi4_dim4"), ScriptedLLM([]), ScriptedLLM([]), runs_dir=tmp_path
    )
    assert failed.status == "failed" and failed.error
