"""Iterative derive -> audit -> revise loop with deterministic verification.

Roles:
* derivation agent: does the physics and revises its own work from feedback;
* audit agent: names the earliest faulty step and gives a targeted hint; it must NOT
  supply a full solution;
* deterministic layer (``verification``): checks the final answer after every round. Its
  verdict is recorded for measurement and is never shown to the derivation agent, so the
  measured improvement comes from the audit feedback alone.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .derivation import DERIVATION_SYSTEM
from .llm import LLMError
from .verification import verify_final_answer

FINAL_BLOCK_INSTRUCTION = (
    "End your answer with a fenced block labelled final containing exactly three lines, each "
    "a plain arithmetic expression using only c4, s, t, u, mu2, pi, log, numbers and + - * / **:\n"
    "```final\nA_tree = ...\nA_loop = ...   (the order-c4^2 term of A, finite MS-bar part)\n"
    "beta_c4 = ...   (mu dc4/dmu at one loop)\n```"
)

REVISE_SYSTEM = DERIVATION_SYSTEM + (
    " You may be given reviewer feedback. Treat it as fallible: re-derive the flagged step "
    "from the definitions, fix it if the reviewer is right, and say so if you disagree."
)

AUDIT_SYSTEM = (
    "You are a skeptical quantum field theory reviewer guiding a student. Read the derivation "
    "step by step, find the EARLIEST incorrect or unjustified step, explain the error and give "
    "a targeted hint (which definition, factor, sign or branch to recheck). Do NOT write a "
    "complete corrected solution or the final amplitude. If every step is justified, say so. "
    "No code."
)

AUDIT_REQUEST = (
    "Review the derivation above. Reply in exactly this format:\n"
    "VERDICT: PASS | FAIL | UNCERTAIN\n"
    "FIRST_ERROR: <step number or 'none'>\n"
    "ERROR: <what is wrong, with the decisive equation>\n"
    "HINT: <targeted correction hint; no full solution>\n"
    "Stay under 250 words."
)

_VERDICT = re.compile(r"VERDICT:\s*(PASS|FAIL|UNCERTAIN)", re.IGNORECASE)


def parse_verdict(audit: str) -> str:
    match = _VERDICT.search(audit or "")
    return match.group(1).upper() if match else "UNCERTAIN"


@dataclass
class Round:
    index: int
    audit_verdict: str = ""
    verification_passed: bool = False
    failed_checks: list[str] = field(default_factory=list)


@dataclass
class FeedbackResult:
    task: str
    status: str  # "audit_passed" | "max_rounds" | "failed"
    rounds: list[Round]
    final_verification_passed: bool
    run_dir: str
    error: str = ""


def _ask(llm, system: str, text: str) -> str:
    reply = llm.complete(system, [{"role": "user", "text": text}])
    if not reply.strip():
        raise LLMError("empty response")
    return reply


def run_feedback_loop(
    task, llm, audit_llm, runs_dir="runs", max_rounds=3, echo=None
) -> FeedbackResult:
    say = echo or (lambda message: None)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / f"{stamp}-{task.name}-feedback"
    run_dir.mkdir(parents=True, exist_ok=True)
    rounds: list[Round] = []
    derivation, error, status = "", "", "max_rounds"
    try:
        for index in range(1, max_rounds + 1):
            if index == 1:
                prompt = f"{task.derivation_prompt}\n\n{FINAL_BLOCK_INSTRUCTION}"
            else:
                prompt = (
                    f"{task.derivation_prompt}\n\nYour previous derivation:\n{derivation}\n\n"
                    f"Reviewer feedback:\n{audit}\n\nRevise your own calculation.\n\n"
                    f"{FINAL_BLOCK_INSTRUCTION}"
                )
            derivation = _ask(llm, REVISE_SYSTEM, prompt)
            audit = _ask(
                audit_llm,
                AUDIT_SYSTEM,
                f"Problem:\n{task.derivation_prompt}\n\nDerivation:\n{derivation}\n\n{AUDIT_REQUEST}",
            )
            report = verify_final_answer(derivation)
            record = Round(index, parse_verdict(audit), report.passed, report.failed_names)
            rounds.append(record)
            (run_dir / f"round{index}_derivation.md").write_text(derivation, encoding="utf-8")
            (run_dir / f"round{index}_audit.md").write_text(audit, encoding="utf-8")
            (run_dir / f"round{index}_verification.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
            say(f"round {index}: audit {record.audit_verdict}, verification "
                f"{'PASS' if report.passed else 'FAIL'} {record.failed_checks}")
            if record.audit_verdict == "PASS":
                status = "audit_passed"
                break
    except LLMError as exc:
        error, status = str(exc), "failed"
        say(f"Request failed: {error}")
    result = FeedbackResult(
        task=task.name,
        status=status,
        rounds=rounds,
        final_verification_passed=bool(rounds and rounds[-1].verification_passed),
        run_dir=str(run_dir),
        error=error,
    )
    (run_dir / "outcome.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result
