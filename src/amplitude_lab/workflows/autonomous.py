"""Audited, checkpointed derivation with deterministic final-answer verification."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .llm import LLMError
from .verification import verify_final_answer

PLAN_SYSTEM = (
    "You are a research-planning assistant for quantum field theory. Read only the supplied "
    "problem and identify the theory, requested quantity, and assumptions that affect the "
    "derivation. Select a suitable approach rather than forcing a fixed checklist: include "
    "kinematics, Feynman rules, diagrams, symmetry factors, loop integration, regularization, "
    "renormalization, crossing, or other work only when applicable to this problem. Make the "
    "plan dependency-ordered, specific, and complete enough to derive and check the requested "
    "2-to-2 amplitude. Do not solve any step, state expected answers, or use external sources, "
    "web search, or other models. Avoid private chain-of-thought; give only a concise "
    "interpretation and approach summary in Markdown. Delimit inline mathematics with `$...$` "
    "and displayed equations with `$$...$$`. Output only JSON with shape "
    '{"interpretation":"problem and requested quantity","strategy":"chosen approach",'
    '"assumptions":["explicit assumption"],"steps":[{"id":"short_id",'
    '"goal":"one mathematical deliverable","depends_on":["earlier_id"]}]} . '
    "Use 1-30 steps and an empty assumptions list when none are needed."
)

DERIVATION_SYSTEM = (
    "You are an independent quantum field theory derivation agent. Derive only the requested "
    "mathematical step from the supplied Lagrangian, assumptions, and accepted dependency "
    "checkpoints. Do not assume a target amplitude or quote an unsupported remembered result. "
    "You have no web search, external solutions, or other LLM assistance. Show concise, "
    "checkable equations, state conventions and relevant assumptions, and explain how the "
    "result meets this step's goal. Format the response as concise Markdown and delimit inline "
    "mathematics with `$...$` and displayed equations with `$$...$$`; do not provide private "
    "chain-of-thought. Do not silently accept reviewer feedback: independently recompute the "
    "affected step."
)

AUDIT_SYSTEM = (
    "You are a skeptical quantum field theory auditor. Audit the proposed step's mathematics, "
    "physical assumptions, signs, factors of i, coefficients, dimensions, integrations, "
    "whether the proposal actually satisfies its stated goal, consistency with the problem, "
    "and accepted checkpoints. Do not solve the step or give a corrected formula. Return a "
    "targeted diagnostic identifying what to re-derive. If an accepted earlier checkpoint is "
    "wrong, identify its step ID so that it and dependent work can be invalidated. Avoid private "
    "chain-of-thought. Output only JSON with shape "
    '{"verdict":"ACCEPT or REJECT","target_step_id":"step_id",'
    '"diagnostic":"brief evidence and next thing to re-derive"}.'
)

_STEP_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_FINAL_BLOCK = re.compile(r"```final\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class PlanError(ValueError):
    """The model did not return a usable, acyclic derivation plan."""


@dataclass(frozen=True)
class DerivationStep:
    id: str
    goal: str
    depends_on: tuple[str, ...]


@dataclass
class StepAttempt:
    step_id: str
    attempt: int
    proposal: str
    verdict: str
    diagnostic: str
    status: str
    audit_response: str = ""
    deterministic_checks: dict | None = None


@dataclass
class Backtrack:
    target_step_id: str
    invalidated_steps: list[str]
    diagnostic: str


@dataclass
class AutonomousResult:
    task: str
    status: str
    run_dir: str
    derivation_model: str
    audit_model: str
    plan: list[dict] = field(default_factory=list)
    interpretation: str = ""
    strategy: str = ""
    assumptions: list[str] = field(default_factory=list)
    attempts: list[StepAttempt] = field(default_factory=list)
    accepted_steps: list[dict] = field(default_factory=list)
    backtracking: list[Backtrack] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    final_amplitude: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _json_object(text: str) -> dict:
    match = _JSON_FENCE.match(text or "")
    candidate = match.group(1) if match else (text or "").strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def parse_plan(text: str) -> list[DerivationStep]:
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise PlanError(f"derivation plan is not valid JSON: {exc}") from None
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 30:
        raise PlanError("plan must contain between 1 and 30 steps")
    steps = []
    seen = set()
    for row in raw_steps:
        if not isinstance(row, dict):
            raise PlanError("each plan entry must be an object")
        step_id, goal, dependencies = row.get("id"), row.get("goal"), row.get("depends_on", [])
        if not isinstance(step_id, str) or not _STEP_ID.fullmatch(step_id) or step_id in seen:
            raise PlanError(f"invalid or duplicate step ID {step_id!r}")
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 500:
            raise PlanError(f"step {step_id!r} must have a non-empty goal of at most 500 chars")
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(item, str) for item in dependencies)
            or len(dependencies) != len(set(dependencies))
            or any(item not in seen for item in dependencies)
        ):
            raise PlanError(f"step {step_id!r} dependencies must refer to unique earlier steps")
        steps.append(DerivationStep(step_id, goal.strip(), tuple(dependencies)))
        seen.add(step_id)
    return steps


def _plan_overview(text: str) -> tuple[str, str, list[str]]:
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise PlanError(f"derivation plan is not valid JSON: {exc}") from None
    interpretation = payload.get("interpretation", "")
    strategy = payload.get("strategy", "")
    assumptions = payload.get("assumptions", [])
    if not isinstance(interpretation, str) or len(interpretation) > 2000:
        raise PlanError("plan interpretation must be text of at most 2000 characters")
    if not isinstance(strategy, str) or len(strategy) > 2000:
        raise PlanError("plan strategy must be text of at most 2000 characters")
    if (
        not isinstance(assumptions, list)
        or len(assumptions) > 20
        or any(
            not isinstance(item, str) or not item.strip() or len(item) > 500
            for item in assumptions
        )
    ):
        raise PlanError("plan assumptions must be up to 20 non-empty strings of at most 500 chars")
    return interpretation.strip(), strategy.strip(), [item.strip() for item in assumptions]


def _audit_response(text: str, valid_targets: set[str], current_id: str) -> tuple[str, str, str]:
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError):
        return "UNCERTAIN", current_id, "Audit response was not valid JSON; re-derive this step."
    verdict = payload.get("verdict")
    target = payload.get("target_step_id")
    diagnostic = payload.get("diagnostic")
    if verdict not in ("ACCEPT", "REJECT"):
        return "UNCERTAIN", current_id, "Audit verdict was missing or invalid; re-derive this step."
    if not isinstance(target, str) or target not in valid_targets:
        return (
            "UNCERTAIN",
            current_id,
            "Audit target was not the current step or accepted checkpoint; re-derive the step.",
        )
    if verdict == "ACCEPT" and target != current_id:
        return (
            "UNCERTAIN",
            current_id,
            "Audit accepted the current step but targeted an earlier checkpoint; re-derive it.",
        )
    if not isinstance(diagnostic, str) or not diagnostic.strip():
        return "UNCERTAIN", current_id, "Audit omitted its diagnostic; re-derive this step."
    return verdict, target, diagnostic.strip()


def _complete(llm, system, messages, on_token: Callable[[str], None]) -> str:
    stream = getattr(llm, "complete_stream", None)
    if callable(stream):
        return stream(system, messages, on_token)
    response = llm.complete(system, messages)
    if response:
        on_token(response)
    return response


def _token_handler(log, state: str, agent: str, step_id: str | None = None, parts=None):
    collected = parts if parts is not None else []

    def receive(token: str) -> None:
        collected.append(token)
        log.emit(
            "token",
            state,
            agent=agent,
            step_id=step_id,
            text=token,
        )

    return receive


class _RunLog:
    def __init__(self, path: Path, on_event: Callable[[dict], None] | None):
        self.path = path
        self.on_event = on_event
        self.lock = threading.Lock()

    def emit(self, event_type: str, state: str, **data) -> None:
        event = {
            "type": event_type,
            "state": state,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **data,
        }
        with self.lock:
            with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:
                self.on_event = None
                message = f"event stream callback failed: {type(exc).__name__}: {exc}"
                raise LLMError(message) from None


def _descendants(step_id: str, steps: list[DerivationStep]) -> set[str]:
    invalid = {step_id}
    changed = True
    while changed:
        changed = False
        for step in steps:
            if step.id not in invalid and any(dep in invalid for dep in step.depends_on):
                invalid.add(step.id)
                changed = True
    return invalid


def _render_report(result: AutonomousResult) -> str:
    sections = [
        f"# Autonomous derivation: {result.task}",
        "",
        f"- Status: **{result.status}**",
        f"- Derivation model: `{result.derivation_model}`",
        f"- Audit model: `{result.audit_model}`",
        "",
        "## Problem interpretation",
        "",
        result.interpretation or "Not supplied by the planner.",
        "",
        "## Selected approach",
        "",
        result.strategy or "Not supplied by the planner.",
        "",
        "## Planning assumptions",
        "",
        *(
            [f"- {assumption}" for assumption in result.assumptions]
            if result.assumptions
            else ["No additional assumptions listed."]
        ),
        "",
        "## Derivation plan",
        "",
    ]
    for step in result.plan:
        dependencies = ", ".join(step["depends_on"]) or "none"
        sections.append(f"- `{step['id']}`: {step['goal']} (depends on: {dependencies})")
    sections.extend(["", "## Accepted derivation", ""])
    for row in result.accepted_steps:
        sections.extend([f"### {row['step_id']}: {row['goal']}", "", row["content"], ""])
    sections.extend(["## Attempts and audit history", ""])
    for attempt in result.attempts:
        sections.extend(
            [
                f"### {attempt.step_id}, attempt {attempt.attempt}: {attempt.status}",
                "",
                f"Audit verdict: {attempt.verdict}",
                "",
                f"Diagnostic: {attempt.diagnostic}",
                "",
                "Auditor response:",
                "",
                attempt.audit_response or "(no complete audit response)",
                "",
                "Proposal:",
                "",
                attempt.proposal,
                "",
            ]
        )
    sections.extend(["## Backtracking history", ""])
    if result.backtracking:
        for row in result.backtracking:
            sections.append(
                f"- Reopened `{row.target_step_id}`; invalidated "
                f"{', '.join(row.invalidated_steps)}: {row.diagnostic}"
            )
    else:
        sections.append("No backtracking was required.")
    sections.extend(["", "## Deterministic verification", ""])
    sections.append(json.dumps(result.verification, indent=2, ensure_ascii=False))
    sections.extend(["", "## Final amplitude", ""])
    sections.append(result.final_amplitude or "No final amplitude was accepted.")
    if result.error:
        sections.extend(["", "## Workflow error", "", result.error])
    sections.append("")
    return "\n".join(sections)


def _render_steps_report(result: AutonomousResult) -> str:
    accepted = {row["step_id"]: row for row in result.accepted_steps}
    attempts_by_step: dict[str, list[StepAttempt]] = {}
    for attempt in result.attempts:
        attempts_by_step.setdefault(attempt.step_id, []).append(attempt)

    sections = [
        f"# Derivation steps: {result.task}",
        "",
        f"- Run status: **{result.status}**",
        "",
        "## Problem interpretation",
        "",
        result.interpretation or "Not supplied by the planner.",
        "",
        "## Selected approach",
        "",
        result.strategy or "Not supplied by the planner.",
        "",
    ]
    for index, step in enumerate(result.plan, start=1):
        step_id = step["id"]
        row = accepted.get(step_id)
        dependencies = ", ".join(step["depends_on"]) or "none"
        sections.extend(
            [
                f"## Step {index}: {step['goal']}",
                "",
                f"- ID: `{step_id}`",
                f"- Dependencies: {dependencies}",
                f"- Status: **{'accepted' if row else 'not accepted'}**",
                "",
            ]
        )
        if row:
            sections.extend(["### Accepted result", "", row["content"], ""])
            sections.extend(["### AI audit", "", row["audit"] or "No audit response.", ""])
        for attempt in attempts_by_step.get(step_id, []):
            sections.extend(
                [
                    f"### Attempt {attempt.attempt}: {attempt.status}",
                    "",
                    f"AI audit verdict: **{attempt.verdict}**",
                    "",
                    f"Diagnostic: {attempt.diagnostic or 'None'}",
                    "",
                    "#### Proposed work",
                    "",
                    attempt.proposal or "No proposal was produced.",
                    "",
                    "#### Audit response",
                    "",
                    attempt.audit_response or "No complete audit response.",
                    "",
                ]
            )
            if attempt.deterministic_checks is not None:
                sections.extend(
                    [
                        "#### Deterministic checks",
                        "",
                        "```json",
                        json.dumps(attempt.deterministic_checks, indent=2, ensure_ascii=False),
                        "```",
                        "",
                    ]
                )
    sections.extend(["## Backtracking", ""])
    if result.backtracking:
        for item in result.backtracking:
            sections.append(
                f"- Returned to `{item.target_step_id}`; invalidated "
                f"{', '.join(item.invalidated_steps)}: {item.diagnostic}"
            )
    else:
        sections.append("No backtracking was recorded.")
    sections.extend(["", "## Deterministic verification", ""])
    sections.append(json.dumps(result.verification, indent=2, ensure_ascii=False))
    sections.append("")
    return "\n".join(sections)


def _write_markdown_reports(result: AutonomousResult) -> None:
    run_dir = Path(result.run_dir)
    (run_dir / "result.md").write_text(_render_report(result), encoding="utf-8")
    (run_dir / "steps.md").write_text(_render_steps_report(result), encoding="utf-8")


def run_autonomous_derivation(
    task,
    derivation_llm,
    audit_llm,
    runs_dir="runs",
    max_step_retries=3,
    on_event=None,
) -> AutonomousResult:
    """Plan, derive, audit, checkpoint, backtrack, and verify one step at a time."""
    if max_step_retries < 0:
        raise ValueError("max_step_retries must be non-negative")
    if not task.derivation_prompt.strip():
        raise ValueError(f"task {task.name!r} has no derivation prompt")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / f"{stamp}-{task.name}-agent"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = Path(runs_dir) / f"{stamp}-{task.name}-agent-{suffix}"
    run_dir.mkdir(parents=True)
    log = _RunLog(run_dir, on_event)
    result = AutonomousResult(
        task=task.name,
        status="failed",
        run_dir=str(run_dir),
        derivation_model=getattr(derivation_llm, "name", "unknown"),
        audit_model=getattr(audit_llm, "name", "unknown"),
    )
    (run_dir / "prompt.md").write_text(task.derivation_prompt, encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "task": task.name,
                "workflow": "autonomous_stepwise_audit",
                "derivation_model": result.derivation_model,
                "audit_model": result.audit_model,
                "derivation_client": (
                    derivation_llm.describe()
                    if hasattr(derivation_llm, "describe")
                    else {"name": result.derivation_model}
                ),
                "audit_client": (
                    audit_llm.describe()
                    if hasattr(audit_llm, "describe")
                    else {"name": result.audit_model}
                ),
                "max_step_retries": max_step_retries,
                "streaming": bool(getattr(derivation_llm, "stream", False)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    accepted: dict[str, dict] = {}
    retry_counts: dict[str, int] = {}
    pending: list[str] = []
    step_by_id: dict[str, DerivationStep] = {}

    try:
        log.emit(
            "plan.started",
            "planning",
            message="Derivation agent is interpreting the problem.",
        )
        plan_text = _complete(
            derivation_llm,
            PLAN_SYSTEM,
            [{"role": "user", "text": task.derivation_prompt}],
            _token_handler(log, "planning", "derivation"),
        )
        steps = parse_plan(plan_text)
        (
            result.interpretation,
            result.strategy,
            result.assumptions,
        ) = _plan_overview(plan_text)
        step_by_id = {step.id: step for step in steps}
        pending = [step.id for step in steps]
        result.plan = [asdict(step) for step in steps]
        (run_dir / "plan.json").write_text(json.dumps(result.plan, indent=2), encoding="utf-8")
        _write_markdown_reports(result)
        log.emit(
            "plan.ready",
            "generating",
            interpretation=result.interpretation,
            strategy=result.strategy,
            assumptions=result.assumptions,
            steps=result.plan,
            message="Derivation plan is ready.",
        )

        while pending:
            step_id = pending.pop(0)
            step = step_by_id[step_id]
            unmet = [dependency for dependency in step.depends_on if dependency not in accepted]
            if unmet:
                pending.insert(0, step_id)
                continue
            attempt_number = sum(row.step_id == step_id for row in result.attempts) + 1
            prior_feedback = accepted.get(step_id, {}).get("feedback", "")
            dependencies = [
                {
                    "step_id": dependency,
                    "goal": accepted[dependency]["goal"],
                    "content": accepted[dependency]["content"],
                }
                for dependency in step.depends_on
            ]
            prompt = {
                "problem": task.derivation_prompt,
                "current_step": {"id": step.id, "goal": step.goal},
                "accepted_dependencies": dependencies,
                "reviewer_diagnostic_to_recheck_independently": prior_feedback,
            }
            log.emit(
                "step.generating",
                "generating",
                step_id=step_id,
                attempt=attempt_number,
                goal=step.goal,
            )
            proposal_parts = []
            try:
                proposal = _complete(
                    derivation_llm,
                    DERIVATION_SYSTEM,
                    [{"role": "user", "text": json.dumps(prompt, ensure_ascii=False)}],
                    _token_handler(log, "generating", "derivation", step_id, proposal_parts),
                )
            except LLMError as exc:
                result.attempts.append(
                    StepAttempt(
                        step_id,
                        attempt_number,
                        "".join(proposal_parts),
                        "UNAVAILABLE",
                        str(exc),
                        "generation_failed",
                    )
                )
                raise
            if not proposal.strip():
                result.attempts.append(
                    StepAttempt(
                        step_id,
                        attempt_number,
                        proposal,
                        "UNCERTAIN",
                        "Derivation agent returned an empty step.",
                        "generation_failed",
                    )
                )
                raise LLMError(f"Derivation agent returned an empty step for {step_id!r}.")
            log.emit("step.awaiting_audit", "awaiting_audit", step_id=step_id)
            accepted_context = [
                {
                    "step_id": item.id,
                    "goal": item.goal,
                    "content": accepted[item.id]["content"],
                }
                for item in steps
                if item.id in accepted and "content" in accepted[item.id]
            ]
            audit_prompt = {
                "problem": task.derivation_prompt,
                "current_step": {"id": step.id, "goal": step.goal, "proposal": proposal},
                "accepted_checkpoints": accepted_context,
                "instruction": (
                    "Audit only; do not solve. target_step_id must be the current step or an "
                    "accepted checkpoint that should be invalidated."
                ),
            }
            audit_parts = []
            try:
                audit_response = _complete(
                    audit_llm,
                    AUDIT_SYSTEM,
                    [{"role": "user", "text": json.dumps(audit_prompt, ensure_ascii=False)}],
                    _token_handler(log, "awaiting_audit", "audit", step_id, audit_parts),
                )
            except LLMError as exc:
                result.attempts.append(
                    StepAttempt(
                        step_id,
                        attempt_number,
                        proposal,
                        "UNAVAILABLE",
                        str(exc),
                        "audit_failed",
                        "".join(audit_parts),
                    )
                )
                raise
            if not audit_response.strip():
                result.attempts.append(
                    StepAttempt(
                        step_id,
                        attempt_number,
                        proposal,
                        "UNCERTAIN",
                        "Audit agent returned an empty response.",
                        "audit_failed",
                    )
                )
                raise LLMError(f"Audit agent returned an empty response for {step_id!r}.")
            valid_targets = {
                key for key, checkpoint in accepted.items() if "content" in checkpoint
            } | {step_id}
            verdict, target_id, diagnostic = _audit_response(audit_response, valid_targets, step_id)
            log.emit(
                "step.audit",
                "accepted" if verdict == "ACCEPT" else "rejected",
                step_id=step_id,
                verdict=verdict,
                target_step_id=target_id,
                diagnostic=diagnostic,
                response=audit_response,
            )
            attempt = StepAttempt(
                step_id=step_id,
                attempt=attempt_number,
                proposal=proposal,
                verdict=verdict,
                diagnostic=diagnostic,
                status="accepted" if verdict == "ACCEPT" else "rejected",
                audit_response=audit_response,
            )
            result.attempts.append(attempt)
            _write_markdown_reports(result)

            if target_id != step_id:
                affected = _descendants(target_id, steps)
                invalidated = affected & set(accepted)
                for invalid_id in invalidated:
                    accepted.pop(invalid_id, None)
                    if invalid_id != target_id:
                        retry_counts[invalid_id] = 0
                accepted[target_id] = {
                    **accepted.get(target_id, {}),
                    "feedback": diagnostic,
                }
                result.backtracking.append(
                    Backtrack(
                        target_id,
                        [item.id for item in steps if item.id in affected],
                        diagnostic,
                    )
                )
                attempt.status = "backtrack_triggered"
                log.emit(
                    "backtracking.started",
                    "revising",
                    target_step_id=target_id,
                    invalidated_steps=sorted(invalidated),
                    diagnostic=diagnostic,
                )
                result.accepted_steps = [
                    {
                        "step_id": item.id,
                        "goal": item.goal,
                        "content": accepted[item.id]["content"],
                        "audit": accepted[item.id]["audit"],
                        "attempt": accepted[item.id]["attempt"],
                    }
                    for item in steps
                    if item.id in accepted and "content" in accepted[item.id]
                ]
                _write_markdown_reports(result)
                if retry_counts.get(target_id, 0) >= max_step_retries:
                    result.status = "retry_exhausted"
                    result.error = (
                        f"Checkpoint {target_id!r} exhausted its correction attempts "
                        "after an audit backtrack."
                    )
                    break
                retry_counts[target_id] = retry_counts.get(target_id, 0) + 1
                pending = [item.id for item in steps if item.id in affected] + [
                    queued for queued in pending if queued not in affected
                ]
                continue

            if verdict != "ACCEPT":
                log.emit(
                    "step.rejected",
                    "rejected",
                    step_id=step_id,
                    attempt=attempt_number,
                    diagnostic=diagnostic,
                    audit_verdict=verdict,
                )
                if retry_counts.get(step_id, 0) >= max_step_retries:
                    result.status = "retry_exhausted"
                    result.error = (
                        f"Step {step_id!r} was not accepted after {attempt_number} attempts."
                    )
                    break
                retry_counts[step_id] = retry_counts.get(step_id, 0) + 1
                accepted[step_id] = {"feedback": diagnostic}
                pending.insert(0, step_id)
                log.emit(
                    "step.revising",
                    "revising",
                    step_id=step_id,
                    retries_used=retry_counts[step_id],
                    max_step_retries=max_step_retries,
                )
                _write_markdown_reports(result)
                continue

            accepted[step_id] = {
                "goal": step.goal,
                "content": proposal,
                "feedback": "",
                "attempt": attempt_number,
                "audit": audit_response,
            }
            result.accepted_steps = [
                {
                    "step_id": item.id,
                    "goal": item.goal,
                    "content": accepted[item.id]["content"],
                    "audit": accepted[item.id]["audit"],
                    "attempt": accepted[item.id]["attempt"],
                }
                for item in steps
                if item.id in accepted and "content" in accepted[item.id]
            ]
            log.emit(
                "step.accepted",
                "accepted",
                step_id=step_id,
                attempt=attempt_number,
                content=proposal,
                audit=audit_response,
            )
            _write_markdown_reports(result)

            if step_id == steps[-1].id and task.name == "phi4_dim4":
                log.emit("verification.started", "verifying", step_id=step_id)
                report = verify_final_answer(
                    "\n\n".join(row["content"] for row in result.accepted_steps)
                )
                attempt.deterministic_checks = report.to_dict()
                result.verification = {
                    "available": True,
                    "passed": report.passed,
                    "kind": "deterministic",
                    "checks": report.to_dict(),
                }
                log.emit(
                    "verification.finished",
                    "verifying",
                    passed=report.passed,
                    checks=report.to_dict(),
                )
                if not report.passed:
                    attempt.status = "deterministic_rejection"
                    diagnostic = "Deterministic checks failed: " + ", ".join(report.failed_names)
                    accepted.pop(step_id, None)
                    result.accepted_steps = [
                        row for row in result.accepted_steps if row["step_id"] != step_id
                    ]
                    if retry_counts.get(step_id, 0) >= max_step_retries:
                        result.status = "retry_exhausted"
                        result.error = f"Final step {step_id!r} failed deterministic verification."
                        break
                    retry_counts[step_id] = retry_counts.get(step_id, 0) + 1
                    accepted[step_id] = {"feedback": diagnostic}
                    pending.insert(0, step_id)
                    log.emit(
                        "step.rejected",
                        "rejected",
                        step_id=step_id,
                        attempt=attempt_number,
                        diagnostic=diagnostic,
                        audit_verdict="ACCEPT",
                        deterministic_failure=True,
                    )
                    continue
                result.final_amplitude = _FINAL_BLOCK.findall(proposal)[-1]

        if not pending and result.status == "failed":
            result.status = (
                "completed_verified"
                if result.verification.get("passed")
                else ("completed_unverified")
            )
            if not result.verification:
                result.verification = {
                    "available": False,
                    "passed": None,
                    "kind": "not_configured_for_task",
                    "message": (
                        "No task-specific deterministic reference verifier is configured; "
                        "only the independent AI audit completed."
                    ),
                }
            if not result.final_amplitude and result.accepted_steps:
                result.final_amplitude = result.accepted_steps[-1]["content"]
            log.emit(
                "derivation.completed",
                result.status,
                run_dir=str(run_dir),
                final_amplitude=result.final_amplitude,
                verification=result.verification,
            )
        elif result.status == "retry_exhausted":
            log.emit(
                "derivation.completed",
                "retry_exhausted",
                run_dir=str(run_dir),
                error=result.error,
                accepted_steps=result.accepted_steps,
            )
    except (LLMError, OSError, PlanError, ValueError) as exc:
        result.status = "failed"
        result.error = str(exc)
        log.emit("derivation.failed", "failed", run_dir=str(run_dir), error=result.error)
    finally:
        result.accepted_steps = [
            {
                "step_id": item.id,
                "goal": item.goal,
                "content": accepted[item.id]["content"],
                "audit": accepted[item.id]["audit"],
                "attempt": accepted[item.id]["attempt"],
            }
            for item in step_by_id.values()
            if item.id in accepted and "content" in accepted[item.id]
        ]
        (run_dir / "outcome.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_markdown_reports(result)
    return result
