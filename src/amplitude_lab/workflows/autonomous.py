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
    "derivation. Formulate the theory's explicit Lagrangian from its name and stated "
    "conventions; do not require the user to provide the full Lagrangian. If the theory does "
    "not determine it uniquely, make the ambiguity explicit in assumptions. The first plan "
    "step must establish and justify this Lagrangian so it can be independently audited. "
    "Select a suitable approach rather than forcing a fixed checklist: include "
    "kinematics, Feynman rules, diagrams, symmetry factors, loop integration, regularization, "
    "renormalization, crossing, or other work only when applicable to this problem. Make the "
    "plan dependency-ordered and complete enough to derive and check the requested quantity, "
    "using 5-12 concise steps and keeping the interpretation and strategy to 1-3 sentences each. "
    "Do not solve any step, state expected answers, or use external sources, "
    "web search, or other models. Avoid private chain-of-thought; give only a concise "
    "interpretation and approach summary in Markdown. Delimit inline mathematics with `$...$` "
    "and displayed equations with `$$...$$`. Output only JSON with shape "
    '{"interpretation":"problem and requested quantity","lagrangian":"explicit proposed '
    'Lagrangian and conventions","strategy":"chosen approach",'
    '"assumptions":["explicit assumption"],"steps":[{"id":"short_id",'
    '"goal":"one mathematical deliverable","depends_on":["earlier_id"]}]} . '
    "Use 5-12 steps and an empty assumptions list when none are needed. Make the first step "
    "establish the Lagrangian and a short notation contract: define the denominator-only "
    "integral, specify where mu^{2 eps} and coupling factors live, and state sign/i conventions "
    "for A, Disc, and the i0 prescription. Define every object used by multiple steps in an "
    "earlier step, and make that defining step a dependency of every step that uses the object. "
    "Do not create independent steps that compute the same quantity. Put physics cross-checks "
    "(optical theorem, crossing, mu-independence) in a late dedicated step, not inside "
    "derivation steps. Only the last step produces the final answer block."
)

DERIVATION_SYSTEM = (
    "You are an independent quantum field theory derivation agent. Derive only the requested "
    "mathematical step from the planner's proposed Lagrangian, assumptions, and accepted "
    "dependency checkpoints. For the Lagrangian-establishing step, independently derive and "
    "justify the Lagrangian from the named theory and conventions; flag ambiguity rather than "
    "silently accepting an unsupported choice. Do not assume a target amplitude or quote an "
    "unsupported remembered result. You have no web search, external solutions, or other LLM "
    "assistance. Be concise but checkable: state definitions needed, show essential intermediate "
    "equations, explain non-obvious signs or factors, then give the result and a dimensional or "
    "limiting-case sanity check. Full physics cross-checks belong to a later step. Avoid repeating "
    "the problem or adding lengthy prose; expand only enough to "
    "verify nontrivial algebra and do not skip essential steps. Use clear Markdown and delimit "
    "inline mathematics with `$...$` and displayed equations with `$$...$$`. Do not provide "
    "private chain-of-thought. Always obey the notation contract in the accepted checkpoints; "
    "never redefine a symbol they define. Compute scratch work silently and never write "
    "retractions ('wait', 'let me redo'); show only the clean chain. End with a `checkpoint` "
    "block listing this step's key results; emit a `final` block only if the task says this is "
    "the last step. REVISION MODE: if you are given a previous proposal and audit issues, treat "
    "the issues as the work list. Re-derive each flagged item from scratch, independently, "
    "without assuming the auditor is right or wrong. Items not flagged stay unchanged unless "
    "the re-derivation forces a change; if it does, say so in one line. Output the full clean "
    "step, then a short 'Revision notes' section: one line per issue id saying what you "
    "re-derived and whether the result changed."
)

AUDIT_SYSTEM = (
    "You are a skeptical but scoped quantum field theory auditor. Audit only this step's stated "
    "goal and deliverables against the accepted checkpoints. You may recompute key quantities "
    "privately to verify them, but never supply a corrected formula; point to the exact equation, "
    "say what is inconsistent, and say what to re-derive. Classify each issue as BLOCKING (changes "
    "a result, a sign, a factor of i, a coefficient, a power of mu, violates the stated goal or the "
    "notation contract, or contains an unresolved retraction) or MINOR (presentation or optional "
    "improvement). Do not raise issues that belong to other steps; if an accepted checkpoint is "
    "wrong, report it separately with its step ID. You are given your previous issues: first "
    "state, for each, RESOLVED or STILL OPEN with evidence. Do not introduce new blocking "
    "issues on parts you already passed unless a change made them wrong. ACCEPT if there are no "
    "blocking issues. Output exactly:\n"
    "VERDICT: ACCEPT|REJECT\n"
    "PREVIOUS: <id>: RESOLVED|OPEN: <evidence>   (one per line, or 'none')\n"
    "BLOCKING: <id> | <equation/location> | <what is inconsistent> | <what to re-derive>   (one per line, or 'none')\n"
    "MINOR: <id> | <note>   (one per line, or 'none')\n"
    "INVALIDATE: <accepted step id or 'none'>\n"
    "Write math in plain text without backslashes."
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
    lagrangian: str = ""
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


def _plan_overview(text: str) -> tuple[str, str, str, list[str]]:
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise PlanError(f"derivation plan is not valid JSON: {exc}") from None
    interpretation = payload.get("interpretation", "")
    lagrangian = payload.get("lagrangian", "")
    strategy = payload.get("strategy", "")
    assumptions = payload.get("assumptions", [])
    if not isinstance(interpretation, str) or len(interpretation) > 2000:
        raise PlanError("plan interpretation must be text of at most 2000 characters")
    if not isinstance(lagrangian, str) or not lagrangian.strip() or len(lagrangian) > 4000:
        raise PlanError("plan must provide an explicit Lagrangian of at most 4000 characters")
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
    return (
        interpretation.strip(),
        lagrangian.strip(),
        strategy.strip(),
        [item.strip() for item in assumptions],
    )


def _audit_response(text: str, valid_targets: set[str], current_id: str) -> tuple[str, str, str]:
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if payload is None:
        return _structured_audit_response(text, valid_targets, current_id)
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


def _structured_audit_response(
    text: str, valid_targets: set[str], current_id: str
) -> tuple[str, str, str]:
    verdict_match = re.search(r"(?m)^VERDICT:\s*(ACCEPT|REJECT)\s*$", text)
    invalidate_match = re.search(r"(?m)^INVALIDATE:\s*(\S+)\s*$", text)
    if verdict_match is None or invalidate_match is None:
        return "UNCERTAIN", current_id, "Audit response did not match the required format."

    verdict = verdict_match.group(1)
    target = invalidate_match.group(1)
    if target.casefold() == "none":
        target = current_id
    elif target not in valid_targets:
        return (
            "UNCERTAIN",
            current_id,
            "Audit invalidation target was not the current step or an accepted checkpoint.",
        )
    if verdict == "ACCEPT" and target != current_id:
        return (
            "UNCERTAIN",
            current_id,
            "Audit accepted the current step but targeted an earlier checkpoint.",
        )

    blocking = re.findall(r"(?m)^BLOCKING:\s*(?!none\s*$)(.+)$", text)
    if verdict == "ACCEPT" and blocking:
        return "UNCERTAIN", current_id, "Audit accepted the step while reporting blocking issues."
    if verdict == "REJECT" and not blocking:
        return "UNCERTAIN", current_id, "Audit rejected the step without reporting a blocking issue."

    previous_open = re.findall(r"(?m)^PREVIOUS:\s*(.+)$", text)
    previous_open = [
        issue for issue in previous_open if re.search(r":\s*(?:STILL OPEN|OPEN)\s*:", issue)
    ]
    minor = re.findall(r"(?m)^MINOR:\s*(?!none\s*$)(.+)$", text)
    diagnostics = blocking + previous_open + minor
    if not diagnostics:
        diagnostics = ["Audit accepted the step with no blocking issues."]
    return verdict, target, "\n".join(diagnostics)


def _format_audit_response(text: str) -> str:
    if not text.strip():
        return "No complete audit response was recorded."
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError):
        return text

    sections = []
    verdict = payload.get("verdict")
    target = payload.get("target_step_id")
    diagnostic = payload.get("diagnostic")
    if isinstance(verdict, str):
        sections.append(f"**Verdict:** {verdict}")
    if isinstance(target, str):
        sections.append(f"**Review target:** `{target}`")
    if isinstance(diagnostic, str):
        sections.extend(["", "**Auditor's explanation:**", "", diagnostic or "No explanation supplied."])
    if not sections:
        return text
    return "\n".join(sections)


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
    def __init__(self, on_event: Callable[[dict], None] | None):
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


def _ancestors(step_id: str, step_by_id: dict[str, DerivationStep]) -> set[str]:
    ancestors = set()
    pending = list(step_by_id[step_id].depends_on)
    while pending:
        dependency = pending.pop()
        if dependency not in ancestors:
            ancestors.add(dependency)
            pending.extend(step_by_id[dependency].depends_on)
    return ancestors


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
        "## Planner-proposed Lagrangian",
        "",
        result.lagrangian or "Not supplied by the planner.",
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
                _format_audit_response(attempt.audit_response),
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
        "## Planner-proposed Lagrangian",
        "",
        result.lagrangian or "Not supplied by the planner.",
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
            sections.extend(["### AI audit", "", _format_audit_response(row["audit"]), ""])
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
                    _format_audit_response(attempt.audit_response),
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


def _render_response(result: AutonomousResult) -> str:
    sections = [
        f"# Derivation response: {result.task}",
        "",
        f"- Status: **{result.status}**",
        "",
        "## Problem",
        "",
        result.interpretation or "No problem interpretation was produced.",
        "",
        "## Proposed Lagrangian and conventions",
        "",
        result.lagrangian or "No Lagrangian was produced.",
        "",
        "## Derivation",
        "",
    ]
    if result.accepted_steps:
        sections.extend(
            f"### {row['goal']}\n\n{row['content']}\n"
            for row in result.accepted_steps
        )
    else:
        sections.append("No derivation steps were accepted.")
    sections.extend(["## Result", "", result.final_amplitude or "No final result was accepted."])
    if result.verification:
        passed = result.verification.get("passed")
        status = "passed" if passed is True else "failed" if passed is False else "unavailable"
        sections.extend(["", f"Verification: **{status}**."])
    if result.error:
        sections.extend(["", f"Workflow error: {result.error}"])
    sections.append("")
    return "\n".join(sections)


def _write_markdown_reports(result: AutonomousResult) -> None:
    run_dir = Path(result.run_dir)
    (run_dir / "result.md").write_text(_render_report(result), encoding="utf-8")
    (run_dir / "steps.md").write_text(_render_steps_report(result), encoding="utf-8")
    (run_dir / "response.md").write_text(_render_response(result), encoding="utf-8")


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
    log = _RunLog(on_event)
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
            result.lagrangian,
            result.strategy,
            result.assumptions,
        ) = _plan_overview(plan_text)
        if "lagrangian" not in steps[0].goal.casefold():
            raise PlanError("first plan step must establish and justify the Lagrangian")
        step_by_id = {step.id: step for step in steps}
        pending = [step.id for step in steps]
        result.plan = [asdict(step) for step in steps]
        (run_dir / "plan.json").write_text(json.dumps(result.plan, indent=2), encoding="utf-8")
        _write_markdown_reports(result)
        log.emit(
            "plan.ready",
            "generating",
            interpretation=result.interpretation,
            lagrangian=result.lagrangian,
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
                "problem_context": {
                    "interpretation": result.interpretation,
                    "assumptions": result.assumptions,
                    "strategy": result.strategy,
                },
                "planner_lagrangian": result.lagrangian,
                "current_step": {"id": step.id, "goal": step.goal},
                "accepted_dependencies": dependencies,
                "reviewer_diagnostic_to_recheck_independently": prior_feedback,
            }
            if step_id == steps[0].id:
                prompt["original_problem"] = task.derivation_prompt
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
            relevant_checkpoint_ids = _ancestors(step_id, step_by_id)
            accepted_context = [
                {
                    "step_id": item.id,
                    "goal": item.goal,
                    "content": accepted[item.id]["content"],
                }
                for item in steps
                if item.id in relevant_checkpoint_ids and item.id in accepted
            ]
            audit_prompt = {
                "problem_context": {
                    "interpretation": result.interpretation,
                    "assumptions": result.assumptions,
                    "strategy": result.strategy,
                },
                "planner_lagrangian": result.lagrangian,
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
            valid_targets = relevant_checkpoint_ids | {step_id}
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
