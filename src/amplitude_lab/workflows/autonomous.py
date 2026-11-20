"""Audited, checkpointed derivation with deterministic final-answer verification."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .audit_format import IssueLedger, ParsedAudit, decide, parse_audit
from .llm import LLMError
from .verification import verify_final_answer

_LOGGER = logging.getLogger(__name__)

PLAN_SYSTEM = (
    "You are a research-planning assistant for quantum field theory. Read only the supplied "
    "problem and identify the theory, requested quantity, and assumptions that affect the "
    "derivation. Formulate the theory's explicit Lagrangian from its name and stated "
    "conventions; do not require the user to provide the full Lagrangian. If the theory does "
    "not determine it uniquely, make the ambiguity explicit in assumptions. The first plan "
    "step must establish and justify this Lagrangian so it can be independently audited. "
    "Break parent steps into small auditable substeps and include a non-empty `substeps` array "
    "on each parent; child dependencies are local to that parent. "
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
    '"goal":"one mathematical deliverable","depends_on":["earlier_id"],'
    '"substeps":[{"id":"child_id","goal":"one bounded deliverable",'
    '"depends_on":["earlier_child_id"]}]}]} . '
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
    "re-derived and whether the result changed. In REVISION MODE follow the user message exactly: "
    "patch only flagged items, copy the rest verbatim, end with Revision notes."
)

AUDIT_SYSTEM = (
    "You are a skeptical but scoped quantum field theory auditor. Audit only this step's stated "
    "goal and deliverables against the accepted checkpoints. You may recompute key quantities "
    "privately to verify them, but never supply a corrected formula; point to the exact equation, "
    "say what is inconsistent, and say what to re-derive. Classify each issue as BLOCKING (changes "
    "a result, a sign, a factor of i, a coefficient, or a power of mu, "
    "violating the stated goal or the "
    "notation contract, or contains an unresolved retraction) or MINOR (presentation or optional "
    "improvement). Do not raise issues that belong to other steps; if an accepted checkpoint is "
    "wrong, report it separately with its step ID. You are given your previous issues: first "
    "state, for each, RESOLVED or STILL OPEN with evidence. Do not introduce new blocking "
    "issues on parts you already passed unless a change made them wrong. ACCEPT if there are no "
    "blocking issues. Output exactly:\n"
    "VERDICT: ACCEPT|REJECT\n"
    "PREVIOUS: <id>: RESOLVED|OPEN: <evidence>   (one per line, or 'none')\n"
    "BLOCKING: <id> | <equation/location> | <what is inconsistent> | "
    "<what to re-derive>   (one per line, or 'none')\n"
    "MINOR: <id> | <note>   (one per line, or 'none')\n"
    "INVALIDATE: <current|accepted step id|parent step id|none>\n"
    "When auditing a nested substep, use 'current' for a local rejection. Naming "
    "its parent step id on a rejection invalidates that parent's full substep subtree "
    "and dependent checkpoints; name a specific child checkpoint to invalidate only "
    "that child and its dependents.\n"
    "Write math in plain text without backslashes."
)

_STEP_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_FINAL_BLOCK = re.compile(r"```final\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CHECKPOINT_BLOCK = re.compile(r"```checkpoint\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_RETRACTION = re.compile(r"\b(wait,|let me redo|let me recompute|actually,)", re.IGNORECASE)
_BLOCKING_ID = re.compile(r"(?m)^BLOCKING:\s*([^|\s]+)\s*\|", re.IGNORECASE)
_PREVIOUS_LINE = re.compile(r"(?im)^PREVIOUS:\s*([^:\s]+)\s*:\s*(RESOLVED|STILL OPEN|OPEN)\b")
_MINOR_LINE = re.compile(r"(?im)^MINOR:\s*(?!none\s*$)(.+)$")


class PlanError(ValueError):
    """The model did not return a usable, acyclic derivation plan."""


@dataclass(frozen=True)
class DerivationStep:
    id: str
    goal: str
    depends_on: tuple[str, ...]
    substeps: tuple["DerivationSubstep", ...] = ()
    explicit_substeps: bool = False


@dataclass(frozen=True)
class DerivationSubstep:
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
    substep_id: str | None = None


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
    notation_contract: str = ""
    issue_ledger: dict = field(default_factory=dict)
    open_issues: dict = field(default_factory=dict)
    minor_notes: list[str] = field(default_factory=list)
    error_code: str | None = None
    child_statuses: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        value = asdict(self)
        for attempt in value["attempts"]:
            if attempt.get("substep_id") is None:
                attempt.pop("substep_id", None)
        if self.error_code is None:
            value.pop("error_code", None)
        if not self.child_statuses:
            value.pop("child_statuses", None)
        return value


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
    total_children = 0
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
        explicit_substeps = "substeps" in row
        if explicit_substeps:
            if set(row) != {"id", "goal", "depends_on", "substeps"}:
                raise PlanError(f"step {step_id!r} has unknown fields")
            raw_children = row["substeps"]
            if not isinstance(raw_children, list) or not 1 <= len(raw_children) <= 20:
                raise PlanError(f"step {step_id!r} must contain between 1 and 20 substeps")
            substeps = []
            child_ids = set()
            for child in raw_children:
                if not isinstance(child, dict) or set(child) - {"id", "goal", "depends_on"}:
                    raise PlanError(f"substeps in {step_id!r} must use the exact child schema")
                child_id, child_goal = child.get("id"), child.get("goal")
                child_deps = child.get("depends_on", [])
                if (
                    not isinstance(child_id, str)
                    or not _STEP_ID.fullmatch(child_id)
                    or child_id in child_ids
                ):
                    raise PlanError(f"invalid or duplicate substep ID {child_id!r} in {step_id!r}")
                if (
                    not isinstance(child_goal, str)
                    or not child_goal.strip()
                    or len(child_goal) > 500
                ):
                    raise PlanError(
                        f"substep {child_id!r} must have a non-empty goal of at most 500 chars"
                    )
                if (
                    not isinstance(child_deps, list)
                    or any(not isinstance(item, str) for item in child_deps)
                    or len(child_deps) != len(set(child_deps))
                    or any(item not in child_ids for item in child_deps)
                ):
                    raise PlanError(
                        f"substep {child_id!r} dependencies must refer to unique earlier substeps"
                    )
                substeps.append(DerivationSubstep(child_id, child_goal.strip(), tuple(child_deps)))
                child_ids.add(child_id)
        else:
            # Legacy parent-only entries historically tolerated unrelated extra keys.
            substeps = [DerivationSubstep("main", goal.strip(), ())]
        total_children += len(substeps)
        if total_children > 120:
            raise PlanError("plan must contain at most 120 substeps")
        steps.append(
            DerivationStep(
                step_id,
                goal.strip(),
                tuple(dependencies),
                tuple(substeps),
                explicit_substeps,
            )
        )
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
            not isinstance(item, str) or not item.strip() or len(item) > 500 for item in assumptions
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
        return (
            "UNCERTAIN",
            current_id,
            ("Audit rejected the step without reporting a blocking issue."),
        )

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
        sections.extend(
            [
                "",
                "**Auditor's explanation:**",
                "",
                diagnostic or "No explanation supplied.",
            ]
        )
    if not sections:
        return text
    return "\n".join(sections)


def _hygiene_problem(proposal: str, is_last_step: bool) -> str:
    """Harness-level output checks that run before any audit call."""
    if not is_last_step and _FINAL_BLOCK.search(proposal):
        return "Intermediate step emitted a final block; only the last step may do so."
    match = _RETRACTION.search(proposal)
    if match:
        return (
            f"Step contains a retraction ({match.group(0)!r}); "
            "re-derive and show only the clean chain."
        )
    return ""


def _issue_ids(text: str) -> list[str]:
    return [item.casefold() for item in _BLOCKING_ID.findall(text)]


def _ledger_update(ledger: dict[str, str], audit_text: str) -> None:
    """Apply PREVIOUS lines from a re-audit to an OPEN/RESOLVED ledger."""
    for issue_id, status in _PREVIOUS_LINE.findall(audit_text):
        ledger[issue_id.casefold()] = "RESOLVED" if status.upper() == "RESOLVED" else "OPEN"


def _complete(llm, system, messages, on_token: Callable[[str], None]) -> str:
    stream = getattr(llm, "complete_stream", None)
    if callable(stream):
        return stream(system, messages, on_token)
    response = llm.complete(system, messages)
    if response:
        on_token(response)
    return response


def _token_handler(
    log,
    state: str,
    agent: str,
    step_id: str | None = None,
    parts=None,
    substep_id: str | None = None,
):
    collected = parts if parts is not None else []

    def receive(token: str) -> None:
        collected.append(token)
        log.emit(
            "token",
            state,
            agent=agent,
            step_id=step_id,
            **({"substep_id": substep_id} if substep_id else {}),
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


def _render_report(result: AutonomousResult, accepted_children: dict | None = None) -> str:
    accepted_children = accepted_children or {}
    accepted_parents = {row["step_id"] for row in result.accepted_steps}
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
        for child in step.get("substeps", []):
            child_dependencies = ", ".join(child["depends_on"]) or "none"
            sections.append(
                f"  - `{step['id']}.{child['id']}`: {child['goal']} "
                f"(depends on: {child_dependencies})"
            )
    sections.extend(["", "## Accepted derivation", ""])
    for row in result.accepted_steps:
        sections.extend([f"### {row['step_id']}: {row['goal']}", "", row["content"], ""])
        for child in row.get("substeps", []):
            sections.extend(
                [
                    f"#### {row['step_id']}.{child['substep_id']}: {child['goal']}",
                    "",
                    child["content"],
                    "",
                ]
            )
    for step in result.plan:
        if step["id"] in accepted_parents:
            continue
        partial_children = [
            (child, accepted_children[f"{step['id']}.{child['id']}"])
            for child in step.get("substeps", [])
            if f"{step['id']}.{child['id']}" in accepted_children
        ]
        if not partial_children:
            continue
        sections.extend([f"### {step['id']}: {step['goal']} (parent not accepted)", ""])
        for child, accepted_child in partial_children:
            sections.extend(
                [
                    f"#### {step['id']}.{child['id']}: {child['goal']}",
                    "",
                    accepted_child["content"],
                    "",
                    "Child audit:",
                    "",
                    _format_audit_response(accepted_child["audit"]),
                    "",
                ]
            )
    sections.extend(["## Attempts and audit history", ""])
    for attempt in result.attempts:
        title_id = attempt.step_id
        if attempt.substep_id:
            title_id += f".{attempt.substep_id}"
        sections.extend(
            [
                f"### {title_id}, attempt {attempt.attempt}: {attempt.status}",
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
    if result.child_statuses:
        sections.extend(["", "## Substep progress", ""])
        for node_id, child in result.child_statuses.items():
            sections.append(
                f"- `{node_id}`: **{child['status']}** ({child['attempts']} attempt(s))"
            )
    sections.extend(["", "## Deterministic verification", ""])
    sections.append(json.dumps(result.verification, indent=2, ensure_ascii=False))
    sections.extend(["", "## Final amplitude", ""])
    sections.append(result.final_amplitude or "No final amplitude was accepted.")
    if result.error:
        sections.extend(["", "## Workflow error", "", result.error])
    sections.append("")
    return "\n".join(sections)


def _render_steps_report(result: AutonomousResult, accepted_children: dict | None = None) -> str:
    accepted_children = accepted_children or {}
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
        for child in step.get("substeps", []):
            qualified = f"{step_id}.{child['id']}"
            child_state = result.child_statuses.get(qualified, {})
            sections.extend(
                [
                    f"### Substep {child['id']}: {child['goal']}",
                    "",
                    f"- Status: **{child_state.get('status', 'pending')}**",
                    "",
                ]
            )
            accepted_child = accepted_children.get(qualified)
            if accepted_child is None:
                accepted_child = next(
                    (
                        value
                        for value in (row or {}).get("substeps", [])
                        if value["substep_id"] == child["id"]
                    ),
                    None,
                )
            if accepted_child:
                sections.extend(
                    [
                        accepted_child["content"],
                        "",
                        "#### Child audit",
                        "",
                        _format_audit_response(accepted_child["audit"]),
                        "",
                    ]
                )
        for attempt in attempts_by_step.get(step_id, []):
            attempt_heading = (
                f"### Child {attempt.substep_id} attempt {attempt.attempt}: {attempt.status}"
                if attempt.substep_id
                else f"### Attempt {attempt.attempt}: {attempt.status}"
            )
            sections.extend(
                [
                    attempt_heading,
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
    if result.error:
        sections.extend(["", "## Workflow error", "", result.error])
    sections.append("")
    return "\n".join(sections)


def _render_response(result: AutonomousResult, accepted_children: dict | None = None) -> str:
    accepted_children = accepted_children or {}
    accepted_parents = {row["step_id"] for row in result.accepted_steps}
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
        for row in result.accepted_steps:
            sections.append(f"### {row['goal']}\n\n{row['content']}\n")
            for child in row.get("substeps", []):
                sections.append(
                    f"#### {child['substep_id']}: {child['goal']}\n\n{child['content']}\n"
                )
    else:
        if not accepted_children:
            sections.append("No derivation steps were accepted.")
    for step in result.plan:
        if step["id"] in accepted_parents:
            continue
        partial_children = [
            (child, accepted_children[f"{step['id']}.{child['id']}"])
            for child in step.get("substeps", [])
            if f"{step['id']}.{child['id']}" in accepted_children
        ]
        if partial_children:
            sections.append(f"### {step['goal']} (parent not accepted)\n")
            for child, accepted_child in partial_children:
                sections.append(f"#### {child['goal']}\n\n{accepted_child['content']}\n")
    sections.extend(["## Result", "", result.final_amplitude or "No final result was accepted."])
    if result.verification:
        passed = result.verification.get("passed")
        status = "passed" if passed is True else "failed" if passed is False else "unavailable"
        sections.extend(["", f"Verification: **{status}**."])
    if result.error:
        sections.extend(["", f"Workflow error: {result.error}"])
    sections.append("")
    return "\n".join(sections)


def _render_markdown_reports(
    result: AutonomousResult, accepted_children: dict | None = None
) -> tuple[dict[str, str], bool]:
    renderers = (
        ("result.md", _render_report),
        ("steps.md", _render_steps_report),
        ("response.md", _render_response),
    )
    contents = {}
    succeeded = True
    for filename, renderer in renderers:
        try:
            contents[filename] = renderer(result, accepted_children)
        except Exception:
            succeeded = False
            _LOGGER.warning(
                "Could not render autonomous report %s",
                Path(result.run_dir) / filename,
                exc_info=True,
            )
    return contents, succeeded


def _write_markdown_reports(
    result: AutonomousResult,
    accepted_children: dict | None = None,
    rendered: dict[str, str] | None = None,
) -> bool:
    run_dir = Path(result.run_dir)
    if rendered is None:
        rendered, render_succeeded = _render_markdown_reports(result, accepted_children)
    else:
        render_succeeded = len(rendered) == 3
    succeeded = render_succeeded
    for filename in ("result.md", "steps.md", "response.md"):
        if filename not in rendered:
            continue
        try:
            (run_dir / filename).write_text(rendered[filename], encoding="utf-8")
        except Exception:
            succeeded = False
            _LOGGER.warning(
                "Could not refresh autonomous report %s",
                run_dir / filename,
                exc_info=True,
            )
    return succeeded


class _SnapshotError(OSError):
    def __init__(self, message: str, code: str, replaced: bool = False):
        super().__init__(message)
        self.code = code
        self.replaced = replaced


def _step_dict(step: DerivationStep) -> dict:
    row = {
        "id": step.id,
        "goal": step.goal,
        "depends_on": step.depends_on,
    }
    if step.explicit_substeps:
        row["substeps"] = [
            {"id": child.id, "goal": child.goal, "depends_on": child.depends_on}
            for child in step.substeps
        ]
    return row


def _step_nodes(steps: list[DerivationStep]) -> list[dict]:
    result = []
    for step in steps:
        for child in step.substeps:
            result.append(
                {
                    "key": f"{step.id}.{child.id}",
                    "step_id": step.id,
                    "substep_id": child.id if step.explicit_substeps else None,
                    "goal": child.goal,
                    "local_dependencies": [
                        f"{step.id}.{dependency}" for dependency in child.depends_on
                    ],
                    "parent_dependencies": list(step.depends_on),
                }
            )
    return result


def _node_dependencies(
    node: dict, nodes_by_key: dict[str, dict], steps: list[DerivationStep]
) -> list[str]:
    dependencies = list(node["local_dependencies"])
    for parent_id in node["parent_dependencies"]:
        dependencies.extend(
            item["key"] for item in nodes_by_key.values() if item["step_id"] == parent_id
        )
    return list(dict.fromkeys(dependencies))


def _node_signatures(steps: list[DerivationStep]) -> dict[str, tuple]:
    signatures = {}
    for node in _step_nodes(steps):
        step = next(item for item in steps if item.id == node["step_id"])
        signatures[node["key"]] = (
            node["key"],
            node["goal"],
            tuple(node["local_dependencies"]),
            step.goal,
            tuple(step.depends_on),
        )
    return signatures


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = None
    replaced = False
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=".checkpoints-", suffix=".tmp", dir=str(path.parent)
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        replaced = True
        if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as exc:
        raise _SnapshotError(
            f"checkpoint snapshot persistence failed: {exc}",
            "snapshot_durability_uncertain" if replaced else "snapshot_write_failed",
            replaced=replaced,
        ) from exc
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _audit_payload(text: str, current_key: str) -> ParsedAudit:
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError):
        return parse_audit(text)
    verdict = payload.get("verdict")
    verdict = verdict.upper() if isinstance(verdict, str) else "MISSING"
    diagnostic = payload.get("diagnostic")
    raw_target = payload.get("target_step_id")
    parsed = ParsedAudit(
        verdict=verdict if verdict in ("ACCEPT", "REJECT") else "MISSING",
        blocking=[diagnostic.strip()]
        if verdict == "REJECT" and isinstance(diagnostic, str) and diagnostic.strip()
        else [],
        invalidate=(
            "none"
            if raw_target is None
            or (
                isinstance(raw_target, str)
                and raw_target.casefold() in ("none", "current", current_key.casefold())
            )
            else str(raw_target)
        ),
    )
    if "insert_substep" in payload or "INSERT_SUBSTEP" in payload:
        parsed.format_errors.append("legacy JSON audits do not support INSERT_SUBSTEP")
    if not isinstance(payload.get("diagnostic"), str) or not payload["diagnostic"].strip():
        parsed.format_errors.append("legacy JSON audit omitted its diagnostic")
    return parsed


def _snapshot_state(state: dict) -> dict:
    return {
        "schema_version": 1,
        "status": state["status"],
        "error": state.get("error", ""),
        "error_code": state.get("error_code"),
        "verification": state.get("verification", {}),
        "final_amplitude": state.get("final_amplitude", ""),
        "backtracking": [asdict(item) for item in state.get("backtracking", [])],
        "notation_contract": state.get("notation_contract", ""),
        "plan": [_step_dict(step) for step in state["steps"]],
        "queue": list(state["queue"]),
        "accepted": deepcopy(state["accepted"]),
        "attempts": [_attempt_dict(attempt) for attempt in state["attempts"]],
        "retry_counts": dict(state["retry_counts"]),
        "node_statuses": dict(state["node_statuses"]),
        "retired_ids": sorted(state["retired_ids"]),
        "insertion_count": state["insertion_count"],
        "replans_used": state["replans_used"],
        "issue_ledgers": {
            key: ledger.to_dict() for key, ledger in state.get("ledgers", {}).items()
        },
        "feedback": dict(state.get("feedback", {})),
        "interpretation": state.get("interpretation", ""),
        "lagrangian": state.get("lagrangian", ""),
        "strategy": state.get("strategy", ""),
        "assumptions": list(state.get("assumptions", [])),
        "planner_responses": list(state.get("planner_responses", [])),
    }


def _attempt_dict(attempt: StepAttempt) -> dict:
    row = asdict(attempt)
    if attempt.substep_id is None:
        row.pop("substep_id", None)
    return row


def _aggregate_accepted(steps: list[DerivationStep], accepted: dict) -> list[dict]:
    result = []
    for step in steps:
        child_keys = [f"{step.id}.{child.id}" for child in step.substeps]
        if not child_keys or any(key not in accepted for key in child_keys):
            continue
        rows = [accepted[key] for key in child_keys]
        row = {
            "step_id": step.id,
            "goal": step.goal,
            "content": "\n\n".join(item["content"] for item in rows),
            "audit": "\n\n".join(item["audit"] for item in rows),
            "attempt": rows[-1]["attempt"],
        }
        if step.explicit_substeps:
            row["substeps"] = [
                {
                    "substep_id": child.id,
                    "goal": child.goal,
                    "content": accepted[f"{step.id}.{child.id}"]["content"],
                    "audit": accepted[f"{step.id}.{child.id}"]["audit"],
                    "attempt": accepted[f"{step.id}.{child.id}"]["attempt"],
                }
                for child in step.substeps
            ]
        result.append(row)
    return result


def _display_node_key(key: str, steps: list[DerivationStep]) -> str:
    node = next((item for item in _step_nodes(steps) if item["key"] == key), None)
    if node is not None and node["substep_id"] is None:
        return node["step_id"]
    return key


def _exhaust_node(
    candidate: dict,
    key: str,
    nodes: list[dict],
    dependencies: dict[str, list[str]],
    diagnostic: str,
) -> None:
    affected = {key}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node["key"] not in affected and any(
                dependency in affected for dependency in dependencies.get(node["key"], [])
            ):
                affected.add(node["key"])
                changed = True
    candidate["status"] = "retry_exhausted"
    candidate["node_statuses"][key] = "retry_exhausted"
    for item in affected - {key}:
        candidate["node_statuses"][item] = "blocked"
    candidate["queue"] = [item for item in candidate["queue"] if item not in affected]
    parent, _, child = key.partition(".")
    candidate["error"] = (
        f"Checkpoint {parent!r}/{child!r} exhausted its retry budget; "
        f"last diagnostic: {diagnostic or 'not accepted'}"
    )


def _validate_insertion(
    insertion: dict,
    current: dict,
    state: dict,
    max_insertions: int,
    retry_budget: int,
) -> str:
    step = next(item for item in state["steps"] if item.id == current["step_id"])
    child_name = current["substep_id"] or "main"
    children = list(step.substeps)
    by_id = {child.id: child for child in children}
    requested_id = insertion.get("id")
    goal = insertion.get("goal")
    dependencies = insertion.get("depends_on", [])
    if (
        not isinstance(requested_id, str)
        or not _STEP_ID.fullmatch(requested_id)
        or not isinstance(goal, str)
        or not goal.strip()
        or len(goal.strip()) > 500
        or not isinstance(dependencies, list)
        or any(not isinstance(item, str) for item in dependencies)
        or len(dependencies) != len(set(dependencies))
    ):
        return "INSERT_SUBSTEP has an invalid id, goal, or dependency list."
    inserted_key = f"{step.id}.{requested_id}"
    if inserted_key in state["retired_ids"] or requested_id in by_id:
        return "INSERT_SUBSTEP id collides with a current or retired child."
    if state["insertion_count"] >= max_insertions:
        return "INSERT_SUBSTEP run limit is exhausted."
    if len(children) >= 20 or len(state["retired_ids"]) >= 120:
        return "INSERT_SUBSTEP exceeds the child or lifetime node cap."
    if state["retry_counts"].get(current["key"], 0) >= retry_budget:
        return "The rejected target has no retry budget available for insertion."
    target = by_id.get(child_name)
    if target is None:
        return "INSERT_SUBSTEP target no longer exists."
    if (
        step.id == state["steps"][0].id
        and child_name == step.substeps[0].id
        and not re.search(r"lagrangian|action|\bL\s*=", goal, re.IGNORECASE)
    ):
        return "The first substep must continue to establish the Lagrangian."
    local_ancestors = set()
    pending = list(target.depends_on)
    while pending:
        ancestor = pending.pop()
        if ancestor not in local_ancestors:
            local_ancestors.add(ancestor)
            pending.extend(by_id[ancestor].depends_on)
    if any(
        dependency not in local_ancestors or f"{step.id}.{dependency}" not in state["accepted"]
        for dependency in dependencies
    ):
        return "INSERT_SUBSTEP dependencies must be accepted local ancestors of the target."
    return ""


def run_autonomous_derivation(
    task,
    derivation_llm,
    audit_llm,
    runs_dir="runs",
    max_step_retries=5,
    on_event=None,
    self_consistency_samples=0,
    max_replans=1,
    max_substep_retries=None,
    max_substep_insertions=3,
) -> AutonomousResult:
    """Run a bounded parent/substep DAG, checkpointing each committed transition."""
    for name, value in (
        ("max_step_retries", max_step_retries),
        ("max_replans", max_replans),
        ("max_substep_insertions", max_substep_insertions),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if max_substep_retries is not None and (
        not isinstance(max_substep_retries, int)
        or isinstance(max_substep_retries, bool)
        or max_substep_retries < 0
    ):
        raise ValueError("max_substep_retries must be a non-negative integer")
    if max_substep_retries is None:
        max_substep_retries = max_step_retries
    if not task.derivation_prompt.strip():
        raise ValueError(f"task {task.name!r} has no derivation prompt")
    for client in (derivation_llm, audit_llm):
        if hasattr(client, "timeout"):
            client.timeout = min(float(client.timeout), 119.0)

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
    last_written_reports = None
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
                "max_substep_retries": max_substep_retries,
                "max_substep_insertions": max_substep_insertions,
                "max_replans": max_replans,
                "streaming": bool(getattr(derivation_llm, "stream", False)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state = {
        "status": "running",
        "steps": [],
        "queue": [],
        "accepted": {},
        "attempts": [],
        "retry_counts": {},
        "node_statuses": {},
        "retired_ids": set(),
        "insertion_count": 0,
        "replans_used": 0,
        "ledgers": {},
        "feedback": {},
        "error": "",
        "error_code": None,
        "verification": {},
        "final_amplitude": "",
        "backtracking": [],
        "notation_contract": "",
        "interpretation": "",
        "lagrangian": "",
        "strategy": "",
        "assumptions": [],
        "planner_responses": [],
    }
    initial_snapshot = run_dir / "checkpoints.json"

    def _apply_public_state(candidate: dict) -> None:
        result.status = candidate["status"]
        result.error = candidate.get("error", "")
        result.error_code = candidate.get("error_code")
        result.verification = candidate.get("verification", {})
        result.final_amplitude = candidate.get("final_amplitude", "")
        result.backtracking = candidate.get("backtracking", [])
        result.notation_contract = candidate.get("notation_contract", "")
        result.interpretation = candidate.get("interpretation", "")
        result.lagrangian = candidate.get("lagrangian", "")
        result.strategy = candidate.get("strategy", "")
        result.assumptions = candidate.get("assumptions", [])
        result.plan = [_step_dict(step) for step in candidate["steps"]]
        result.attempts = candidate["attempts"]
        result.accepted_steps = _aggregate_accepted(candidate["steps"], candidate["accepted"])
        if any(step.explicit_substeps for step in candidate["steps"]):
            result.child_statuses = {
                node["key"]: {
                    "parent_step_id": node["step_id"],
                    "substep_id": node["substep_id"] or "main",
                    "status": candidate["node_statuses"].get(node["key"], "pending"),
                    "attempts": sum(
                        item.step_id == node["step_id"] and item.substep_id == node["substep_id"]
                        for item in candidate["attempts"]
                    ),
                }
                for node in _step_nodes(candidate["steps"])
            }
        by_key = {node["key"]: node for node in _step_nodes(candidate["steps"])}
        result.issue_ledger = {
            (
                by_key[key]["step_id"]
                if key in by_key and by_key[key]["substep_id"] is None
                else key
            ): ledger.to_dict()
            for key, ledger in candidate["ledgers"].items()
        }
        result.open_issues = {
            (
                by_key[key]["step_id"]
                if key in by_key and by_key[key]["substep_id"] is None
                else key
            ): {issue.harness_id: issue.text for issue in ledger.open_issues()}
            for key, ledger in candidate["ledgers"].items()
            if ledger.open_issues()
        }
        result.minor_notes = [
            note for ledger in candidate["ledgers"].values() for note in ledger.minor
        ]

    def refresh_reports(force: bool = False) -> None:
        nonlocal last_written_reports
        accepted_children = deepcopy(state["accepted"])
        rendered, render_succeeded = _render_markdown_reports(result, accepted_children)
        projection = tuple(
            rendered.get(filename) for filename in ("result.md", "steps.md", "response.md")
        )
        if not force and projection == last_written_reports:
            return
        if _write_markdown_reports(result, accepted_children, rendered):
            last_written_reports = projection

    def publish(candidate: dict) -> None:
        nonlocal state
        _atomic_checkpoint(initial_snapshot, _snapshot_state(candidate))
        state = candidate
        _apply_public_state(candidate)
        refresh_reports()

    def graph(candidate: dict):
        nodes = _step_nodes(candidate["steps"])
        by_key = {node["key"]: node for node in nodes}
        dependencies = {
            key: _node_dependencies(node, by_key, candidate["steps"])
            for key, node in by_key.items()
        }
        return nodes, by_key, dependencies

    def ready_keys(candidate: dict, candidates=None) -> list[str]:
        nodes, _by_key, dependencies = graph(candidate)
        selected = set(candidates) if candidates is not None else set(candidate["node_statuses"])
        return [
            node["key"]
            for node in nodes
            if node["key"] in selected
            and candidate["node_statuses"].get(node["key"]) not in ("accepted", "blocked")
            and node["key"] not in candidate["queue"]
            and all(dependency in candidate["accepted"] for dependency in dependencies[node["key"]])
        ]

    def enqueue(candidate: dict, keys=None, front=False) -> None:
        newly_ready = ready_keys(candidate, keys)
        if front:
            candidate["queue"] = newly_ready + [
                key for key in candidate["queue"] if key not in newly_ready
            ]
        else:
            candidate["queue"].extend(key for key in newly_ready if key not in candidate["queue"])

    def retry_limit(node: dict) -> int:
        return max_substep_retries if node["substep_id"] is not None else max_step_retries

    def ancestors(node_key: str, dependency_map: dict[str, list[str]]) -> set[str]:
        found = set()
        pending = list(dependency_map.get(node_key, []))
        while pending:
            item = pending.pop()
            if item not in found:
                found.add(item)
                pending.extend(dependency_map.get(item, []))
        return found

    def parent_ancestors(parent_id: str, steps: list[DerivationStep]) -> set[str]:
        by_parent = {step.id: step for step in steps}
        found = set()
        pending = list(by_parent[parent_id].depends_on)
        while pending:
            parent = pending.pop()
            if parent not in found:
                found.add(parent)
                pending.extend(by_parent[parent].depends_on)
        return found

    def authorized_target(
        parsed: ParsedAudit, verdict: str, node: dict, candidate: dict
    ) -> tuple[str, str]:
        raw = parsed.invalidate.strip()
        current = node["key"]
        if raw.casefold() in ("", "none", "current"):
            return "current", ""
        if raw.casefold() == "plan":
            if verdict != "REJECT" or parsed.insert_substep:
                return "invalid", (
                    "PLAN invalidation is only valid with REJECT and without insertion."
                )
            return "plan", ""
        if raw == node["step_id"]:
            if node["substep_id"] is not None and verdict == "REJECT":
                if parsed.insert_substep:
                    return "invalid", "A parent target is valid only with REJECT."
                return f"parent:{raw}", ""
            return "current", ""
        nodes, by_key, dependency_map = graph(candidate)
        if raw in {step.id for step in candidate["steps"]}:
            allowed = parent_ancestors(node["step_id"], candidate["steps"])
            if raw in allowed and all(
                f"{raw}.{child.id}" in candidate["accepted"]
                for step in candidate["steps"]
                if step.id == raw
                for child in step.substeps
            ):
                if verdict != "REJECT" or parsed.insert_substep:
                    return "invalid", "An ancestor target is valid only with REJECT."
                return f"parent:{raw}", ""
            return "invalid", "Parent target is not an accepted strict ancestor."
        if raw in by_key:
            if raw in ancestors(current, dependency_map) and raw in candidate["accepted"]:
                if verdict != "REJECT" or parsed.insert_substep:
                    return "invalid", "A child ancestor target is valid only with REJECT."
                return f"node:{raw}", ""
        return "invalid", "Audit target is unknown, unaccepted, future, or unrelated."

    def accepted_context(node: dict, candidate: dict) -> list[dict]:
        nodes, by_key, dependencies = graph(candidate)
        required = set(dependencies[node["key"]])
        return [
            {
                "step_id": by_key[item["key"]]["step_id"],
                **(
                    {"substep_id": by_key[item["key"]]["substep_id"]}
                    if by_key[item["key"]]["substep_id"]
                    else {}
                ),
                "goal": by_key[item["key"]]["goal"],
                "content": candidate["accepted"][item["key"]]["content"],
            }
            for item in nodes
            if item["key"] in required and item["key"] in candidate["accepted"]
        ]

    def sync_failure(code: str, message: str) -> None:
        result.status = "failed"
        result.error_code = code
        result.error = message
        result.accepted_steps = _aggregate_accepted(state["steps"], state["accepted"])
        result.attempts = state["attempts"]

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
        plan_candidate = deepcopy(state)
        plan_candidate["planner_responses"].append(plan_text)
        publish(plan_candidate)
        (run_dir / "plan_raw.md").write_text(plan_text, encoding="utf-8")
        steps = parse_plan(plan_text)
        (
            result.interpretation,
            result.lagrangian,
            result.strategy,
            result.assumptions,
        ) = _plan_overview(plan_text)
        state["interpretation"] = result.interpretation
        state["lagrangian"] = result.lagrangian
        state["strategy"] = result.strategy
        state["assumptions"] = list(result.assumptions)
        first_goal = steps[0].substeps[0].goal
        if not re.search(r"lagrangian|action|\bL\s*=", first_goal, re.IGNORECASE):
            raise PlanError(
                "first plan step must establish and justify the Lagrangian "
                f"(first substep goal was: {first_goal[:160]!r})"
            )
        state["steps"] = steps
        nodes = _step_nodes(steps)
        state["node_statuses"] = {node["key"]: "pending" for node in nodes}
        state["retired_ids"] = set(state["node_statuses"])
        enqueue(state)
        (run_dir / "plan.json").write_text(
            json.dumps([_step_dict(step) for step in steps], indent=2), encoding="utf-8"
        )
        publish(deepcopy(state))
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
        while state["status"] == "running":
            unfinished = [
                key
                for key, status in state["node_statuses"].items()
                if status not in ("accepted", "blocked")
            ]
            if not unfinished:
                if task.name == "phi4_dim4":
                    content = "\n\n".join(row["content"] for row in result.accepted_steps)
                    log.emit("verification.started", "verifying")
                    report = verify_final_answer(content)
                    checks = report.to_dict()
                    verification = {
                        "available": True,
                        "passed": report.passed,
                        "kind": "deterministic",
                        "checks": checks,
                    }
                    candidate = deepcopy(state)
                    candidate["verification"] = verification
                    if report.passed:
                        if candidate["attempts"]:
                            last_attempt = max(
                                (
                                    index
                                    for index, item in enumerate(candidate["attempts"])
                                    if item.status == "accepted"
                                ),
                                default=None,
                            )
                            if last_attempt is not None:
                                candidate["attempts"][last_attempt].deterministic_checks = checks
                        last_content = (
                            result.accepted_steps[-1]["content"] if result.accepted_steps else ""
                        )
                        matches = _FINAL_BLOCK.findall(last_content)
                        candidate["final_amplitude"] = matches[-1] if matches else last_content
                        publish(candidate)
                        log.emit("verification.finished", "verifying", passed=True, checks=checks)
                        break
                    nodes_now, by_key_now, deps_now = graph(candidate)
                    final_owners = [
                        item["key"]
                        for item in nodes_now
                        if _FINAL_BLOCK.search(
                            candidate["accepted"].get(item["key"], {}).get("content", "")
                        )
                    ]
                    failed_key = final_owners[-1] if final_owners else nodes_now[-1]["key"]
                    failed_node = by_key_now[failed_key]
                    attempt_index = next(
                        index
                        for index in range(len(candidate["attempts"]) - 1, -1, -1)
                        if candidate["attempts"][index].step_id == failed_node["step_id"]
                        and candidate["attempts"][index].substep_id == failed_node["substep_id"]
                    )
                    failed_attempt = candidate["attempts"][attempt_index]
                    failed_attempt.status = "deterministic_rejection"
                    failed_attempt.deterministic_checks = checks
                    diagnostic = "Deterministic checks failed: " + ", ".join(report.failed_names)
                    affected = {failed_key}
                    changed = True
                    while changed:
                        changed = False
                        for item in nodes_now:
                            if item["key"] not in affected and any(
                                dependency in affected for dependency in deps_now[item["key"]]
                            ):
                                affected.add(item["key"])
                                changed = True
                    ordered_affected = [
                        item["key"] for item in nodes_now if item["key"] in affected
                    ]
                    exhausted = [
                        item
                        for item in ordered_affected
                        if candidate["retry_counts"].get(item, 0) >= retry_limit(by_key_now[item])
                    ]
                    if exhausted:
                        for item in ordered_affected:
                            candidate["accepted"].pop(item, None)
                            candidate["node_statuses"][item] = "pending"
                        _exhaust_node(candidate, failed_key, nodes_now, deps_now, diagnostic)
                        publish(candidate)
                        log.emit("verification.finished", "verifying", passed=False, checks=checks)
                        break
                    for item in ordered_affected:
                        candidate["accepted"].pop(item, None)
                        candidate["node_statuses"][item] = "pending"
                        candidate["retry_counts"][item] = candidate["retry_counts"].get(item, 0) + 1
                        candidate["feedback"][item] = diagnostic
                    candidate["queue"] = [failed_key] + [
                        item for item in candidate["queue"] if item not in affected
                    ]
                    candidate["verification"] = verification
                    invalidated = [
                        _display_node_key(item, candidate["steps"]) for item in ordered_affected
                    ]
                    candidate["backtracking"].append(
                        Backtrack(
                            _display_node_key(failed_key, candidate["steps"]),
                            invalidated,
                            diagnostic,
                        )
                    )
                    publish(candidate)
                    log.emit("verification.finished", "verifying", passed=False, checks=checks)
                    log.emit(
                        "backtracking.started",
                        "revising",
                        target_step_id=_display_node_key(failed_key, candidate["steps"]),
                        invalidated_steps=invalidated,
                        diagnostic=diagnostic,
                    )
                    continue
                candidate = deepcopy(state)
                candidate["verification"] = {
                    "available": False,
                    "passed": None,
                    "kind": "not_configured_for_task",
                    "message": (
                        "No task-specific deterministic reference verifier is configured; "
                        "only the independent AI audit completed."
                    ),
                }
                last_content = result.accepted_steps[-1]["content"] if result.accepted_steps else ""
                matches = _FINAL_BLOCK.findall(last_content)
                candidate["final_amplitude"] = matches[-1] if matches else last_content
                publish(candidate)
                break
            if not state["queue"]:
                candidate = deepcopy(state)
                candidate["status"] = "failed"
                candidate["error_code"] = "scheduler_stalled"
                candidate["error"] = (
                    "Checkpoint scheduler stalled with unfinished nodes: " + ", ".join(unfinished)
                )
                publish(candidate)
                break

            key = state["queue"][0]
            nodes, by_key, dependency_map = graph(state)
            node = by_key.get(key)
            if node is None or any(
                dependency not in state["accepted"] for dependency in dependency_map[key]
            ):
                candidate = deepcopy(state)
                candidate["queue"] = [item for item in candidate["queue"] if item != key]
                publish(candidate)
                continue
            candidate = deepcopy(state)
            candidate["queue"].pop(0)
            candidate["node_statuses"][key] = "generating"
            publish(candidate)
            state_node = by_key[key]
            step = next(item for item in state["steps"] if item.id == state_node["step_id"])
            child = next(
                item for item in step.substeps if item.id == (state_node["substep_id"] or "main")
            )
            attempt_number = (
                sum(
                    attempt.step_id == node["step_id"] and attempt.substep_id == node["substep_id"]
                    for attempt in state["attempts"]
                )
                + 1
            )
            dependency_rows = accepted_context(node, state)
            prompt = {
                "problem_context": {
                    "interpretation": result.interpretation,
                    "assumptions": result.assumptions,
                    "strategy": result.strategy,
                },
                "planner_lagrangian": result.lagrangian,
                "current_step": {
                    "id": node["step_id"],
                    "goal": step.goal,
                    "substep_id": node["substep_id"] or "main",
                    "substep_goal": child.goal,
                },
                "accepted_dependencies": dependency_rows,
                "notation_contract": result.notation_contract,
            }
            feedback = state["feedback"].get(key, "")
            if feedback:
                prompt["revision_mode"] = {
                    "previous_proposal": state["accepted"].get(key, {}).get("proposal", ""),
                    "blocking_issues": state["accepted"].get(key, {}).get("blocking_lines", []),
                    "open_issue_ids": sorted(result.open_issues.get(key, {})),
                    "diagnostic": feedback,
                }
            first_node = nodes[0]["key"]
            if key == first_node:
                prompt["original_problem"] = task.derivation_prompt
            last_node = nodes[-1]["key"]
            event_ids = {"step_id": node["step_id"]}
            if node["substep_id"] is not None:
                event_ids["substep_id"] = node["substep_id"]
            log.emit(
                "step.generating",
                "generating",
                **event_ids,
                attempt=attempt_number,
                goal=child.goal,
            )
            proposal_parts = []
            try:
                proposal = _complete(
                    derivation_llm,
                    DERIVATION_SYSTEM,
                    [{"role": "user", "text": json.dumps(prompt, ensure_ascii=False)}],
                    _token_handler(
                        log,
                        "generating",
                        "derivation",
                        node["step_id"],
                        proposal_parts,
                        node["substep_id"],
                    ),
                )
            except LLMError as exc:
                candidate = deepcopy(state)
                candidate["attempts"].append(
                    StepAttempt(
                        node["step_id"],
                        attempt_number,
                        "".join(proposal_parts),
                        "UNAVAILABLE",
                        str(exc),
                        "generation_failed",
                        substep_id=node["substep_id"],
                    )
                )
                candidate["node_statuses"][key] = "generation_failed"
                publish(candidate)
                raise
            if not proposal.strip():
                candidate = deepcopy(state)
                candidate["attempts"].append(
                    StepAttempt(
                        node["step_id"],
                        attempt_number,
                        proposal,
                        "UNCERTAIN",
                        "Derivation agent returned an empty step.",
                        "generation_failed",
                        substep_id=node["substep_id"],
                    )
                )
                candidate["node_statuses"][key] = "generation_failed"
                publish(candidate)
                raise LLMError(f"Derivation agent returned an empty step for {key!r}.")

            attempt = StepAttempt(
                node["step_id"],
                attempt_number,
                proposal,
                "PENDING",
                "",
                "generated",
                substep_id=node["substep_id"],
            )
            candidate = deepcopy(state)
            candidate["attempts"].append(attempt)
            candidate["node_statuses"][key] = "awaiting_audit"
            publish(candidate)
            attempt_index = len(state["attempts"]) - 1
            hygiene = _hygiene_problem(proposal, key == last_node)
            if hygiene:
                attempt = deepcopy(state["attempts"][attempt_index])
                attempt.verdict = "HYGIENE"
                attempt.diagnostic = hygiene
                attempt.status = "hygiene_rejected"
                candidate = deepcopy(state)
                candidate["attempts"][attempt_index] = attempt
                candidate["node_statuses"][key] = "rejected"
                candidate["feedback"][key] = hygiene
                if candidate["retry_counts"].get(key, 0) >= retry_limit(node):
                    _exhaust_node(candidate, key, nodes, dependency_map, hygiene)
                    publish(candidate)
                    log.emit(
                        "step.rejected",
                        "rejected",
                        **event_ids,
                        attempt=attempt_number,
                        diagnostic=hygiene,
                        audit_verdict="HYGIENE",
                    )
                    break
                candidate["retry_counts"][key] = candidate["retry_counts"].get(key, 0) + 1
                candidate["node_statuses"][key] = "pending"
                enqueue(candidate, [key], front=True)
                publish(candidate)
                log.emit(
                    "step.rejected",
                    "rejected",
                    **event_ids,
                    attempt=attempt_number,
                    diagnostic=hygiene,
                    audit_verdict="HYGIENE",
                )
                continue

            log.emit("step.awaiting_audit", "awaiting_audit", **event_ids)
            audit_prompt = {
                "plan": result.plan,
                "problem_context": prompt["problem_context"],
                "planner_lagrangian": result.lagrangian,
                "current_step": {
                    **prompt["current_step"],
                    "proposal": proposal,
                },
                "accepted_checkpoints": dependency_rows,
                "notation_contract": result.notation_contract,
                "previous_blocking_issues": [
                    {
                        "id": issue.harness_id,
                        "status": issue.status,
                        "open_rounds": issue.open_rounds,
                        "text": issue.text,
                    }
                    for issue in state["ledgers"].get(key, IssueLedger()).issues.values()
                ],
                "instruction": (
                    "Audit only; do not solve. Output the line-based format from the system "
                    "prompt. PREVIOUS must cover every previous blocking issue id."
                ),
            }
            if key == first_node:
                audit_prompt["original_problem"] = task.derivation_prompt
            raw_audits = []
            chosen = None
            chosen_target = "current"
            target_diagnostic = ""
            for audit_index in range(3):
                if audit_index:
                    audit_prompt["instruction"] = (
                        "Re-audit the exact same proposal. The previous response was contradictory "
                        "or unparseable. Return the required format without contradictory metadata."
                    )
                audit_parts = []
                try:
                    audit_response = _complete(
                        audit_llm,
                        AUDIT_SYSTEM,
                        [{"role": "user", "text": json.dumps(audit_prompt, ensure_ascii=False)}],
                        _token_handler(
                            log,
                            "awaiting_audit",
                            "audit",
                            node["step_id"],
                            audit_parts,
                            node["substep_id"],
                        ),
                    )
                except LLMError as exc:
                    candidate = deepcopy(state)
                    failed_attempt = deepcopy(candidate["attempts"][attempt_index])
                    failed_attempt.verdict = "UNAVAILABLE"
                    failed_attempt.diagnostic = str(exc)
                    failed_attempt.status = "audit_failed"
                    candidate["attempts"][attempt_index] = failed_attempt
                    candidate["node_statuses"][key] = "audit_failed"
                    publish(candidate)
                    raise
                raw_audits.append(audit_response)
                parsed = _audit_payload(audit_response, key)
                decision = decide(parsed)
                target_kind, target_error = authorized_target(parsed, parsed.verdict, node, state)
                if decision != "CONTRADICTORY" and target_kind != "invalid":
                    if parsed.insert_substep and target_kind != "current":
                        target_error = "Insertion cannot be combined with invalidation."
                        target_kind = "invalid"
                    if parsed.verdict == "ACCEPT" and parsed.insert_substep:
                        target_error = "Insertion is only valid with REJECT."
                        target_kind = "invalid"
                    if parsed.insert_substep and not target_error:
                        target_error = _validate_insertion(
                            parsed.insert_substep,
                            node,
                            state,
                            max_substep_insertions,
                            retry_limit(node),
                        )
                        if target_error:
                            target_kind = "invalid"
                if decision != "CONTRADICTORY" and target_kind != "invalid":
                    chosen = parsed
                    chosen_target = target_kind
                    break
                target_diagnostic = (
                    target_error
                    or "; ".join(parsed.format_errors)
                    or ("Audit response was contradictory or malformed.")
                )
                log.emit(
                    "step.audit_retried",
                    "awaiting_audit",
                    **event_ids,
                    audit_number=audit_index + 1,
                    response=audit_response,
                )
            if chosen is None:
                chosen = ParsedAudit(
                    verdict="REJECT",
                    blocking=[target_diagnostic or "Audit metadata remained contradictory."],
                )
                chosen_target = "current"
            audit_response = "\n\n--- RE-AUDIT ---\n\n".join(raw_audits)
            verdict = chosen.verdict
            diagnostic = "\n".join(chosen.blocking)
            audit_attempt = deepcopy(state["attempts"][attempt_index])
            audit_attempt.verdict = verdict
            audit_attempt.diagnostic = diagnostic
            audit_attempt.audit_response = audit_response
            audit_attempt.status = "accepted" if verdict == "ACCEPT" else "rejected"
            candidate = deepcopy(state)
            candidate["attempts"][attempt_index] = audit_attempt
            ledger = candidate["ledgers"].setdefault(key, IssueLedger())
            ledger.apply_previous(chosen.previous)
            ledger.add_minor(chosen.minor)
            opened = []
            if verdict != "ACCEPT":
                opened = ledger.add_blocking(chosen.blocking, attempt_number)
                diagnostic = "\n".join(issue.text for issue in opened) or diagnostic
                audit_attempt.diagnostic = diagnostic
                candidate["attempts"][attempt_index] = audit_attempt
                candidate["node_statuses"][key] = "rejected"
            candidate["ledgers"][key] = ledger
            publish(candidate)
            candidate = deepcopy(state)
            log.emit(
                "step.audit",
                "accepted" if verdict == "ACCEPT" else "rejected",
                **event_ids,
                verdict=verdict,
                target_step_id=(
                    node["step_id"]
                    if chosen_target == "current"
                    else chosen_target.split(":", 1)[-1].upper()
                    if chosen_target == "plan"
                    else chosen_target.split(":", 1)[-1]
                ),
                diagnostic=diagnostic,
                response=audit_response,
            )
            if verdict != "ACCEPT":
                if (
                    chosen_target == "plan"
                    and candidate["replans_used"] < max_replans
                    and candidate["retry_counts"].get(key, 0) < retry_limit(node)
                ):
                    replan_prompt = (
                        task.derivation_prompt
                        + "\n\nAudit-established corrections: "
                        + (diagnostic or "The current plan needs revision.")
                    )
                    new_plan_text = _complete(
                        derivation_llm,
                        PLAN_SYSTEM,
                        [{"role": "user", "text": replan_prompt}],
                        _token_handler(log, "planning", "derivation"),
                    )
                    plan_candidate = deepcopy(state)
                    plan_candidate["planner_responses"].append(new_plan_text)
                    publish(plan_candidate)
                    candidate = deepcopy(state)
                    new_steps = parse_plan(new_plan_text)
                    new_first_goal = new_steps[0].substeps[0].goal
                    if not re.search(r"lagrangian|action|\bL\s*=", new_first_goal, re.IGNORECASE):
                        raise PlanError(
                            "first plan substep must establish and justify the Lagrangian"
                        )
                    old_steps = candidate["steps"]
                    old_nodes = _step_nodes(old_steps)
                    old_keys = {item["key"] for item in old_nodes}
                    new_nodes = _step_nodes(new_steps)
                    new_keys = {item["key"] for item in new_nodes}
                    fresh_ids = new_keys - old_keys
                    if fresh_ids & candidate["retired_ids"]:
                        raise PlanError("replan attempted to reuse a retired substep ID")
                    if len(candidate["retired_ids"] | new_keys) > 120:
                        raise PlanError("replan exceeds the 120-node lifetime limit")
                    old_signatures = _node_signatures(candidate["steps"])
                    new_signatures = _node_signatures(new_steps)
                    kept = {
                        item
                        for item in new_keys
                        if item in candidate["accepted"]
                        and old_signatures.get(item) == new_signatures.get(item)
                    }
                    new_by_key = {item["key"]: item for item in new_nodes}
                    new_dep_map = {
                        item["key"]: _node_dependencies(item, new_by_key, new_steps)
                        for item in new_nodes
                    }
                    changed = True
                    while changed:
                        changed = False
                        for item in tuple(kept):
                            if any(dependency not in kept for dependency in new_dep_map[item]):
                                kept.remove(item)
                                changed = True
                    invalidated = old_keys - kept
                    replacements = {
                        item
                        for item in (old_keys & new_keys) - kept
                        if item in candidate["accepted"] or item == key
                    }
                    exhausted_replacements = [
                        item
                        for item in replacements
                        if candidate["retry_counts"].get(item, 0)
                        >= retry_limit(next(node for node in old_nodes if node["key"] == item))
                    ]
                    if exhausted_replacements:
                        old_by_key = {item["key"]: item for item in old_nodes}
                        old_deps = {
                            item["key"]: _node_dependencies(item, old_by_key, old_steps)
                            for item in old_nodes
                        }
                        for item in invalidated:
                            candidate["accepted"].pop(item, None)
                        _exhaust_node(
                            candidate,
                            exhausted_replacements[0],
                            old_nodes,
                            old_deps,
                            diagnostic,
                        )
                        publish(candidate)
                        break
                    for item in replacements:
                        candidate["retry_counts"][item] = candidate["retry_counts"].get(item, 0) + 1
                    candidate["accepted"] = {
                        item: value for item, value in candidate["accepted"].items() if item in kept
                    }
                    candidate["retired_ids"].update(old_keys - new_keys)
                    candidate["retired_ids"].update(fresh_ids)
                    candidate["steps"] = new_steps
                    candidate["node_statuses"] = {
                        item: ("accepted" if item in kept else "pending") for item in new_keys
                    }
                    candidate["queue"] = []
                    candidate["replans_used"] += 1
                    candidate["feedback"][key] = diagnostic
                    enqueue(candidate)
                    candidate["backtracking"].append(
                        Backtrack(
                            "PLAN",
                            [_display_node_key(item, old_steps) for item in sorted(invalidated)],
                            diagnostic,
                        )
                    )
                    audit_attempt.status = "replan_triggered"
                    candidate["attempts"][attempt_index] = audit_attempt
                    publish(candidate)
                    log.emit(
                        "plan.updated",
                        "planning",
                        steps=result.plan,
                        invalidated_steps=sorted(invalidated),
                        reason="replan",
                    )
                    log.emit(
                        "backtracking.started",
                        "planning",
                        target_step_id="PLAN",
                        invalidated_steps=sorted(invalidated),
                        diagnostic=diagnostic,
                    )
                    continue

                if chosen_target.startswith(("parent:", "node:")):
                    kind, target = chosen_target.split(":", 1)
                    nodes_now, by_key_now, deps_now = graph(candidate)
                    if kind == "parent":
                        affected = {item["key"] for item in nodes_now if item["step_id"] == target}
                    else:
                        affected = {target}
                    changed = True
                    while changed:
                        changed = False
                        for candidate_node in nodes_now:
                            if candidate_node["key"] not in affected and any(
                                dependency in affected
                                for dependency in deps_now[candidate_node["key"]]
                            ):
                                affected.add(candidate_node["key"])
                                changed = True
                    ordered_affected = [
                        item["key"] for item in nodes_now if item["key"] in affected
                    ]
                    exhausted = [
                        affected_key
                        for affected_key in ordered_affected
                        if candidate["retry_counts"].get(affected_key, 0)
                        >= retry_limit(by_key_now[affected_key])
                        and (affected_key in candidate["accepted"] or affected_key == key)
                    ]
                    if exhausted:
                        candidate["backtracking"].append(
                            Backtrack(
                                target,
                                [
                                    _display_node_key(item, candidate["steps"])
                                    for item in ordered_affected
                                ],
                                diagnostic,
                            )
                        )
                        for affected_key in ordered_affected:
                            candidate["accepted"].pop(affected_key, None)
                            candidate["node_statuses"][affected_key] = "pending"
                        _exhaust_node(candidate, exhausted[0], nodes_now, deps_now, diagnostic)
                        publish(candidate)
                        break
                    for affected_key in ordered_affected:
                        if affected_key in candidate["accepted"] or affected_key == key:
                            candidate["retry_counts"][affected_key] = (
                                candidate["retry_counts"].get(affected_key, 0) + 1
                            )
                        candidate["accepted"].pop(affected_key, None)
                        candidate["node_statuses"][affected_key] = "pending"
                        candidate["feedback"][affected_key] = diagnostic
                    candidate["queue"] = [
                        item for item in candidate["queue"] if item not in affected
                    ]
                    enqueue(candidate, ordered_affected, front=True)
                    candidate["backtracking"].append(
                        Backtrack(
                            target,
                            [
                                _display_node_key(item, candidate["steps"])
                                for item in ordered_affected
                            ],
                            diagnostic,
                        )
                    )
                    audit_attempt.status = "backtrack_triggered"
                    candidate["attempts"][attempt_index] = audit_attempt
                    publish(candidate)
                    log.emit(
                        "backtracking.started",
                        "revising",
                        target_step_id=target,
                        invalidated_steps=ordered_affected,
                        diagnostic=diagnostic,
                    )
                    continue

                if chosen.insert_substep is not None:
                    insertion = chosen.insert_substep
                    parent = next(item for item in candidate["steps"] if item.id == node["step_id"])
                    children = list(parent.substeps)
                    child_index = next(
                        index
                        for index, item in enumerate(children)
                        if item.id == (node["substep_id"] or "main")
                    )
                    child_id = insertion["id"]
                    inserted = DerivationSubstep(
                        child_id,
                        insertion["goal"].strip(),
                        tuple(insertion.get("depends_on", [])),
                    )
                    current_child = children[child_index]
                    children.insert(child_index, inserted)
                    children[child_index + 1] = DerivationSubstep(
                        current_child.id,
                        current_child.goal,
                        tuple(dict.fromkeys((*current_child.depends_on, child_id))),
                    )
                    replacement_parent = DerivationStep(
                        parent.id,
                        parent.goal,
                        parent.depends_on,
                        tuple(children),
                        True,
                    )
                    candidate["steps"] = [
                        replacement_parent if item.id == parent.id else item
                        for item in candidate["steps"]
                    ]
                    inserted_key = f"{parent.id}.{child_id}"
                    candidate["node_statuses"][inserted_key] = "pending"
                    candidate["retired_ids"].add(inserted_key)
                    candidate["insertion_count"] += 1
                    candidate["retry_counts"][key] = candidate["retry_counts"].get(key, 0) + 1
                    candidate["feedback"][key] = diagnostic
                    candidate["accepted"].pop(key, None)
                    candidate["node_statuses"][key] = "pending"
                    candidate["queue"] = [item for item in candidate["queue"] if item != key]
                    enqueue(candidate, [inserted_key], front=True)
                    audit_attempt.status = "insertion_triggered"
                    candidate["attempts"][attempt_index] = audit_attempt
                    publish(candidate)
                    log.emit(
                        "plan.updated",
                        "revising",
                        steps=result.plan,
                        invalidated_steps=[key],
                        reason="substep_inserted",
                    )
                    continue

                candidate["feedback"][key] = diagnostic
                if candidate["retry_counts"].get(key, 0) >= retry_limit(node):
                    _exhaust_node(candidate, key, nodes, dependency_map, diagnostic)
                    publish(candidate)
                    log.emit(
                        "step.rejected",
                        "rejected",
                        **event_ids,
                        attempt=attempt_number,
                        diagnostic=diagnostic,
                        audit_verdict=verdict,
                    )
                    break
                candidate["retry_counts"][key] = candidate["retry_counts"].get(key, 0) + 1
                candidate["node_statuses"][key] = "pending"
                candidate["queue"] = [key] + [item for item in candidate["queue"] if item != key]
                publish(candidate)
                log.emit(
                    "step.rejected",
                    "rejected",
                    **event_ids,
                    attempt=attempt_number,
                    diagnostic=diagnostic,
                    audit_verdict=verdict,
                )
                log.emit(
                    "step.revising",
                    "revising",
                    **event_ids,
                    retries_used=state["retry_counts"][key],
                    max_step_retries=retry_limit(node),
                )
                continue

            ledger.close_all()
            candidate["ledgers"][key] = ledger
            candidate["accepted"][key] = {
                "goal": child.goal,
                "content": proposal,
                "feedback": "",
                "attempt": attempt_number,
                "audit": audit_response,
                "proposal": proposal,
            }
            candidate["feedback"].pop(key, None)
            candidate["node_statuses"][key] = "accepted"
            if node["key"] == nodes[0]["key"]:
                blocks = _CHECKPOINT_BLOCK.findall(proposal)
                candidate["notation_contract"] = (
                    blocks[-1].strip() if blocks else proposal.strip()[:800]
                )
            accepted_parent = all(
                f"{step.id}.{item.id}" in candidate["accepted"] for item in step.substeps
            )
            publish(candidate)
            if accepted_parent:
                parent_content = "\n\n".join(
                    state["accepted"][f"{step.id}.{item.id}"]["content"] for item in step.substeps
                )
                parent_audit = "\n\n".join(
                    state["accepted"][f"{step.id}.{item.id}"]["audit"] for item in step.substeps
                )
                log.emit(
                    "step.accepted",
                    "accepted",
                    step_id=step.id,
                    attempt=attempt_number,
                    content=parent_content,
                    audit=parent_audit,
                )
            if node["substep_id"] is not None:
                log.emit(
                    "substep.accepted",
                    "accepted",
                    **event_ids,
                    attempt=attempt_number,
                    content=proposal,
                    audit=audit_response,
                )
            enqueue_candidate = deepcopy(state)
            enqueue(enqueue_candidate)
            if enqueue_candidate["queue"] != state["queue"]:
                publish(enqueue_candidate)
        if state["status"] == "running":
            candidate = deepcopy(state)
            candidate["status"] = (
                "completed_verified"
                if candidate.get("verification", {}).get("passed") is True
                else "completed_unverified"
            )
            publish(candidate)
            log.emit(
                "derivation.completed",
                result.status,
                run_dir=str(run_dir),
                final_amplitude=result.final_amplitude,
                verification=result.verification,
            )
        elif state["status"] == "retry_exhausted":
            log.emit(
                "derivation.completed",
                "retry_exhausted",
                run_dir=str(run_dir),
                error=result.error,
                accepted_steps=result.accepted_steps,
            )
        else:
            log.emit(
                "derivation.failed",
                "failed",
                run_dir=str(run_dir),
                error=result.error,
                error_code=result.error_code,
            )
    except _SnapshotError as exc:
        sync_failure(exc.code, str(exc))
        try:
            log.emit(
                "derivation.failed",
                "failed",
                run_dir=str(run_dir),
                error=result.error,
                error_code=result.error_code,
            )
        except Exception:
            pass
    except Exception as exc:
        code = "invalid_plan" if isinstance(exc, PlanError) else None
        message = str(exc) or f"{type(exc).__name__} during checkpoint workflow"
        candidate = deepcopy(state)
        candidate["status"] = "failed"
        candidate["error_code"] = code
        candidate["error"] = message
        try:
            publish(candidate)
        except _SnapshotError as snapshot_exc:
            sync_failure(snapshot_exc.code, str(snapshot_exc))
        if not result.error:
            result.status = "failed"
            result.error_code = code
            result.error = message
        try:
            log.emit(
                "derivation.failed",
                "failed",
                run_dir=str(run_dir),
                error=result.error,
                error_code=result.error_code,
            )
        except Exception:
            pass
    finally:
        result.accepted_steps = _aggregate_accepted(state["steps"], state["accepted"])
        result.attempts = state["attempts"]
        if result.status == "failed" and not result.error:
            result.error = state.get("error", "Workflow failed.")
            result.error_code = state.get("error_code")
        refresh_reports(force=True)
        try:
            (run_dir / "outcome.json").write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            _LOGGER.warning(
                "Could not write autonomous outcome %s",
                run_dir / "outcome.json",
                exc_info=True,
            )
    return result
