"""Run an assistant task: ask, parse safely, verify deterministically, report."""

from __future__ import annotations

from dataclasses import dataclass

from ..problem import ProblemSpec
from .client import AssistantClient
from .errors import AssistantConfigError, AssistantRequestError, AssistantUnavailableError
from .safe_expr import CandidateParseError
from .tasks import ProposalTask

MAX_RAW_CHARS = 2000


@dataclass
class AssistantOutcome:
    task: str
    client: str
    status: str  # verified | rejected | rejected_unparseable | unavailable
    raw_proposal: str | None = None
    expression: str | None = None
    verification: dict | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "client": self.client,
            "status": self.status,
            "raw_proposal": self.raw_proposal,
            "expression": self.expression,
            "verification": self.verification,
            "detail": self.detail,
            "note": "A 'verified' status means the finite-sample deterministic test passed; "
            "it is not a proof. The assistant never influences the overall status.",
        }


def run_task(client: AssistantClient, task: ProposalTask, problem: ProblemSpec) -> AssistantOutcome:
    outcome = AssistantOutcome(task=task.name, client=getattr(client, "name", "unknown"), status="")
    try:
        raw = client.generate(task.prompt, task.system_instruction)
    except (AssistantConfigError, AssistantUnavailableError, AssistantRequestError) as exc:
        outcome.status = "unavailable"
        outcome.detail = str(exc)
        return outcome
    outcome.raw_proposal = raw[:MAX_RAW_CHARS]
    try:
        expr = task.parse(raw)
    except CandidateParseError as exc:
        outcome.status = "rejected_unparseable"
        outcome.detail = str(exc)
        return outcome
    outcome.expression = str(expr)
    outcome.verification = task.verify(expr, problem)
    outcome.status = "verified" if outcome.verification["passed"] else "rejected"
    return outcome
