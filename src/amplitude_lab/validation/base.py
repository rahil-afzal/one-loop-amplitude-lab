"""Check infrastructure: evidence taxonomy, result type, registry and runner.

Evidence kinds (strongest claim each check is allowed to make):

implementation  two evaluations of the SAME definition agree
consistency     the assembled amplitude satisfies internal identities
reference       agreement with an externally stated formula (paper / textbook)
independent     the quantity is obtained by a genuinely different route
                (different input physics or different mathematical method)

No check calls an LLM. No check certifies global analytic correctness from finitely
many sample points.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

from ..problem import ProblemSpec

IMPLEMENTATION = "implementation"
CONSISTENCY = "consistency"
REFERENCE = "reference"
INDEPENDENT = "independent"
EVIDENCE_KINDS = (IMPLEMENTATION, CONSISTENCY, REFERENCE, INDEPENDENT)


@dataclass
class Outcome:
    ok: bool
    message: str
    details: dict = field(default_factory=dict)
    tolerance: float | None = None


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "error"
    evidence: str
    description: str
    message: str
    details: dict = field(default_factory=dict)
    tolerance: float | None = None
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
            "description": self.description,
            "message": self.message,
            "tolerance": self.tolerance,
            "duration_s": round(self.duration_s, 4),
            "details": self.details,
        }


CheckFn = Callable[[ProblemSpec], Outcome]


@dataclass(frozen=True)
class CheckSpec:
    name: str
    evidence: str
    description: str
    func: CheckFn

    def run(self, problem: ProblemSpec) -> CheckResult:
        start = time.perf_counter()
        try:
            outcome = self.func(problem)
            status = "pass" if outcome.ok else "fail"
            result = CheckResult(
                name=self.name,
                status=status,
                evidence=self.evidence,
                description=self.description,
                message=outcome.message,
                details=outcome.details,
                tolerance=outcome.tolerance,
            )
        except Exception as exc:  # a crashing check is a failure, never a silent pass
            result = CheckResult(
                name=self.name,
                status="error",
                evidence=self.evidence,
                description=self.description,
                message=f"{type(exc).__name__}: {exc}",
                details={"traceback": traceback.format_exc(limit=3)},
            )
        result.duration_s = time.perf_counter() - start
        return result


REGISTRY: list[CheckSpec] = []


def register(name: str, evidence: str, description: str) -> Callable[[CheckFn], CheckFn]:
    if evidence not in EVIDENCE_KINDS:
        raise ValueError(f"Unknown evidence kind {evidence!r}.")

    def decorator(func: CheckFn) -> CheckFn:
        if any(spec.name == name for spec in REGISTRY):
            raise ValueError(f"Duplicate check name {name!r}.")
        REGISTRY.append(CheckSpec(name, evidence, description, func))
        return func

    return decorator


def rel_err(value: complex, reference: complex) -> float:
    """|value - reference| / max(1, |reference|)."""
    return float(abs(value - reference) / max(1.0, abs(reference)))
