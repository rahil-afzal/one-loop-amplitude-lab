import secrets
from dataclasses import dataclass, field

from .candidate import Candidate, CandidateError
from .checks import SUITES, Sampler


@dataclass
class CheckResult:
    name: str
    ok: bool
    hint: str  # what the agent is told
    detail: dict = field(default_factory=dict)  # numbers for the log, not shown to the agent


@dataclass
class Report:
    results: list
    seed: int
    suite: str

    @property
    def ran(self):
        return not any(r.name == "runs" for r in self.results)

    @property
    def passed(self):
        return all(r.ok for r in self.results)

    @property
    def structural_ok(self):
        """Every check except the comparison with the reference passes."""
        return self.ran and all(r.ok for r in self.results if r.name != "reference_match")

    def feedback(self, attempt=None, max_attempts=None):
        head = f"{sum(r.ok for r in self.results)} of {len(self.results)} checks passed"
        if not self.ran:
            head = "your file did not run, so no checks were evaluated"
        if attempt is not None:
            head = f"submission {attempt} of {max_attempts}: " + head
        lines = [head]
        for r in self.results:
            lines.append(("PASS " if r.ok else "FAIL ") + r.name + ("" if r.ok else ": " + r.hint))
        return "\n".join(lines)

    def to_dict(self):
        return {
            "passed": self.passed,
            "structural_ok": self.structural_ok,
            "suite": self.suite,
            "seed": self.seed,
            "checks": [
                {"name": r.name, "ok": r.ok, "hint": r.hint, "detail": r.detail}
                for r in self.results
            ],
        }


def verify(candidate: Candidate, suite="phi4_dim4", seed=None) -> Report:
    """Run a check suite. A fresh random seed per call means the points are never reused."""
    if seed is None:
        seed = secrets.randbits(32)
    checks = SUITES[suite]

    smoke = Sampler(seed, "runs")
    (s, t, u), c4, mu2 = smoke.kinematics(1)[0], smoke.couplings(1)[0], smoke.mu2s(1)[0]
    try:
        candidate.evaluate([(s, t, u, c4, mu2)])
    except CandidateError as exc:
        return Report([CheckResult("runs", False, f"your file failed to run: {exc}")], seed, suite)

    results = []
    for name, check, hint in checks:
        try:
            ok, detail = check(candidate.evaluate, Sampler(seed, name))
            results.append(CheckResult(name, bool(ok), hint, detail))
        except CandidateError as exc:
            results.append(CheckResult(name, False, f"amplitude() failed in this check: {exc}"))

    # the structural checks are blind to an additive constant; saying so is useful, but the
    # reference check can also fail for plain numerical inaccuracy, so the hint names both
    only_reference_fails = [r.name for r in results if not r.ok] == ["reference_match"]
    if only_reference_fails:
        results[-1].hint = (
            "every other check passes, but the amplitude differs from the MS-bar reference: "
            "a finite constant, or numerical accuracy below the required level"
        )
    return Report(results, seed, suite)
