"""Deterministic verification of a derivation agent's final answer.

The derivation agent must end its answer with a fenced block labelled ``final``::

    ```final
    A_tree = c4
    A_loop = c4**2/(32*pi**2) * (6 - log(-s/mu2) - log(-t/mu2) - log(-u/mu2))
    beta_c4 = -3*c4**2/(16*pi**2)
    ```

Each right-hand side is parsed through the strict AST allowlist (never executed) and
compared with the repository's own amplitude code. No LLM is called here. The reference
beta sign follows ``conventions.BETA_C4_COEFFICIENT``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import mpmath as mp
import sympy as sp

from . import conventions
from .amplitude import assemble
from .assistant.safe_expr import CandidateParseError, parse_expression
from .benchmarks import pure_phi4
from .kinematics import Kinematics

_FENCE = re.compile(r"```final\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
KEYS = ("A_tree", "A_loop", "beta_c4")
EXTRA_NAMES = ("c4", "t", "u")
C4 = 0.7
POINTS = ((5.0, -2.0), (2.0, -1.0), (0.5, -0.25))  # physical (s, t); u = -s - t
MU2_VALUES = (1.0, 2.7)
ETA = 1e-12  # s -> s + i*eta


@dataclass
class Check:
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class VerificationReport:
    parsed: dict
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failed_names(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "parsed": self.parsed,
            "checks": [asdict(check) for check in self.checks],
        }

    def summary(self) -> str:
        return "\n".join(
            f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.message}" for c in self.checks
        )


def extract_final_block(text: str) -> dict[str, str]:
    """Return ``{key: right-hand side}`` from the last ```final block (empty if absent)."""
    blocks = _FENCE.findall(text or "")
    found: dict[str, str] = {}
    for line in (blocks[-1].splitlines() if blocks else []):
        key, sep, value = line.partition("=")
        if sep and key.strip() in KEYS and value.strip():
            found[key.strip()] = value.strip()
    return found


def _numeric(expr: sp.Expr, values: dict[str, complex]) -> complex:
    symbols = sorted(expr.free_symbols, key=str)
    function = sp.lambdify(symbols, expr, modules="mpmath")
    with mp.workdps(30):
        return complex(function(*(mp.mpmathify(values[str(sym)]) for sym in symbols)))


def _point(s: float, t: float, mu2: float, c4: float = C4) -> dict[str, complex]:
    return {"s": complex(s, ETA), "t": complex(t), "u": complex(-s - t), "mu2": mu2, "c4": c4}


def _evaluation_failed(name: str, exc: Exception) -> Check:
    return Check(name, False, f"evaluation failed: {type(exc).__name__}")


def check_tree(expr: sp.Expr) -> Check:
    try:
        value = _numeric(expr, _point(*POINTS[0], 1.0))
    except Exception as exc:
        return _evaluation_failed("tree_value", exc)
    return Check("tree_value", abs(value - C4) < 1e-9, f"A_tree={value:.6g}, expected {C4}")


def check_loop(expr: sp.Expr) -> list[Check]:
    """Compare with the assembled one-loop amplitude at physical points (s + i0)."""
    worst, worst_row = 0.0, {}
    for mu2 in MU2_VALUES:
        problem = pure_phi4().with_(c4=C4, mu2=mu2)
        for s, t in POINTS:
            expected = assemble(problem, Kinematics.from_s_t(s, t)).one_loop
            try:
                got = _numeric(expr, _point(s, t, mu2))
            except Exception as exc:
                return [_evaluation_failed("loop_physical_region", exc)]
            error = abs(got - expected) / max(1.0, abs(expected))
            if error >= worst:
                worst = error
                worst_row = {"s": s, "t": t, "mu2": mu2, "got": str(got), "expected": str(expected)}
    ok = worst < 1e-6
    checks = [Check("loop_physical_region", ok, f"worst relative error {worst:.3e}", worst_row)]
    if not ok:
        got, expected = complex(worst_row["got"]), complex(worst_row["expected"])
        checks.append(
            Check(
                "loop_diagnosis",
                False,
                f"Re(got-expected)={(got - expected).real:.4g}, "
                f"Im(got-expected)={(got - expected).imag:.4g}",
            )
        )
    return checks


def check_crossing(expr: sp.Expr) -> Check:
    """Off-axis complex (s, t, u) with s+t+u=0: the result must be permutation invariant."""
    values = (1.0 + 0.5j, -2.0 - 0.3j, 1.0 - 0.2j)
    results = []
    try:
        for order in ((0, 1, 2), (1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)):
            s, t, u = (values[i] for i in order)
            results.append(_numeric(expr, {"s": s, "t": t, "u": u, "mu2": 1.0, "c4": C4}))
    except Exception as exc:
        return _evaluation_failed("crossing_symmetry", exc)
    spread = max(abs(r - results[0]) for r in results)
    return Check("crossing_symmetry", spread < 1e-9, f"spread over permutations {spread:.3e}")


def check_beta(expr: sp.Expr) -> Check:
    try:
        value = _numeric(expr, _point(*POINTS[0], 1.0))
    except Exception as exc:
        return _evaluation_failed("beta_c4", exc)
    expected = conventions.BETA_C4_COEFFICIENT * C4**2
    ok = abs(value - expected) < 1e-9
    note = " (sign opposite to reference)" if abs(value + expected) < 1e-9 else ""
    return Check("beta_c4", ok, f"beta={value:.8g}, expected {expected:.8g}{note}")


def verify_final_answer(text: str) -> VerificationReport:
    """Verify the ```final block of ``text``; a missing or unparseable entry fails."""
    raw = extract_final_block(text)
    checks: list[Check] = []
    exprs: dict[str, sp.Expr] = {}
    parsed: dict[str, str] = {}
    for key in KEYS:
        if key not in raw:
            checks.append(Check(f"parse_{key}", False, "missing from final block"))
            continue
        try:
            exprs[key] = parse_expression(raw[key], extra_names=EXTRA_NAMES)
        except CandidateParseError as exc:
            checks.append(Check(f"parse_{key}", False, f"rejected: {exc}"))
            continue
        parsed[key] = str(exprs[key])
        checks.append(Check(f"parse_{key}", True, parsed[key]))
    if "A_tree" in exprs:
        checks.append(check_tree(exprs["A_tree"]))
    if "A_loop" in exprs:
        checks.extend(check_loop(exprs["A_loop"]))
        checks.append(check_crossing(exprs["A_loop"]))
    if "beta_c4" in exprs:
        checks.append(check_beta(exprs["beta_c4"]))
    return VerificationReport(parsed, checks)

