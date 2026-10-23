"""Assistant tasks: a prompt, a parser, and a deterministic verifier.

A task never trusts the model. ``verify`` compares the parsed proposal with the
deterministic numerical Feynman-parameter integral (mpmath), at points off the cut, plus
boundary points that pin down the physical sheet (+i*pi for s > 0).
"""

from __future__ import annotations

import math
from typing import Protocol

import sympy as sp

from ..integrals.bubble import bubble_numeric_mpmath
from ..problem import ProblemSpec
from .safe_expr import evaluate, parse_expression


class ProposalTask(Protocol):
    name: str
    system_instruction: str
    prompt: str

    def parse(self, text: str) -> sp.Expr: ...

    def verify(self, expr: sp.Expr, problem: ProblemSpec) -> dict: ...


class BubbleClosedFormTask:
    """Ask for the closed form of the finite massless bubble B(s; mu2)."""

    name = "bubble_closed_form"

    system_instruction = (
        "You are a theoretical physics calculation assistant.\n"
        "Definition: B(s; mu2) = - Integral_0^1 dx Log[(-x(1-x)s - i0)/mu2], with Log the\n"
        "principal complex logarithm. The physical sheet is the upper-half s-plane boundary.\n"
        "Return ONLY one symbolic expression in the variables s and mu2, valid for complex s\n"
        "off the positive real axis, using the principal branch of log. Do not add explicit\n"
        "i*pi terms. No piecewise functions. Allowed: s, mu2, pi, I, log, ln, sqrt, exp,\n"
        "numbers, + - * / and ^ with small numeric exponents. No code, no commentary."
    )
    prompt = "Derive the closed form of B(s; mu2) and return it as a single expression."

    interior_points = (-3 + 1j, 5 + 2j, 0.5 - 0.3j, -0.2 - 0.7j, 12 - 0.05j)
    boundary_points = (0.5, 5.0)
    tolerance = 1e-9

    def parse(self, text: str) -> sp.Expr:
        return parse_expression(text)

    def verify(self, expr: sp.Expr, problem: ProblemSpec) -> dict:
        rows = []
        ok = True
        for mu2 in (problem.mu2, 2.7):
            for s in self.interior_points:
                rows.append(self._compare(expr, s, mu2, bubble_numeric_mpmath(s, mu2), "interior"))
            rows.append(
                self._compare(expr, -3.0, mu2, bubble_numeric_mpmath(-3.0, mu2), "euclidean")
            )
            for s in self.boundary_points:
                eta = 1e-9
                expected = complex(2.0 - math.log(s / mu2), math.pi)  # physical-sheet boundary
                rows.append(self._compare(expr, s + 1j * eta, mu2, expected, "boundary", 1e-6))
        ok = all(row["passed"] for row in rows)
        return {
            "expression": str(expr),
            "passed": ok,
            "points": rows,
            "reference": "mpmath Feynman-parameter quadrature; analytic boundary value for s > 0",
            "caveat": "Finite-sample test on the physical sheet; not a proof of analyticity.",
        }

    def _compare(self, expr, s, mu2, reference, kind, tol=None) -> dict:
        tol = tol or self.tolerance
        try:
            guess = evaluate(expr, s, mu2)
            error = abs(guess - reference) / max(1.0, abs(reference))
            return {
                "kind": kind,
                "s": str(s),
                "mu2": mu2,
                "passed": bool(error < tol),
                "relative_error": error,
            }
        except Exception as exc:
            return {
                "kind": kind,
                "s": str(s),
                "mu2": mu2,
                "passed": False,
                "error": type(exc).__name__,
            }
