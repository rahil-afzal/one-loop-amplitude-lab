"""Checks on the bubble integral itself."""

from __future__ import annotations

import math

from .. import conventions
from ..integrals.bubble import (
    bubble_analytic,
    bubble_discontinuity,
    bubble_numeric_mpmath,
    bubble_real_part_quad,
)
from ..integrals.momentum_space import euclidean_bubble_difference_numeric
from ..problem import ProblemSpec
from .base import IMPLEMENTATION, INDEPENDENT, Outcome, register, rel_err


@register(
    "integral_feynman_quadrature",
    IMPLEMENTATION,
    "Closed-form bubble vs scipy and mpmath quadrature of the Feynman-parameter integral.",
)
def check_feynman_quadrature(problem: ProblemSpec) -> Outcome:
    mu2 = problem.mu2
    rows = {}
    worst_scipy = worst_mp = 0.0
    for s in (-5.0, -1.0, -0.2):
        ana = bubble_analytic(s, mu2)
        e_scipy = rel_err(bubble_real_part_quad(s, mu2), ana.real)
        e_mp = rel_err(bubble_numeric_mpmath(s, mu2), ana)
        rows[str(s)] = {"scipy": e_scipy, "mpmath": e_mp}
        worst_scipy, worst_mp = max(worst_scipy, e_scipy), max(worst_mp, e_mp)
    complex_rows = {}
    worst_complex = 0.0
    for s in (-3 + 1j, 5 + 2j, 0.5 - 0.3j, -0.2 - 0.7j):
        err = rel_err(bubble_numeric_mpmath(s, mu2), bubble_analytic(s, mu2))
        complex_rows[str(s)] = err
        worst_complex = max(worst_complex, err)
    tol = 1e-10
    ok = max(worst_scipy, worst_mp, worst_complex) < tol
    return Outcome(
        ok,
        "Analytic bubble agrees with both quadratures (real axis and complex plane).",
        {"real_axis": rows, "complex_plane": complex_rows},
        tol,
    )


@register(
    "integral_momentum_space",
    INDEPENDENT,
    "B(s1)-B(s2) vs a 4D Euclidean momentum-space integral (no Feynman parameters).",
)
def check_momentum_space(problem: ProblemSpec) -> Outcome:
    rows = []
    worst = 0.0
    for p1_sq, p2_sq in ((0.5, 3.0), (1.0, 7.0), (0.2, 0.9)):
        numeric = euclidean_bubble_difference_numeric(p1_sq, p2_sq)
        closed = (bubble_analytic(-p1_sq, problem.mu2) - bubble_analytic(-p2_sq, problem.mu2)).real
        err = abs(numeric - closed)
        worst = max(worst, err)
        rows.append({"P1_sq": p1_sq, "P2_sq": p2_sq, "numeric": numeric, "closed_form": closed})
    tol = 1e-8
    return Outcome(
        worst < tol,
        "Momentum-space difference integral reproduces ln(P2^2/P1^2) = B(s1) - B(s2).",
        {"points": rows, "worst_abs_error": worst},
        tol,
    )


@register(
    "integral_branch_cut",
    IMPLEMENTATION,
    "Explicit continuation s -> s + i*eta: Im B -> +pi, discontinuity 2*pi*i.",
)
def check_branch_cut(problem: ProblemSpec) -> Outcome:
    mu2 = problem.mu2
    rows = []
    ok = True
    for s in (0.5, 2.0, 5.0):
        ana_up = bubble_analytic(s, mu2, side=+1)
        ana_down = bubble_analytic(s, mu2, side=-1)
        ok = ok and abs(ana_up.imag - math.pi) < 1e-14 and abs(ana_down.imag + math.pi) < 1e-14
        ok = ok and abs(bubble_discontinuity(s, mu2) - 2j * math.pi) < 1e-14
        per_eta = {}
        for eta in (1e-3, 1e-4, 1e-5):
            up = bubble_numeric_mpmath(s, mu2, eta=eta)
            down = bubble_numeric_mpmath(s, mu2, eta=-eta)
            expected_im = math.pi - math.atan(eta / s)  # exact for the principal log
            ok = ok and abs(up.imag - expected_im) < 1e-9
            ok = ok and abs((up - down).imag - 2.0 * expected_im) < 1e-9
            per_eta[str(eta)] = {"Im_up": up.imag, "Im_down": down.imag}
        last = bubble_numeric_mpmath(s, mu2, eta=1e-5)
        ok = ok and abs(last.imag - math.pi) < 10.0 * 1e-5 / s
        rows.append({"s": s, "by_eta": per_eta})
    return Outcome(
        ok,
        "Numerical continuation converges to +pi from above and -pi from below.",
        {"points": rows, "expected_imag": math.pi},
        1e-9,
    )


@register(
    "integral_symmetry_prefactor_link",
    IMPLEMENTATION,
    "Loop prefactor equals (Wick symmetry factor) x (1/16pi^2).",
)
def check_prefactor_link(problem: ProblemSpec) -> Outcome:
    prefactor = conventions.ONE_LOOP_PREFACTOR  # read at call time, never a stale import
    expected = conventions.BUBBLE_SYMMETRY_FACTOR / (16.0 * math.pi**2)
    err = abs(prefactor - expected) / expected
    return Outcome(
        err < 1e-14,
        "ONE_LOOP_PREFACTOR is consistent with its documented decomposition.",
        {"ONE_LOOP_PREFACTOR": prefactor, "expected": expected},
        1e-14,
    )
