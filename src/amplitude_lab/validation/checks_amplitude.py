"""Checks on the assembled amplitude: combinatorics, crossing, scaling."""

from __future__ import annotations

import math
from fractions import Fraction

from .. import combinatorics, conventions
from ..amplitude import assemble
from ..kinematics import BENCHMARK_POINTS, PERMUTATIONS
from ..problem import ProblemSpec
from .base import CONSISTENCY, INDEPENDENT, Outcome, register, rel_err


@register(
    "combinatorics_wick_factors",
    INDEPENDENT,
    "Brute-force Wick enumeration: tree coefficient 1, bubble coefficient 1/2, c42 definition.",
)
def check_wick_factors(problem: ProblemSpec) -> Outcome:
    phi4 = ("1", "1", "1", "1")
    mixed = ("1", "1", "2", "2")
    tree = combinatorics.coefficients(phi4, phi4, 1)
    loop_c4 = combinatorics.coefficients(phi4, phi4, 2)
    loop_c42 = combinatorics.coefficients(mixed, phi4, 2)
    tree_c42 = combinatorics.coefficients(mixed, mixed, 1)

    half = Fraction(1, 2)
    expected_ok = (
        tree == {"contact": Fraction(1)}
        and loop_c4 == {"s": half, "t": half, "u": half}
        and loop_c42 == {"s": half, "t": half, "u": half}
        and tree_c42 == {"contact": Fraction(1)}
    )

    unit = problem.with_(c4=1.0, c42=0.0)
    kin = BENCHMARK_POINTS[0]
    tree_in_code = assemble(unit, kin).tree.real
    loop_coefficient_in_code = conventions.ONE_LOOP_PREFACTOR * 16.0 * math.pi**2
    linked = abs(tree_in_code - float(tree["contact"])) < 1e-14 and (
        abs(loop_coefficient_in_code - float(half)) < 1e-14
    )
    return Outcome(
        expected_ok and linked,
        "Enumerated coefficients match the constants used by the amplitude assembly.",
        {
            "tree_phi4": {k: str(v) for k, v in tree.items()},
            "loop_c4": {k: str(v) for k, v in loop_c4.items()},
            "loop_c42": {k: str(v) for k, v in loop_c42.items()},
            "tree_phi1phi1_to_phi2phi2": {k: str(v) for k, v in tree_c42.items()},
            "code_tree_over_c4": tree_in_code,
            "code_loop_coefficient": loop_coefficient_in_code,
        },
    )


@register(
    "amplitude_crossing",
    CONSISTENCY,
    "Assembled amplitude is invariant under all six permutations of (s, t, u).",
)
def check_crossing(problem: ProblemSpec) -> Outcome:
    worst = 0.0
    for kin in BENCHMARK_POINTS:
        base = assemble(problem, kin).total
        for order in PERMUTATIONS:
            worst = max(worst, rel_err(assemble(problem, kin.permuted(order)).total, base))
    tol = 1e-12
    return Outcome(
        worst < tol,
        "Crossing symmetry holds for identical external states.",
        {"worst_relative_error": worst, "permutations_tested": 6 * len(BENCHMARK_POINTS)},
        tol,
    )


@register(
    "amplitude_scaling",
    CONSISTENCY,
    "Joint rescaling of (s, t, u, mu2) is a symmetry; rescaling alone shifts by -3 ln(lambda).",
)
def check_scaling(problem: ProblemSpec) -> Outcome:
    kin = BENCHMARK_POINTS[0]
    lam = 2.7
    base = assemble(problem, kin).total
    joint = assemble(problem.with_(mu2=problem.mu2 * lam), kin.scaled(lam)).total
    shifted = assemble(problem, kin.scaled(lam)).total
    expected_shift = -3.0 * conventions.ONE_LOOP_PREFACTOR * problem.coupling_sq * math.log(lam)
    err_joint = abs(joint - base)
    err_shift = abs((shifted - base) - expected_shift)
    tol = 1e-12
    return Outcome(
        err_joint < tol and err_shift < tol,
        "Amplitude depends on s/mu2 only; anomalous log scaling matches -3 ln(lambda).",
        {"joint_scaling_error": err_joint, "log_shift_error": err_shift, "lambda": lam},
        tol,
    )
