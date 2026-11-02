"""Tree-level cross sections and two-body phase space (inputs independent of the loop).

Final states of phi1 phi1 -> X in the dim-4 sector: phi1 phi1 (tree amplitude c4) and
phi2 phi2 (tree amplitude c42). Each state consists of two identical particles, so each
carries a factor 1/2.

    dsigma/dt = |A|^2 / (16 pi s^2)           (massless 2 -> 2)
    sigma     = (c4^2 + c42^2) / (32 pi s)

Unitarity / optical theorem:
    2 Im A(s, t) = Sum_X S_X Int dPhi_2 A(12->X) A*(34->X),   Im A(s, 0) = s * sigma.
"""

from __future__ import annotations

import math

from scipy.integrate import quad

from .problem import ProblemSpec

IDENTICAL_PAIR_FACTOR = 0.5


def final_states(problem: ProblemSpec) -> list[tuple[str, float, float]]:
    """(label, tree amplitude, identical-particle factor) for each two-body final state."""
    return [
        ("phi1 phi1", problem.c4, IDENTICAL_PAIR_FACTOR),
        ("phi2 phi2", problem.c42, IDENTICAL_PAIR_FACTOR),
    ]


def two_body_phase_space_numeric() -> float:
    """Integral of dPhi_2 for massless particles, by quadrature over the solid angle.

    dPhi_2 = |p|/(16 pi^2 sqrt(s)) dOmega = dOmega/(32 pi^2) since |p| = sqrt(s)/2.
    Expected value: 1/(8 pi).
    """
    cos_integral, _ = quad(lambda _c: 1.0, -1.0, 1.0)
    solid_angle = 2.0 * math.pi * cos_integral
    return solid_angle / (32.0 * math.pi**2)


def sigma_tree(problem: ProblemSpec, s: float, method: str = "closed") -> float:
    """Total tree-level cross section phi1 phi1 -> X.

    method="closed":      c^2/(32 pi s)
    method="dsigma_dt":   numerical integration of dsigma/dt over t in [-s, 0]
    method="phase_space": |A|^2 * Phi_2 / (2 s) with the massless flux factor 2s
    """
    if s <= 0.0:
        raise ValueError("Cross sections require s > 0.")
    if method == "closed":
        return problem.coupling_sq / (32.0 * math.pi * s)
    total = 0.0
    if method == "dsigma_dt":
        for _label, amp, factor in final_states(problem):
            value, _ = quad(lambda _t, a=amp: a * a / (16.0 * math.pi * s * s), -s, 0.0)
            total += factor * value
        return float(total)
    if method == "phase_space":
        phi2 = two_body_phase_space_numeric()
        for _label, amp, factor in final_states(problem):
            total += factor * amp * amp * phi2 / (2.0 * s)
        return float(total)
    raise ValueError(f"Unknown method {method!r}.")


def im_amplitude_from_unitarity(problem: ProblemSpec) -> float:
    """Im A(phi1 phi1 -> phi1 phi1) at one loop from the tree-level two-body cut.

    The tree amplitudes are constants, so the result is angle independent:
    Im A = (1/2) Sum_X S_X |A_X|^2 Phi_2 = (c4^2 + c42^2) / (32 pi).
    """
    phi2 = two_body_phase_space_numeric()
    total = sum(factor * amp * amp for _label, amp, factor in final_states(problem))
    return 0.5 * total * phi2


def im_bubble_from_unitarity() -> float:
    """Im B for s > 0 implied by unitarity alone (should be +pi)."""
    from . import conventions

    unit = ProblemSpec(name="unit", c4=1.0, c42=0.0)
    return im_amplitude_from_unitarity(unit) / conventions.ONE_LOOP_PREFACTOR
