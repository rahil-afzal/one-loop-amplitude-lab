"""Dispersion integrals: the dim-4 Sigma benchmark and a bubble-level dispersion check.

Sigma (dim-4 piece, Ye-He-Gu Eq. (3.11) subset):

    Sigma = (2/pi) Int_0^inf ds s^4 sigma(s) / (s^2 + s0^2)^3 = c^2 / (64 pi^2 s0^2)

where c^2 = c4^2 + c42^2. This is NOT the full Eq. (3.9).

Bubble dispersion (twice-subtracted, Euclidean x < 0):

    B''(x) = (2/pi) Int_0^inf ds' Im B(s') / (s' - x)^3 = 1/x^2

with Im B(s') taken from unitarity (two-body phase space), and B''(x) from numerical
differentiation of the Feynman-parameter integral. This links the cut to the real part.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mpmath as mp
import sympy as sp
from scipy.integrate import quad

from .amplitude import forward_one_loop_imag
from .crosssection import im_bubble_from_unitarity, sigma_tree
from .integrals.bubble import feynman_integral_mp
from .problem import ProblemSpec


def sigma_reference(problem: ProblemSpec, s0: float) -> float:
    """Closed form c^2 / (64 pi^2 s0^2)."""
    return problem.coupling_sq / (64.0 * math.pi**2 * s0 * s0)


def sigma_from_cross_section(problem: ProblemSpec, s0: float) -> float:
    """Right-hand side: integrate s^4 sigma(s)/(s^2+s0^2)^3 numerically."""

    def integrand(s: float) -> float:
        if s <= 0.0:
            return 0.0
        return s**4 * sigma_tree(problem, s) / (s * s + s0 * s0) ** 3

    value, _ = quad(integrand, 0.0, math.inf, limit=200)
    return (2.0 / math.pi) * float(value)


def sigma_from_amplitude(problem: ProblemSpec, s0: float) -> float:
    """Left-hand side: integrate s^3 Im A_forward(s)/(s^2+s0^2)^3, Im A from the assembled loop."""

    def integrand(s: float) -> float:
        if s <= 0.0:
            return 0.0
        return s**3 * forward_one_loop_imag(problem, s) / (s * s + s0 * s0) ** 3

    value, _ = quad(integrand, 0.0, math.inf, limit=200)
    return (2.0 / math.pi) * float(value)


@lru_cache(maxsize=1)
def _symbolic_s_integral() -> sp.Expr:
    s = sp.Symbol("s", positive=True)
    s0 = sp.Symbol("s0", positive=True)
    return sp.simplify(sp.integrate(s**3 / (s**2 + s0**2) ** 3, (s, 0, sp.oo)))


def sigma_symbolic(problem: ProblemSpec, s0: float) -> float:
    """Sigma via SymPy symbolic integration of the s-dependence (third route)."""
    s0_sym = sp.Symbol("s0", positive=True)
    integral = _symbolic_s_integral().subs(s0_sym, s0)
    im_forward = problem.coupling_sq / (32.0 * math.pi)  # constant Im A = s * sigma
    return float((2.0 / math.pi) * im_forward * integral)


def bubble_second_derivative_numeric(x: float, mu2: float = 1.0, precision: int = 30) -> float:
    """B''(x) for x < 0 by numerical differentiation of the Feynman-parameter integral."""
    if x >= 0.0:
        raise ValueError("Euclidean point required (x < 0).")
    with mp.workdps(precision):
        mu2_mp = mp.mpf(mu2)
        value = mp.diff(lambda y: mp.re(feynman_integral_mp(mp.mpf(y), mu2_mp)), mp.mpf(x), 2)
        return float(value)


def bubble_second_derivative_dispersive(x: float) -> float:
    """(2/pi) Int_0^inf Im B(s')/(s'-x)^3 ds' with Im B from unitarity."""
    if x >= 0.0:
        raise ValueError("Euclidean point required (x < 0).")
    im_b = im_bubble_from_unitarity()
    value, _ = quad(lambda sp_: 1.0 / (sp_ - x) ** 3, 0.0, math.inf, limit=200)
    return (2.0 / math.pi) * im_b * float(value)
