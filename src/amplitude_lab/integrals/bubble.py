"""Finite MSbar massless scalar bubble: analytic and numerical evaluations.

    B(s; mu2) = - Int_0^1 dx Log[(-x(1-x) s - i0) / mu2]  =  2 - Log(-s/mu2)

The function is analytic in the s-plane cut along s > 0. The physical sheet is the
upper-half-plane boundary, so for real s > 0:  B = 2 - ln(s/mu2) + i*pi.

Note on the massless Feynman-parameter integral: x(1-x) factorizes out of the log, so
quadrature mostly validates the constant 2 and the branch handling. The nontrivial,
independent evidence comes from the momentum-space, unitarity and dispersion checks.
"""

from __future__ import annotations

import cmath
import math

import mpmath as mp
from scipy.integrate import quad

from ..conventions import DEFAULT_MU2


def _check_mu2(mu2: float) -> None:
    if not math.isfinite(mu2) or mu2 <= 0.0:
        raise ValueError("mu2 must be finite and positive.")


def bubble_analytic(s: complex, mu2: float = DEFAULT_MU2, side: int = +1) -> complex:
    """Closed-form bubble on the physical sheet.

    ``side=+1`` selects the upper-half-plane boundary value on the real axis
    (+i*pi for s > 0); ``side=-1`` the lower boundary (-i*pi). Off the real axis
    ``side`` is irrelevant.
    """
    _check_mu2(mu2)
    if side not in (-1, +1):
        raise ValueError("side must be +1 or -1.")
    z = complex(s)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise ValueError("s must be finite.")
    if z.imag == 0.0:
        if z.real == 0.0:
            raise ValueError("s = 0 is singular for the massless bubble.")
        if z.real < 0.0:
            return complex(2.0 - math.log(-z.real / mu2), 0.0)
        return complex(2.0 - math.log(z.real / mu2), side * math.pi)
    return complex(2.0 - cmath.log(-z / mu2))


def bubble_discontinuity(s: float, mu2: float = DEFAULT_MU2) -> complex:
    """B(s + i0) - B(s - i0): equals 2*pi*i for s > 0 and 0 for s < 0."""
    if s < 0.0:
        return 0j
    return bubble_analytic(s, mu2, side=+1) - bubble_analytic(s, mu2, side=-1)


def bubble_real_part_quad(s: float, mu2: float = DEFAULT_MU2) -> float:
    """Real part by scipy quadrature: -Int_0^1 dx ln[x(1-x)|s|/mu2] (real s != 0)."""
    _check_mu2(mu2)
    if s == 0.0 or not math.isfinite(s):
        raise ValueError("s must be finite and nonzero.")
    a = abs(s) / mu2

    def integrand(x: float) -> float:
        y = x * (1.0 - x) * a
        return -math.log(y) if y > 0.0 else 0.0

    value, _ = quad(integrand, 0.0, 1.0, limit=200)
    return float(value)


def feynman_integral_mp(s_mp: mp.mpc, mu2_mp: mp.mpf) -> mp.mpc:
    """Defining Feynman-parameter integral with the principal complex log (mpmath)."""

    def integrand(x):
        return -mp.log(-x * (1 - x) * s_mp / mu2_mp)

    return mp.quad(integrand, [0, mp.mpf("0.5"), 1])


def bubble_numeric_mpmath(
    s: complex,
    mu2: float = DEFAULT_MU2,
    eta: float = 0.0,
    precision: int = 30,
) -> complex:
    """Numerical evaluation at s + i*eta (a genuine complex continuation).

    ``eta`` may be negative (lower side). On the cut (real s > 0) a nonzero ``eta`` is
    mandatory: no imaginary part is ever hard-coded.
    """
    _check_mu2(mu2)
    z = complex(s)
    im = z.imag + eta
    if im == 0.0 and z.real > 0.0:
        raise ValueError("On the cut (real s > 0) a nonzero eta is required.")
    with mp.workdps(precision):
        result = feynman_integral_mp(mp.mpc(z.real, im), mp.mpf(mu2))
        return complex(mp.re(result), mp.im(result))
