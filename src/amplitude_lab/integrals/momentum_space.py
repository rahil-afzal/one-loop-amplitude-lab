"""Independent momentum-space route to the bubble (Euclidean region).

For Euclidean external momentum P (s = -P^2) the UV-divergent bubble

    I(P^2) = Int d^4k/pi^2  1 / (k^2 (k+P)^2)

obeys I(P1^2) - I(P2^2) = ln(P2^2/P1^2), a finite convergent integral. This must equal
B(s1) - B(s2) from the Feynman-parameter closed form, but is computed by a different
route (4D radial x angular integration, no Feynman parameters).

d^4k = k^3 dk dOmega_3, with Int dOmega_3 = 2 pi^2 and the S^3 angular weight
(2/pi) sin^2(theta) dtheta (normalized to 1). Beyond k > max(P1, P2) the angular
average of 1/(k+P)^2 equals 1/k^2 exactly, so the difference integrand vanishes there.
"""

from __future__ import annotations

import math

from scipy.integrate import quad


def _angular_average(k: float, p: float) -> float:
    """<1/(k+P)^2> over S^3, by numerical quadrature."""

    def integrand(theta: float) -> float:
        return math.sin(theta) ** 2 / (k * k + p * p + 2.0 * k * p * math.cos(theta))

    value, _ = quad(integrand, 0.0, math.pi, epsabs=1e-13, epsrel=1e-13, limit=200)
    return (2.0 / math.pi) * value


def euclidean_bubble_difference_numeric(p1_sq: float, p2_sq: float) -> float:
    """Numerically evaluate I(P1^2) - I(P2^2) (should equal ln(P2^2/P1^2))."""
    if p1_sq <= 0.0 or p2_sq <= 0.0:
        raise ValueError("Euclidean invariants P^2 must be positive.")
    p1, p2 = math.sqrt(p1_sq), math.sqrt(p2_sq)
    upper = 2.0 * max(p1, p2)

    def integrand(k: float) -> float:
        if k <= 0.0:
            return 0.0
        return 2.0 * k * (_angular_average(k, p1) - _angular_average(k, p2))

    value, _ = quad(
        integrand, 0.0, upper, points=sorted({p1, p2}), epsabs=1e-12, epsrel=1e-12, limit=200
    )
    return float(value)
