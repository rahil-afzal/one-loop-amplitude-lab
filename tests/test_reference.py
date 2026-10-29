"""The answer key has to be right before anything is judged against it."""

import cmath
import math
from fractions import Fraction

import mpmath as mp
import pytest

from amplitude_lab import reference

from .helpers import euclid_difference, wick_coefficients


def feynman_integral(p2, mu2):
    # defining integral, evaluated numerically with the principal log of the product
    mp.mp.dps = 25
    value = -mp.quad(lambda x: mp.log(-x * (1 - x) * p2 / mu2), [0, 0.5, 1])
    return complex(value)


@pytest.mark.parametrize("p2", [-4.0, -0.3, 2 + 1j, -1 - 2j, 0.5 - 0.3j, 7 + 0.01j])
def test_bubble_matches_defining_integral(p2):
    assert reference.bubble(p2, 1.7) == pytest.approx(feynman_integral(p2, 1.7), abs=1e-12)


def test_exactly_real_positive_input_is_refused():
    # the side of the cut is a choice the caller has to make; silently picking -i*pi was a bug
    with pytest.raises(ValueError, match="cut"):
        reference.bubble(4.0, 1.0)
    with pytest.raises(ValueError):
        reference.amplitude(4.0, -1.0, -3.0, 1.0, 1.0)
    assert reference.bubble(4.0 + 1e-12j, 1.0).imag == pytest.approx(math.pi, abs=1e-9)
    assert reference.bubble(4.0 - 1e-12j, 1.0).imag == pytest.approx(-math.pi, abs=1e-9)


def test_known_value():
    assert reference.bubble(-4.0, 1.0) == pytest.approx(2 - math.log(4), abs=1e-14)


def test_discontinuity_is_two_pi_i():
    eta = 1e-12
    up = reference.bubble(4 + 1j * eta, 1.0)
    down = reference.bubble(4 - 1j * eta, 1.0)
    assert up.imag == pytest.approx(math.pi, abs=1e-9)
    assert (up - down) == pytest.approx(2j * math.pi, abs=1e-9)


def test_momentum_space_difference():
    # B(s1) - B(s2) with s = -P^2 against the 4d momentum-space integral
    for p1, p2 in [(0.5, 3.0), (1.0, 7.0), (0.2, 0.9)]:
        closed = (reference.bubble(-p1, 1.0) - reference.bubble(-p2, 1.0)).real
        assert euclid_difference(p1, p2) == pytest.approx(closed, abs=1e-8)
        assert closed == pytest.approx(math.log(p2 / p1))


def test_wick_counts():
    assert wick_coefficients(1) == {"contact": Fraction(1)}
    half = Fraction(1, 2)
    assert wick_coefficients(2) == {"s": half, "t": half, "u": half}


def test_prefactor_is_wick_factor_over_16_pi_squared():
    half = float(wick_coefficients(2)["s"])  # symmetry factor of one bubble
    s, t = 5.0 + 0.1j, -2.0 + 0.3j
    u = -s - t
    loop = reference.amplitude(s, t, u, 1.0, 1.0) - 1.0
    bubbles = sum(reference.bubble(x, 1.0) for x in (s, t, u))
    assert loop == pytest.approx(half / (16 * math.pi**2) * bubbles, rel=1e-14)


def test_amplitude_is_real_analytic():
    s, t = 1.3 + 0.7j, -0.4 + 0.2j
    a = reference.amplitude(s, t, -s - t, 0.8, 1.1)
    b = reference.amplitude(s.conjugate(), t.conjugate(), (-s - t).conjugate(), 0.8, 1.1)
    assert b == pytest.approx(a.conjugate(), abs=1e-14)
    assert cmath.isfinite(a)
