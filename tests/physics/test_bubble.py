import math

import pytest

from amplitude_lab.integrals.bubble import (
    bubble_analytic,
    bubble_discontinuity,
    bubble_numeric_mpmath,
    bubble_real_part_quad,
)
from amplitude_lab.integrals.momentum_space import euclidean_bubble_difference_numeric


def test_euclidean_closed_form():
    value = bubble_analytic(-4.0)
    assert value.real == pytest.approx(2.0 - math.log(4.0), abs=1e-14)
    assert value.imag == 0.0


def test_timelike_sign_is_plus_pi():
    value = bubble_analytic(4.0)
    assert value.real == pytest.approx(2.0 - math.log(4.0), abs=1e-14)
    assert value.imag == pytest.approx(math.pi, abs=1e-14)


def test_lower_side_is_minus_pi():
    assert bubble_analytic(4.0, side=-1).imag == pytest.approx(-math.pi)


def test_discontinuity():
    assert bubble_discontinuity(4.0) == pytest.approx(2j * math.pi)
    assert bubble_discontinuity(-4.0) == 0j


def test_mu2_dependence_is_additive_log():
    shift = bubble_analytic(-3.0, mu2=2.0) - bubble_analytic(-3.0, mu2=1.0)
    assert shift.real == pytest.approx(math.log(2.0))


def test_known_value():
    assert bubble_analytic(-4.0, 1.0).real == pytest.approx(0.6137056388801094, abs=1e-14)


@pytest.mark.parametrize("s", [-5.0, -1.0, 2.0, 7.0])
def test_scipy_real_part(s):
    assert bubble_real_part_quad(s) == pytest.approx(bubble_analytic(s).real, abs=1e-10)


@pytest.mark.parametrize("s", [-4.0, -0.3, 2 + 1j, -1 - 2j, 0.5 - 0.3j])
def test_mpmath_matches_analytic_off_cut(s):
    assert bubble_numeric_mpmath(s) == pytest.approx(bubble_analytic(s), abs=1e-12)


def test_continuation_is_exact_principal_log():
    s, eta = 4.0, 1e-4
    up = bubble_numeric_mpmath(s, eta=eta)
    down = bubble_numeric_mpmath(s, eta=-eta)
    assert up.imag == pytest.approx(math.pi - math.atan(eta / s), abs=1e-10)
    assert down.imag == pytest.approx(-(math.pi - math.atan(eta / s)), abs=1e-10)


def test_continuation_converges_monotonically():
    errors = [abs(bubble_numeric_mpmath(4.0, eta=e).imag - math.pi) for e in (1e-2, 1e-3, 1e-4)]
    assert errors[0] > errors[1] > errors[2]


def test_cut_requires_eta():
    with pytest.raises(ValueError):
        bubble_numeric_mpmath(4.0, eta=0.0)


def test_momentum_space_difference():
    numeric = euclidean_bubble_difference_numeric(0.5, 3.0)
    assert numeric == pytest.approx(math.log(3.0 / 0.5), abs=1e-8)
    assert numeric == pytest.approx((bubble_analytic(-0.5) - bubble_analytic(-3.0)).real, abs=1e-8)


def test_momentum_space_antisymmetric():
    a = euclidean_bubble_difference_numeric(1.0, 4.0)
    b = euclidean_bubble_difference_numeric(4.0, 1.0)
    assert a == pytest.approx(-b, abs=1e-9)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bubble_analytic(0.0)
    with pytest.raises(ValueError):
        bubble_analytic(1.0, mu2=0.0)
    with pytest.raises(ValueError):
        bubble_analytic(1.0, side=0)
    with pytest.raises(ValueError):
        bubble_analytic(float("nan"))
    with pytest.raises(ValueError):
        euclidean_bubble_difference_numeric(-1.0, 1.0)
