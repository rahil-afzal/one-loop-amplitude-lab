import math

import pytest

from amplitude_lab import physics, reference


def test_phase_space_from_general_formula():
    # dPhi2 = |p| dOmega / (16 pi^2 sqrt(s)); independent of s for massless legs
    for s in (0.3, 1.0, 40.0):
        p = math.sqrt(s) / 2
        assert 4 * math.pi * p / (16 * math.pi**2 * math.sqrt(s)) == pytest.approx(
            physics.PHASE_SPACE_2
        )


def test_cut_matches_optical_theorem():
    # sigma = (1/2) Int_{-s}^0 dt |A|^2 / (16 pi s^2) = c^2 / (32 pi s);  Im A(s, 0) = s sigma
    c4 = 0.8
    for s in (0.5, 2.0, 9.0):
        sigma = 0.5 * c4**2 * s / (16 * math.pi * s**2)
        assert s * sigma == pytest.approx(physics.cut_im_part(c4), rel=1e-14)


def test_cut_value():
    assert physics.cut_im_part(1.0) == pytest.approx(1 / (32 * math.pi))


def test_reference_imaginary_part_is_the_cut():
    c4, eta = 0.9, 1e-10
    s, t = 3.0, -1.0
    a = reference.amplitude(s + 1j * eta, t, -s - t - 1j * eta, c4, 1.0)
    assert a.imag == pytest.approx(physics.cut_im_part(c4), rel=1e-8)


def test_beta_coefficient_matches_reference_mu_dependence():
    # d A / d ln(mu) = -beta at order c4^2 (dA/dc4 = 1 at tree level)
    c4, h = 0.6, 1e-4
    s, t = 1.0 + 0.5j, -0.7 + 0.2j
    a = lambda mu2: reference.amplitude(s, t, -s - t, c4, mu2)  # noqa: E731
    dlnmu = (a(math.exp(2 * h)) - a(math.exp(-2 * h))) / (2 * h)
    assert dlnmu.real == pytest.approx(-physics.BETA_COEFF * c4**2, rel=1e-8)
