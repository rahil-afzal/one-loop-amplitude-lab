import math

import pytest

from amplitude_lab import dispersion
from amplitude_lab.crosssection import (
    im_amplitude_from_unitarity,
    im_bubble_from_unitarity,
    sigma_tree,
    two_body_phase_space_numeric,
)


def test_phase_space_is_one_over_eight_pi():
    assert two_body_phase_space_numeric() == pytest.approx(1.0 / (8.0 * math.pi), rel=1e-14)


@pytest.mark.parametrize("method", ["closed", "dsigma_dt", "phase_space"])
def test_sigma_methods_agree(problem, method):
    assert sigma_tree(problem, 3.0, method) == pytest.approx(
        problem.coupling_sq / (32.0 * math.pi * 3.0), rel=1e-12
    )


def test_sigma_rejects_bad_input(problem):
    with pytest.raises(ValueError):
        sigma_tree(problem, -1.0)
    with pytest.raises(ValueError):
        sigma_tree(problem, 1.0, "magic")


def test_unitarity_gives_plus_pi_for_the_bubble():
    assert im_bubble_from_unitarity() == pytest.approx(math.pi, rel=1e-12)


def test_unitarity_cut_value(problem):
    assert im_amplitude_from_unitarity(problem) == pytest.approx(
        problem.coupling_sq / (32.0 * math.pi), rel=1e-12
    )


@pytest.mark.parametrize("s0", [0.1, 0.25, 1.0, 2.0])
def test_sigma_three_routes(problem, s0):
    reference = dispersion.sigma_reference(problem, s0)
    assert dispersion.sigma_from_cross_section(problem, s0) == pytest.approx(reference, rel=1e-8)
    assert dispersion.sigma_from_amplitude(problem, s0) == pytest.approx(reference, rel=1e-8)
    assert dispersion.sigma_symbolic(problem, s0) == pytest.approx(reference, rel=1e-8)


def test_sigma_value_single_scalar(pure_problem):
    # c4 = 1, s0 = 0.25  ->  1 / (64 pi^2 * 0.0625) = 1 / (4 pi^2) ~ 0.025330
    value = dispersion.sigma_reference(pure_problem, 0.25)
    assert value == pytest.approx(1.0 / (4.0 * math.pi**2))
    assert value == pytest.approx(0.0253302959, rel=1e-8)


@pytest.mark.parametrize("x", [-0.5, -2.0])
def test_bubble_second_derivative(x):
    closed = 1.0 / x**2
    assert dispersion.bubble_second_derivative_numeric(x) == pytest.approx(closed, rel=1e-8)
    assert dispersion.bubble_second_derivative_dispersive(x) == pytest.approx(closed, rel=1e-10)


def test_dispersion_requires_euclidean_point():
    with pytest.raises(ValueError):
        dispersion.bubble_second_derivative_numeric(1.0)
    with pytest.raises(ValueError):
        dispersion.bubble_second_derivative_dispersive(1.0)
