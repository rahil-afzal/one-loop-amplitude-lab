"""Physics checks: optical theorem, unitarity, dispersion relations, RG."""

from __future__ import annotations

import math

from .. import conventions, dispersion
from ..amplitude import assemble, forward_one_loop_imag
from ..crosssection import im_amplitude_from_unitarity, im_bubble_from_unitarity, sigma_tree
from ..kinematics import Kinematics
from ..problem import ProblemSpec
from .base import INDEPENDENT, REFERENCE, Outcome, register, rel_err


@register(
    "physics_optical_theorem",
    INDEPENDENT,
    "Im A_1loop (assembled, forward) = s * sigma_tree, with sigma from dsigma/dt and phase space.",
)
def check_optical_theorem(problem: ProblemSpec) -> Outcome:
    rows = []
    worst = 0.0
    for s in (1.0, 3.0, 7.0):
        im_forward = forward_one_loop_imag(problem, s)
        sigma_dt = sigma_tree(problem, s, "dsigma_dt")
        sigma_ps = sigma_tree(problem, s, "phase_space")
        sigma_closed = sigma_tree(problem, s, "closed")
        err = max(
            rel_err(im_forward, s * sigma_dt),
            rel_err(im_forward, s * sigma_ps),
            rel_err(sigma_dt, sigma_closed),
        )
        worst = max(worst, err)
        rows.append({"s": s, "Im_A_forward": im_forward, "s_sigma_dsigma_dt": s * sigma_dt})
    tol = 1e-10
    return Outcome(
        worst < tol,
        "Loop imaginary part reproduces s*sigma computed from tree amplitudes only.",
        {"points": rows, "worst_relative_error": worst},
        tol,
    )


@register(
    "physics_unitarity_cut",
    INDEPENDENT,
    "Im A_1loop at generic physical angles = two-body cut of tree amplitudes; t,u bubbles real.",
)
def check_unitarity_cut(problem: ProblemSpec) -> Outcome:
    expected = im_amplitude_from_unitarity(problem)
    rows = []
    worst = 0.0
    for s, cos_theta in ((2.0, 0.3), (5.0, -0.7), (0.5, 0.9), (3.0, 0.0)):
        kin = Kinematics.from_s_cos_theta(s, cos_theta)
        result = assemble(problem, kin)
        err = max(
            rel_err(result.one_loop.imag, expected),
            abs(result.channels["t"].imag),
            abs(result.channels["u"].imag),
            abs(result.tree.imag),
        )
        worst = max(worst, err)
        rows.append({"s": s, "cos_theta": cos_theta, "Im_A": result.one_loop.imag})
    tol = 1e-10
    positive = expected > 0.0 or problem.coupling_sq == 0.0
    return Outcome(
        worst < tol and positive,
        "Imaginary part is angle independent, positive, and equals the unitarity cut.",
        {"expected_Im_A": expected, "points": rows, "worst_error": worst},
        tol,
    )


@register(
    "physics_bubble_dispersion",
    INDEPENDENT,
    "B''(x) from the Feynman integral = (2/pi) Int Im B(s')/(s'-x)^3 with Im B from unitarity.",
)
def check_bubble_dispersion(problem: ProblemSpec) -> Outcome:
    rows = []
    worst = 0.0
    for x in (-0.5, -2.0):
        numeric = dispersion.bubble_second_derivative_numeric(x, problem.mu2)
        dispersive = dispersion.bubble_second_derivative_dispersive(x)
        closed = 1.0 / (x * x)
        err = max(rel_err(numeric, closed), rel_err(dispersive, closed))
        worst = max(worst, err)
        rows.append({"x": x, "numeric": numeric, "dispersive": dispersive, "closed": closed})
    im_b = im_bubble_from_unitarity()
    tol = 1e-8
    return Outcome(
        worst < tol and abs(im_b - math.pi) < 1e-10,
        "Unitarity cut and real part are linked by a twice-subtracted dispersion relation.",
        {"points": rows, "Im_B_from_unitarity": im_b, "worst_relative_error": worst},
        tol,
    )


@register(
    "physics_sigma_dispersion_dim4",
    REFERENCE,
    "Dim-4 Sigma: three integration routes vs c^2/(64 pi^2 s0^2).",
)
def check_sigma_dispersion(problem: ProblemSpec) -> Outcome:
    s0 = problem.s0
    reference = dispersion.sigma_reference(problem, s0)
    routes = {
        "cross_section_quad": dispersion.sigma_from_cross_section(problem, s0),
        "assembled_amplitude_quad": dispersion.sigma_from_amplitude(problem, s0),
        "sympy_symbolic": dispersion.sigma_symbolic(problem, s0),
    }
    scale = max(abs(reference), 1e-300)
    errors = {name: abs(value - reference) / scale for name, value in routes.items()}
    tol = 1e-8
    return Outcome(
        max(errors.values()) < tol,
        "Dim-4 Sigma matches the Eq. (3.11) subset. This is not the full Eq. (3.9).",
        {"s0": s0, "reference": reference, "routes": routes, "relative_errors": errors},
        tol,
    )


@register(
    "physics_rg_invariance",
    REFERENCE,
    "mu dA/dmu + beta_c4 = 0 through O(c^2), with the textbook beta function as reference.",
)
def check_rg_invariance(problem: ProblemSpec) -> Outcome:
    kin = Kinematics.from_s_t(5.0, -2.0)
    h = 1e-3  # step in ln(mu); mu2 -> mu2 * exp(+-2h)
    up = assemble(problem.with_(mu2=problem.mu2 * math.exp(2.0 * h)), kin).total
    down = assemble(problem.with_(mu2=problem.mu2 * math.exp(-2.0 * h)), kin).total
    d_a_d_ln_mu = (up - down).real / (2.0 * h)
    beta = conventions.BETA_C4_COEFFICIENT * problem.coupling_sq
    residual = d_a_d_ln_mu + beta  # dA/dc4 = 1 at this order
    tol = 1e-10
    return Outcome(
        abs(residual) < tol,
        "Explicit mu dependence of the loop is cancelled by the running of c4.",
        {"dA_dlnmu": d_a_d_ln_mu, "beta_c4": beta, "residual": residual, "order": "O(c^2)"},
        tol,
    )
