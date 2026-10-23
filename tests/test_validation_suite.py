"""The suite must pass on the real physics AND fail on deliberately broken physics."""

import math

import pytest

from amplitude_lab import amplitude, conventions
from amplitude_lab.benchmarks import BUILTIN
from amplitude_lab.integrals.bubble import bubble_analytic
from amplitude_lab.validation import list_checks, run_all
from amplitude_lab.validation.base import EVIDENCE_KINDS, REGISTRY, Outcome, register


@pytest.mark.parametrize("name", list(BUILTIN))
def test_all_checks_pass(name):
    failures = [(r.name, r.message, r.details) for r in run_all(BUILTIN[name]()) if not r.passed]
    assert not failures, failures


def test_passes_for_other_parameter_values(problem):
    other = problem.with_(c4=-1.3, c42=2.0, mu2=7.5, s0=0.3)
    assert all(r.passed for r in run_all(other))


def test_registry_is_well_formed():
    names = [spec.name for spec in list_checks()]
    assert len(names) == len(set(names)) >= 10
    assert all(spec.evidence in EVIDENCE_KINDS for spec in list_checks())
    kinds = {spec.name: spec.evidence for spec in list_checks()}
    assert kinds["physics_optical_theorem"] == "independent"
    assert kinds["physics_unitarity_cut"] == "independent"
    assert kinds["integral_momentum_space"] == "independent"
    assert kinds["combinatorics_wick_factors"] == "independent"


def test_only_selects_and_validates_names(problem):
    results = run_all(problem, only=["amplitude_crossing"])
    assert [r.name for r in results] == ["amplitude_crossing"]
    with pytest.raises(ValueError):
        run_all(problem, only=["not_a_check"])


def test_crashing_check_is_reported_as_error_not_pass(problem):
    @register("zz_temp_crash", "consistency", "temporary")
    def _crash(_problem):
        raise RuntimeError("boom")

    try:
        result = next(r for r in run_all(problem, only=["zz_temp_crash"]))
        assert result.status == "error" and "boom" in result.message
    finally:
        REGISTRY.pop()


def test_failing_check_is_reported_as_fail(problem):
    @register("zz_temp_fail", "consistency", "temporary")
    def _fail(_problem):
        return Outcome(False, "nope")

    try:
        assert run_all(problem, only=["zz_temp_fail"])[0].status == "fail"
    finally:
        REGISTRY.pop()


def _failed(problem):
    return {r.name for r in run_all(problem) if not r.passed}


# ---- negative controls: the validators must have teeth --------------------------------


def test_negative_control_wrong_loop_normalization(problem, monkeypatch):
    """A prefactor 1/(16 pi^2) (symmetry factor forgotten) must be caught."""
    monkeypatch.setattr(conventions, "ONE_LOOP_PREFACTOR", 1.0 / (16.0 * math.pi**2))
    failed = _failed(problem)
    assert {
        "physics_optical_theorem",
        "physics_unitarity_cut",
        "physics_rg_invariance",
        "physics_bubble_dispersion",
        "combinatorics_wick_factors",
        "integral_symmetry_prefactor_link",
    } <= failed


def test_negative_control_wrong_sign_of_imaginary_part(problem, monkeypatch):
    """B with -i*pi on the cut violates unitarity (Im A < 0) and must be caught."""
    monkeypatch.setattr(
        amplitude, "bubble_analytic", lambda s, mu2: bubble_analytic(s, mu2, side=-1)
    )
    failed = _failed(problem)
    assert {"physics_optical_theorem", "physics_unitarity_cut"} <= failed
    assert "amplitude_crossing" not in failed  # symmetric either way


def test_negative_control_tree_summed_over_three_channels(problem, monkeypatch):
    """The classic error: tree = 3*c4 (three channels). Wick enumeration catches it."""
    monkeypatch.setattr(amplitude, "tree_amplitude", lambda p: complex(3.0 * p.c4))
    assert "combinatorics_wick_factors" in _failed(problem)


def test_negative_control_wrong_beta_function(problem, monkeypatch):
    monkeypatch.setattr(conventions, "BETA_C4_COEFFICIENT", -6.0 / (16.0 * math.pi**2))
    assert _failed(problem) == {"physics_rg_invariance"}


def test_negative_control_dropped_phi2_loop(problem, monkeypatch):
    """Using c4^2 instead of c4^2 + c42^2 in the loop must break the unitarity cut."""

    def c4_only(p, kin):
        channels = amplitude.bubble_channels(kin, p.mu2)
        value = conventions.ONE_LOOP_PREFACTOR * p.c4**2 * sum(channels.values(), 0j)
        return complex(value), channels

    monkeypatch.setattr(amplitude, "one_loop_amplitude", c4_only)
    failed = _failed(problem)
    assert {"physics_optical_theorem", "physics_unitarity_cut", "physics_rg_invariance"} <= failed
