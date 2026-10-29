import re

import pytest

from amplitude_lab import reference
from amplitude_lab.candidate import Candidate
from amplitude_lab.verify import verify

from .conftest import CANDIDATES

SEEDS = range(10)


def trusted(name):
    return Candidate(path=str(CANDIDATES / f"{name}.py"), isolated=False)


def isolated(name, timeout=60):
    return Candidate(path=str(CANDIDATES / f"{name}.py"), isolated=True, timeout=timeout)


def fired(name, seeds=SEEDS):
    """For each seed, the set of checks that fail; returns the list of sets."""
    c = trusted(name)
    return [{r.name for r in verify(c, seed=s).results if not r.ok} for s in seeds]


# ---- the right answers pass, for every seed ------------------------------------------------


def test_reference_passes_many_seeds_with_margin():
    c = Candidate.from_function(reference.amplitude)
    for seed in range(200):
        report = verify(c, seed=seed)
        assert report.passed, (seed, report.feedback())
        for r in report.results:
            # every tolerance has at least a factor 10 of room on the reference itself
            assert r.detail["worst_ratio"] < 0.1, (seed, r.name, r.detail)


@pytest.mark.parametrize("name", ["correct", "numeric_quad"])
def test_legitimate_solutions_pass(name):
    assert all(not f for f in fired(name))


def test_numerically_too_coarse_solution_is_rejected():
    assert all(f for f in fired("low_accuracy_quad"))


# ---- wrong answers are caught, for every seed ----------------------------------------------

# checks that must fire on every seed (more may fire; reference_match always does)
MUST_FIRE = {
    "wrong_factor": {"unitarity", "dispersion", "rg_beta"},
    "wrong_sign_loop": {"unitarity", "dispersion", "rg_beta"},
    "tree_times_three": {"coupling_dependence"},
    "missing_channel": {"crossing", "dispersion", "rg_beta"},
    "double_t_channel": {"crossing", "rg_beta"},
    "no_mu": {"scale_covariance", "rg_beta"},
    "wrong_branch": {"unitarity"},
    "wrong_ipi": {"real_analyticity", "unitarity", "dispersion"},
    "wrong_constant": set(),
    # wrong only outside the region the first version of the verifier sampled
    "abs_coupling": {"coupling_dependence"},
    "abs_tree": {"coupling_dependence"},
    "drops_large": {"unitarity"},
    "zero_small": {"unitarity"},
    "mu_bump": set(),
}


@pytest.mark.parametrize("name", sorted(MUST_FIRE))
def test_wrong_candidates_are_caught_on_every_seed(name):
    for seed, f in zip(SEEDS, fired(name), strict=True):
        assert MUST_FIRE[name] | {"reference_match"} <= f, (name, seed, f)


def test_wrong_constant_trips_only_the_reference():
    for f in fired("wrong_constant"):
        assert f == {"reference_match"}
    report = verify(trusted("wrong_constant"), seed=1)
    assert report.structural_ok and not report.passed
    hint = report.results[-1].hint
    assert "every other check passes" in hint and "numerical accuracy" in hint


def test_negative_couplings_are_exercised():
    # the blind spot found in review: loop ~ c4*|c4| used to pass everything
    assert all("coupling_dependence" in f for f in fired("abs_coupling"))


# ---- determinism and bookkeeping -----------------------------------------------------------


def test_same_seed_same_result_different_seed_different_points():
    c = trusted("correct")
    a, b, other = verify(c, seed=7), verify(c, seed=7), verify(c, seed=8)
    assert a.to_dict() == b.to_dict()
    assert [r.detail for r in a.results] != [r.detail for r in other.results]


def test_random_seed_is_recorded_and_not_shown():
    report = verify(trusted("correct"))
    assert isinstance(report.seed, int) and report.to_dict()["seed"] == report.seed
    assert str(report.seed) not in report.feedback(1, 10)


def test_crash_is_reported():
    report = verify(trusted("crashes"))
    assert not report.passed and not report.ran
    assert report.results[0].name == "runs" and "ZeroDivisionError" in report.results[0].hint
    assert report.feedback(1, 10).splitlines()[0].endswith("so no checks were evaluated")


def test_nonfinite_values_are_reported():
    c = Candidate.from_function(lambda s, t, u, c4, mu2: float("nan"))
    assert verify(c).results[0].name == "runs"


def test_feedback_has_names_and_no_decimals():
    text = verify(trusted("wrong_factor"), seed=3).feedback(2, 10)
    assert "FAIL unitarity" in text and "PASS crossing" in text and "submission 2 of 10" in text
    assert not re.search(r"\d+\.\d+|\de[-+]?\d", text)


def test_unknown_suite():
    with pytest.raises(KeyError):
        verify(trusted("correct"), suite="nope")


# ---- isolated mode: the path an agent's code takes -----------------------------------------


def test_isolated_runs_the_correct_example():
    assert verify(isolated("correct")).passed


def test_isolated_catches_a_wrong_example():
    assert "unitarity" in {r.name for r in verify(isolated("wrong_factor")).results if not r.ok}


def test_forbidden_import_is_blocked():
    report = verify(isolated("forbidden_import", 30))
    assert report.results[0].name == "runs" and "not allowed" in report.results[0].hint


def test_timeout():
    report = verify(isolated("hangs", 2))
    assert report.results[0].name == "runs" and "timed out" in report.results[0].hint


@pytest.mark.parametrize("name", ["reads_reference", "reads_reference_file", "forges_output"])
def test_attacks_on_the_answer_key_and_verifier_fail(name):
    # these passed 8 of 8 checks (reads_reference) before the sandbox was hardened
    report = verify(isolated(name, 60))
    assert not report.passed and report.results[0].name == "runs", report.feedback()


def test_exception_text_does_not_carry_the_test_points():
    from amplitude_lab.checks import Sampler

    seed = 5
    report = verify(isolated("leaks_points", 30), seed=seed)
    hint = report.results[0].hint
    assert "ValueError" in hint
    # the point this candidate was called with: none of its numbers may appear
    sm = Sampler(seed, "runs")
    s, t, u = sm.kinematics(1)[0]
    for number in (s.real, s.imag, t.real, t.imag, u.real, u.imag):
        assert f"{abs(number):.4f}" not in hint and f"{abs(number):.3f}" not in hint
    assert not re.search(r"\d+\.\d+", hint)


def test_no_pycache_next_to_the_submitted_file(tmp_path):
    src = tmp_path / "cand.py"
    src.write_text((CANDIDATES / "correct.py").read_text())
    verify(Candidate(path=str(src), isolated=True, timeout=60), seed=1)
    assert not (tmp_path / "__pycache__").exists()
