import math

import pytest

from amplitude_lab import conventions
from amplitude_lab.amplitude import assemble, forward_one_loop_imag
from amplitude_lab.kinematics import PERMUTATIONS, Kinematics


def test_tree_is_single_contact(problem):
    result = assemble(problem, Kinematics.from_s_t(5.0, -2.0))
    assert result.tree == complex(problem.c4)


def test_loop_matches_hand_computation(problem):
    kin = Kinematics.from_s_t(5.0, -2.0)
    result = assemble(problem, kin)
    expected_re = 6.0 - math.log(5.0) - math.log(2.0) - math.log(3.0)
    expected = problem.coupling_sq / (32.0 * math.pi**2) * complex(expected_re, math.pi)
    assert result.one_loop == pytest.approx(expected, abs=1e-14)


def test_prefactor_decomposition():
    assert pytest.approx(1.0 / (32.0 * math.pi**2)) == conventions.ONE_LOOP_PREFACTOR


def test_crossing(problem):
    kin = Kinematics.from_s_t(5.0, -2.0)
    base = assemble(problem, kin).total
    for order in PERMUTATIONS:
        assert assemble(problem, kin.permuted(order)).total == pytest.approx(base, abs=1e-14)


def test_vanishing_coupling():
    from amplitude_lab.problem import ProblemSpec

    result = assemble(ProblemSpec(name="zero", c4=0.0, c42=0.0), Kinematics.from_s_t(5.0, -2.0))
    assert result.total == 0


def test_c42_enters_only_through_loop(problem):
    kin = Kinematics.from_s_t(5.0, -2.0)
    a = assemble(problem, kin)
    b = assemble(problem.with_(c42=0.0), kin)
    assert a.tree == b.tree
    ratio = a.one_loop / b.one_loop
    assert ratio == pytest.approx(problem.coupling_sq / problem.c4**2)


def test_forward_imaginary_part_is_s_independent(problem):
    expected = problem.coupling_sq / (32.0 * math.pi)
    for s in (0.1, 1.0, 50.0):
        assert forward_one_loop_imag(problem, s) == pytest.approx(expected, rel=1e-12)


def test_result_to_dict_is_json_ready(problem):
    import json

    json.dumps(assemble(problem, Kinematics.from_s_t(5.0, -2.0)).to_dict())
