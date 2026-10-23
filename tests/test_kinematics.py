import pytest

from amplitude_lab.kinematics import PERMUTATIONS, Kinematics


def test_sum_rule():
    kin = Kinematics.from_s_t(5.0, -2.0)
    assert kin.u == -3.0


def test_bad_sum_rule_rejected():
    with pytest.raises(ValueError):
        Kinematics(1.0, 1.0, 1.0)


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        Kinematics.from_s_t(float("nan"), 1.0)


def test_permutations_preserve_multiset():
    kin = Kinematics.from_s_t(5.0, -2.0)
    for order in PERMUTATIONS:
        assert sorted(kin.permuted(order).as_tuple()) == sorted(kin.as_tuple())


def test_physical_point_from_angle():
    kin = Kinematics.from_s_cos_theta(4.0, 0.5)
    assert kin.is_physical_s_channel
    assert kin.t == pytest.approx(-1.0)
    assert kin.u == pytest.approx(-3.0)


def test_angle_bounds():
    with pytest.raises(ValueError):
        Kinematics.from_s_cos_theta(4.0, 1.0)
    with pytest.raises(ValueError):
        Kinematics.from_s_cos_theta(-4.0, 0.0)


def test_forward_stays_in_physical_region_at_any_scale():
    for s in (1e-8, 1.0, 1e8):
        kin = Kinematics.forward(s)
        assert kin.is_physical_s_channel and kin.t < 0 and kin.u < 0


def test_scaling_positive_only():
    with pytest.raises(ValueError):
        Kinematics.from_s_t(1.0, -0.5).scaled(-1.0)
