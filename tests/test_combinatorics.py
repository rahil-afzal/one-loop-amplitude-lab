from fractions import Fraction

import pytest

from amplitude_lab import combinatorics as comb

PHI4 = ("1", "1", "1", "1")
MIXED = ("1", "1", "2", "2")


def test_raw_counts_phi4():
    assert comb.count_contractions(PHI4, PHI4, 1) == {"contact": 24}
    counts = comb.count_contractions(PHI4, PHI4, 2)
    assert counts == {"s": 576, "t": 576, "u": 576}


def test_tree_has_single_contact_diagram():
    assert comb.coefficients(PHI4, PHI4, 1) == {"contact": Fraction(1)}


def test_bubble_symmetry_factor_is_one_half():
    half = Fraction(1, 2)
    assert comb.coefficients(PHI4, PHI4, 2) == {"s": half, "t": half, "u": half}


def test_mixed_vertex_loop_and_definition_of_c42():
    half = Fraction(1, 2)
    assert comb.coefficients(MIXED, PHI4, 2) == {"s": half, "t": half, "u": half}
    assert comb.coefficients(MIXED, MIXED, 1) == {"contact": Fraction(1)}


def test_vertex_normalization():
    assert comb.vertex_normalization(PHI4) == 24
    assert comb.vertex_normalization(MIXED) == 4


def test_unsupported_shapes_rejected():
    with pytest.raises(ValueError):
        comb.count_contractions(PHI4, ("1", "1", "1"), 1)
    with pytest.raises(ValueError):
        comb.count_contractions(PHI4, PHI4, 3)
