"""Brute-force Wick-contraction counting: an independent check of symmetry factors.

For vertices ``L_int = (c/N) * prod(fields)`` with ``N = prod(multiplicity!)`` the
coefficient multiplying ``c^n`` in a connected, amputated amplitude is::

    (# Wick contractions) / (n_vertices! * N^n_vertices)

Self-contractions at a single vertex (tadpoles) are excluded: they vanish for massless
propagators in dimensional regularization. Contractions are classified by which
external legs attach to the same vertex.

Examples (verified by tests): the phi^4 contact term has coefficient 1 (one diagram,
not three channels); each s/t/u bubble has coefficient 1/2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import factorial

_CHANNELS = {
    frozenset({frozenset({0, 1}), frozenset({2, 3})}): "s",
    frozenset({frozenset({0, 2}), frozenset({1, 3})}): "t",
    frozenset({frozenset({0, 3}), frozenset({1, 2})}): "u",
}


@dataclass(frozen=True)
class _Node:
    kind: str  # "ext" or "fld"
    species: str
    vertex: int = -1
    index: int = -1


def vertex_normalization(vertex_species: tuple[str, ...]) -> int:
    """N = prod over species of (multiplicity)!  (4! for phi^4, 2!*2! for phi1^2 phi2^2)."""
    result = 1
    for count in Counter(vertex_species).values():
        result *= factorial(count)
    return result


def count_contractions(
    vertex_species: tuple[str, ...],
    external_species: tuple[str, ...],
    n_vertices: int,
) -> dict[str, int]:
    """Count Wick contractions by class ('contact', 's', 't', 'u' or 'other').

    Only 4 external legs and 1 or 2 identical vertices are supported.
    """
    if len(external_species) != 4:
        raise ValueError("Exactly four external legs are supported.")
    if n_vertices not in (1, 2):
        raise ValueError("Only one or two vertices are supported.")

    nodes = [_Node("ext", sp, index=i) for i, sp in enumerate(external_species)]
    for v in range(n_vertices):
        nodes.extend(_Node("fld", sp, vertex=v, index=j) for j, sp in enumerate(vertex_species))

    def can_pair(a: _Node, b: _Node) -> bool:
        if a.species != b.species:
            return False
        if a.kind == "ext" and b.kind == "ext":
            return False  # disconnected
        return not (a.kind == "fld" and b.kind == "fld" and a.vertex == b.vertex)

    counts: Counter[str] = Counter()

    def classify(pairs: list[tuple[_Node, _Node]]) -> str:
        at_vertex: dict[int, set[int]] = {v: set() for v in range(n_vertices)}
        for a, b in pairs:
            if a.kind == "ext":
                at_vertex[b.vertex].add(a.index)
            elif b.kind == "ext":
                at_vertex[a.vertex].add(b.index)
        if n_vertices == 1:
            return "contact"
        partition = frozenset(frozenset(group) for group in at_vertex.values())
        return _CHANNELS.get(partition, "other")

    def recurse(remaining: list[_Node], pairs: list[tuple[_Node, _Node]]) -> None:
        if not remaining:
            counts[classify(pairs)] += 1
            return
        first, rest = remaining[0], remaining[1:]
        for i, other in enumerate(rest):
            if can_pair(first, other):
                recurse(rest[:i] + rest[i + 1 :], [*pairs, (first, other)])

    recurse(nodes, [])
    return dict(counts)


def coefficients(
    vertex_species: tuple[str, ...],
    external_species: tuple[str, ...],
    n_vertices: int,
) -> dict[str, Fraction]:
    """Convert contraction counts into amplitude coefficients of c^n."""
    norm = vertex_normalization(vertex_species) ** n_vertices
    denominator = factorial(n_vertices) * norm
    return {
        label: Fraction(count, denominator)
        for label, count in count_contractions(vertex_species, external_species, n_vertices).items()
    }
