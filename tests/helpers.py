"""Independent routes to the numbers the reference amplitude relies on."""

import math
from collections import Counter
from fractions import Fraction

from scipy.integrate import quad


def euclid_difference(p1_sq, p2_sq):
    """I(P1^2) - I(P2^2) for I = Int d^4k/pi^2 1/(k^2 (k+P)^2), in 4d spherical coordinates.

    Finite, and equal to ln(P2^2/P1^2). Compare with B(-P1^2) - B(-P2^2): different route from
    the Feynman-parameter form the reference uses.
    """
    p1, p2 = math.sqrt(p1_sq), math.sqrt(p2_sq)

    def angular(k, p):
        f = lambda th: math.sin(th) ** 2 / (k * k + p * p + 2 * k * p * math.cos(th))  # noqa: E731
        return (2 / math.pi) * quad(f, 0, math.pi, epsabs=1e-13, epsrel=1e-13, limit=200)[0]

    def radial(k):
        return 2 * k * (angular(k, p1) - angular(k, p2)) if k > 0 else 0.0

    return quad(radial, 0, 2 * max(p1, p2), points=sorted({p1, p2}), epsabs=1e-12, limit=200)[0]


def wick_coefficients(n_vertices):
    """Wick contractions for 4 external legs and n phi^4 vertices, divided by n! (4!)^n.

    Returns {class: coefficient}. Classes: 'contact' (one vertex), 's'/'t'/'u' (two vertices,
    legs paired as 01|23, 02|13, 03|12). Self-contractions at one vertex are dropped
    (massless tadpoles).
    """
    nodes = [("ext", i, -1) for i in range(4)]
    nodes += [("fld", j, v) for v in range(n_vertices) for j in range(4)]
    counts = Counter()

    def ok(a, b):
        if a[0] == "ext" and b[0] == "ext":
            return False
        return not (a[0] == "fld" and b[0] == "fld" and a[2] == b[2])

    def kind(pairs):
        if n_vertices == 1:
            return "contact"
        on = {0: set(), 1: set()}
        for a, b in pairs:
            if a[0] == "ext":
                on[b[2]].add(a[1])
            elif b[0] == "ext":
                on[a[2]].add(b[1])
        groups = {frozenset(g) for g in on.values()}
        names = {
            frozenset({frozenset({0, 1}), frozenset({2, 3})}): "s",
            frozenset({frozenset({0, 2}), frozenset({1, 3})}): "t",
            frozenset({frozenset({0, 3}), frozenset({1, 2})}): "u",
        }
        return names.get(frozenset(groups), "other")

    def go(rest, pairs):
        if not rest:
            counts[kind(pairs)] += 1
            return
        first, others = rest[0], rest[1:]
        for i, other in enumerate(others):
            if ok(first, other):
                go(others[:i] + others[i + 1 :], pairs + [(first, other)])

    go(nodes, [])
    norm = math.factorial(n_vertices) * 24**n_vertices
    return {k: Fraction(v, norm) for k, v in counts.items()}
