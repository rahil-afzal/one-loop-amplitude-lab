# numerically integrates the right thing, but with too few nodes to reach 1e-6
import cmath
import math

import mpmath as mp


def bubble(p2, mu2):
    f = lambda x: -cmath.log(-x * (1 - x) * p2 / mu2)  # noqa: E731
    return complex(mp.quad(f, mp.linspace(0, 1, 6), maxdegree=1))


def amplitude(s, t, u, c4, mu2):
    loop = bubble(s, mu2) + bubble(t, mu2) + bubble(u, mu2)
    return c4 + c4**2 / (32 * math.pi**2) * loop
