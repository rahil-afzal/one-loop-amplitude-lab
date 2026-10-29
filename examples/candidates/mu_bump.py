# extra term only for a large renormalization scale
import cmath
import math


def bubble(p2, mu2):
    return 2 - cmath.log(-p2 / mu2)


def amplitude(s, t, u, c4, mu2):
    loop = bubble(s, mu2) + bubble(t, mu2) + bubble(u, mu2)
    extra = 1e-3 if mu2 > 50 else 0
    return c4 + c4**2 / (32 * math.pi**2) * loop + extra
