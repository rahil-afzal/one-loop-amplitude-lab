# returns zero when every invariant is small
import cmath
import math


def bubble(p2, mu2):
    return 2 - cmath.log(-p2 / mu2)


def amplitude(s, t, u, c4, mu2):
    if max(abs(s), abs(t), abs(u)) < 0.1:
        return 0j
    loop = bubble(s, mu2) + bubble(t, mu2) + bubble(u, mu2)
    return c4 + c4**2 / (32 * math.pi**2) * loop
