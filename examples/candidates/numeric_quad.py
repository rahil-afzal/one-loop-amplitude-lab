# a legitimate alternative: evaluate the Feynman-parameter integral numerically with mpmath
import math

import mpmath as mp


def bubble(p2, mu2):
    return complex(mp.quad(lambda x: -mp.log(-x * (1 - x) * p2 / mu2), [0, 0.5, 1]))


def amplitude(s, t, u, c4, mu2):
    loop = bubble(s, mu2) + bubble(t, mu2) + bubble(u, mu2)
    return c4 + c4**2 / (32 * math.pi**2) * loop
