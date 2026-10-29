# took log(s) instead of log(-s)
import cmath
import math


def bubble(p2, mu2):
    return 2 - cmath.log(-p2 / mu2)


def amplitude(s, t, u, c4, mu2):
    def b(p2):
        return 2 - cmath.log(p2 / mu2)

    loop = b(s) + b(t) + b(u)
    return c4 + c4**2 / (32 * math.pi**2) * loop
