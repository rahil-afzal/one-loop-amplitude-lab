# hand-coded the imaginary part with the wrong sign
import cmath
import math


def bubble(p2, mu2):
    return 2 - cmath.log(-p2 / mu2)


def amplitude(s, t, u, c4, mu2):
    def b(p2):
        return 2 - math.log(abs(p2) / mu2) - (1j * math.pi if p2.real > 0 else 0)

    loop = b(s) + b(t) + b(u)
    return c4 + c4**2 / (32 * math.pi**2) * loop
