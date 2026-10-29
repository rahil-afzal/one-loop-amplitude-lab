import cmath
import math

import sympy as sp

# Feynman-parameter integral of the log, -Int_0^1 ln(x(1-x)) dx. Integrate ln(x) only and use
# x -> 1-x for the other half: sympy's antiderivative of ln(1-x) gives 2 - 2*I*pi for the full
# integral (it picks a branch of log(x-1)).
x = sp.symbols("x", positive=True)
CONST = 2 * float(sp.integrate(-sp.log(x), (x, 0, 1)))  # = 2


def bubble(p2, mu2):
    return CONST - cmath.log(-p2 / mu2)


def amplitude(s, t, u, c4, mu2):
    loop = bubble(s, mu2) + bubble(t, mu2) + bubble(u, mu2)
    return c4 + c4**2 / (32 * math.pi**2) * loop
