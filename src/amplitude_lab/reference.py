"""Answer key for the phi^4 task. The agent never sees this file (see sandbox.py).

A = c4 + c4^2 / (32 pi^2) * [B(s) + B(t) + B(u)]
B(p2) = 2 - Log(-p2 / mu2)      (MS-bar finite part, principal log)

The physical sheet is the upper half plane in s, so a real positive argument is ambiguous
(+i*pi or -i*pi depending on the side). Callers must give p2 + i*eta; exactly real positive
input raises instead of silently picking a side.
"""

import cmath
import math


def bubble(p2, mu2):
    p2 = complex(p2)
    if p2.imag == 0 and p2.real > 0:
        raise ValueError("p2 lies on the cut; pass p2 + i*eta to choose the side")
    return 2.0 - cmath.log(-p2 / mu2)


def amplitude(s, t, u, c4, mu2):
    loop = bubble(s, mu2) + bubble(t, mu2) + bubble(u, mu2)
    return c4 + c4**2 / (32.0 * math.pi**2) * loop
