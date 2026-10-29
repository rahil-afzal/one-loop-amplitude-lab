"""The consistency checks for the phi^4 task.

Each check sees the candidate only through ev(points) with points (s, t, u, c4, mu2), and gets a
Sampler. Points are random: wide ranges of scale, couplings of both signs, mu2 over several
decades, a fresh seed per submission (the seed goes to the log, never to the agent). A check
returns (ok, detail); detail["worst_ratio"] is error / tolerance, so ok means ratio < 1.
Details are for the experimenter and are never shown to the agent.

All kinematic points are complex, or approach the real axis from above, so no check depends
on how a candidate treats the branch cut on the real axis.
"""

import cmath
import itertools
import math
import random

from . import physics, reference

PREF = 1.0 / (32.0 * math.pi**2)


def cut_distance(z):
    """Distance from z to the cut along the positive real axis."""
    return abs(z.imag) if z.real >= 0 else abs(z)


class Sampler:
    def __init__(self, seed, name=""):
        self.rng = random.Random(f"{seed}/{name}")

    def strata(self, n, lo, hi):
        """n log-uniform numbers, one per equal log-width stratum, shuffled (covers the range)."""
        a, b = math.log(lo), math.log(hi)
        vals = [math.exp(a + (b - a) * (i + self.rng.random()) / n) for i in range(n)]
        self.rng.shuffle(vals)
        return vals

    def phase(self, avoid):
        return self.rng.uniform(avoid, 2 * math.pi - avoid)

    def kinematics(self, n, lo=0.01, hi=30.0):
        """n complex (s, t, u), s+t+u=0, all three of one common scale drawn from [lo, hi]."""
        out = []
        for r in self.strata(n, lo, hi):
            while True:
                s = r * self.rng.uniform(0.3, 1.5) * cmath.exp(1j * self.phase(0.2))
                t = r * self.rng.uniform(0.3, 1.5) * cmath.exp(1j * self.phase(0.2))
                u = -s - t
                if abs(u) >= 0.1 * r and cut_distance(u) >= 0.1 * abs(u):
                    out.append((s, t, u))
                    break
        return out

    def couplings(self, n, lo=0.05, hi=3.0):
        """n couplings, magnitudes spread over [lo, hi], signs mixed."""
        mags = self.strata(n, lo, hi)
        return [m * (1 if i % 2 == 0 else -1) for i, m in enumerate(mags)]

    def mu2s(self, n):
        return self.strata(n, 1e-3, 1e3)


def _bundle(sm, n, **kw):
    return list(zip(sm.kinematics(n, **kw), sm.couplings(n), sm.mu2s(n), strict=True))


def crossing(ev, sm):
    pts = []
    for (s, t, u), c, m in _bundle(sm, 6):
        pts += [(*perm, c, m) for perm in itertools.permutations((s, t, u))]
    vals = ev(pts)
    worst = 0.0
    for i in range(0, len(vals), 6):
        ref = vals[i]
        for v in vals[i + 1 : i + 6]:
            worst = max(worst, abs(v - ref) / (1e-9 * max(1.0, abs(ref))))
    return worst < 1, {"worst_ratio": worst}


def real_analyticity(ev, sm):
    bundle = _bundle(sm, 6)
    direct = ev([(s, t, u, c, m) for (s, t, u), c, m in bundle])
    mirror = ev([(s.conjugate(), t.conjugate(), u.conjugate(), c, m) for (s, t, u), c, m in bundle])
    worst = max(
        abs(b - a.conjugate()) / (1e-9 * max(1.0, abs(a)))
        for a, b in zip(direct, mirror, strict=True)
    )
    return worst < 1, {"worst_ratio": worst}


def coupling_dependence(ev, sm):
    eps = 1e-4
    worst = 0.0
    for (s, t, u), c0, m in _bundle(sm, 4, lo=0.05, hi=20.0):
        c0 = max(0.1, min(abs(c0), 2.0))
        a_p, a_m, a_1, a_n1, a_2 = ev([(s, t, u, c, m) for c in (eps, -eps, c0, -c0, 2 * c0)])
        tree = max(abs(a_p / eps - 1), abs(a_m / (-eps) - 1)) / 1e-3
        scale = 1e-8 * PREF * c0**2
        l_1, l_n1, l_2 = a_1 - c0, a_n1 + c0, a_2 - 2 * c0  # loop parts, should be c^2 * f
        even = abs(l_1 - l_n1) / scale
        quad = abs(l_2 - 4 * l_1) / scale
        worst = max(worst, tree, even, quad)
    return worst < 1, {"worst_ratio": worst}


def scale_covariance(ev, sm):
    bundle = _bundle(sm, 6)
    lams = sm.strata(6, 0.1, 20.0)
    base = ev([(s, t, u, c, m) for (s, t, u), c, m in bundle])
    scaled = ev([
        (lam * s, lam * t, lam * u, c, lam * m)
        for ((s, t, u), c, m), lam in zip(bundle, lams, strict=True)
    ])  # fmt: skip
    worst = max(abs(a - b) / (1e-9 * max(1.0, abs(a))) for a, b in zip(base, scaled, strict=True))
    return worst < 1, {"worst_ratio": worst}


def unitarity(ev, sm):
    n = 6
    worst = 0.0
    for s, c, m in zip(sm.strata(n, 0.01, 100.0), sm.couplings(n), sm.mu2s(n), strict=True):
        t = -s * sm.rng.uniform(0.03, 0.97)
        eta = 1e-9 * s
        (v,) = ev([(s + 1j * eta, t, -s - t - 1j * eta, c, m)])
        expected = physics.cut_im_part(c)
        worst = max(worst, abs(v.imag - expected) / (1e-6 * expected))
    return worst < 1, {"worst_ratio": worst}


def dispersion(ev, sm):
    """Second s-derivative at fixed t against a twice-subtracted dispersion integral.

    A''(s) = (2/pi) Im A * Int_0^inf dx [1/(x - s)^3 + 1/(x - u)^3], and the integral of
    1/(x - z)^3 over the positive axis is 1/(2 z^2). The Im A from the cut, the cuts being in s
    and u only. Additive constants (the scheme) drop out of A''.
    """
    worst = 0.0
    for r, c, m in zip(sm.strata(4, 0.2, 5.0), sm.couplings(4, 0.3, 3.0), sm.mu2s(4), strict=True):
        s0 = r * sm.rng.uniform(0.6, 1.4) * cmath.exp(1j * sm.phase(0.45))
        u0 = r * sm.rng.uniform(0.6, 1.4) * cmath.exp(1j * sm.phase(0.45))
        t0 = -s0 - u0
        h = 0.01 * min(cut_distance(s0), cut_distance(u0))
        pts = [(s0 + k * h, t0, -(s0 + k * h) - t0, c, m) for k in (-2, -1, 0, 1, 2)]
        f = ev(pts)
        second = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h * h)
        scale = (
            (2 / math.pi)
            * physics.cut_im_part(c)
            * (1 / (2 * abs(s0) ** 2) + 1 / (2 * abs(u0) ** 2))
        )
        expected = (2 / math.pi) * physics.cut_im_part(c) * (1 / (2 * s0**2) + 1 / (2 * u0**2))
        worst = max(worst, abs(second - expected) / (1e-6 * scale))
    return worst < 1, {"worst_ratio": worst}


def rg_beta(ev, sm):
    h = 1e-3
    worst = 0.0
    for (s, t, u), c, m in _bundle(sm, 4):
        c = math.copysign(max(abs(c), 0.2), c)
        up, down = ev([(s, t, u, c, m * math.exp(h)), (s, t, u, c, m * math.exp(-h))])
        # d/d ln(mu) = 2 d/d ln(mu2); the explicit mu dependence must cancel beta * dA/dc4 = beta
        explicit = 2 * (up - down) / (2 * h)
        expected = -physics.BETA_COEFF * c**2
        worst = max(worst, abs(explicit - expected) / (1e-6 * expected))
    return worst < 1, {"worst_ratio": worst}


def reference_match(ev, sm):
    bundle = _bundle(sm, 8)
    pts = [(s, t, u, c, m) for (s, t, u), c, m in bundle]
    worst = 0.0
    for p, v in zip(pts, ev(pts), strict=True):
        ref = reference.amplitude(*p)
        tol = 1e-6 * abs(ref - p[3]) + 1e-13 * abs(p[3])  # relative to the one-loop part
        worst = max(worst, abs(v - ref) / tol)
    return worst < 1, {"worst_ratio": worst}


SUITES = {
    "phi4_dim4": [
        ("crossing", crossing,
         "the amplitude is not symmetric under permutations of s, t, u"),
        ("real_analyticity", real_analyticity,
         "A(s*, t*, u*) is not the complex conjugate of A(s, t, u) away from the real axis"),
        ("coupling_dependence", coupling_dependence,
         "the c4 dependence is not tree level (c4) plus a one-loop piece even in c4 and "
         "proportional to c4^2"),
        ("scale_covariance", scale_covariance,
         "dimensional analysis fails: the amplitude should depend on s, t, u, mu2 only through "
         "ratios"),
        ("unitarity", unitarity,
         "the imaginary part in the physical region does not match the tree-level two-particle "
         "cut"),
        ("dispersion", dispersion,
         "the analytic structure in s at fixed t is inconsistent with the cut (check the log "
         "terms)"),
        ("rg_beta", rg_beta,
         "the mu dependence is not consistent with the one-loop running of c4"),
        ("reference_match", reference_match,
         "the amplitude does not match the MS-bar reference"),
    ],
}  # fmt: skip
