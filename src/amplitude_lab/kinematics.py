"""Massless 2 -> 2 kinematics and crossing permutations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Kinematics:
    """Mandelstam invariants for four massless particles (s + t + u = 0)."""

    s: float
    t: float
    u: float

    def __post_init__(self) -> None:
        for name in ("s", "t", "u"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        scale = max(1.0, abs(self.s), abs(self.t), abs(self.u))
        if abs(self.s + self.t + self.u) > 1e-9 * scale:
            raise ValueError("Massless kinematics requires s + t + u = 0.")

    @classmethod
    def from_s_t(cls, s: float, t: float) -> Kinematics:
        return cls(s=s, t=t, u=-s - t)

    @classmethod
    def from_s_cos_theta(cls, s: float, cos_theta: float) -> Kinematics:
        """Physical s-channel point: t = -s(1-cos)/2, u = -s(1+cos)/2."""
        if s <= 0.0:
            raise ValueError("Physical s-channel kinematics requires s > 0.")
        if not -1.0 < cos_theta < 1.0:
            raise ValueError("cos(theta) must lie strictly inside (-1, 1).")
        return cls(s=s, t=-s * (1.0 - cos_theta) / 2.0, u=-s * (1.0 + cos_theta) / 2.0)

    @classmethod
    def forward(cls, s: float, rel_eps: float = 1e-6) -> Kinematics:
        """Near-forward point t = -rel_eps * s (strictly inside the physical region).

        Only the imaginary part of the amplitude is finite as t -> 0: the real part
        contains -ln(-t/mu2). Callers must not use a finite real forward amplitude.
        """
        if s <= 0.0:
            raise ValueError("Forward kinematics requires s > 0.")
        if not 0.0 < rel_eps < 1.0:
            raise ValueError("rel_eps must lie in (0, 1).")
        return cls.from_s_t(s, -rel_eps * s)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.s, self.t, self.u)

    @property
    def is_physical_s_channel(self) -> bool:
        return self.s > 0.0 and self.t <= 0.0 and self.u <= 0.0

    def permuted(self, order: tuple[int, int, int]) -> Kinematics:
        values = self.as_tuple()
        return Kinematics(*(values[i] for i in order))

    def scaled(self, factor: float) -> Kinematics:
        if factor <= 0.0:
            raise ValueError("Scaling factor must be positive.")
        return Kinematics(self.s * factor, self.t * factor, self.u * factor)


PERMUTATIONS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)

BENCHMARK_POINTS: tuple[Kinematics, ...] = (
    Kinematics.from_s_t(5.0, -2.0),
    Kinematics.from_s_t(2.0, -1.0),
    Kinematics.from_s_t(0.5, -0.25),
    Kinematics.from_s_t(-1.0, -2.0),
)
