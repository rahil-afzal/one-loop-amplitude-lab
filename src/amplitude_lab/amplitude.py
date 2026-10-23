"""Assembly of the supported one-loop 2 -> 2 amplitude.

    A_tree   = c4                                   (one contact diagram)
    A_1loop  = (c4^2 + c42^2)/(32 pi^2) [B(s) + B(t) + B(u)]

All constants come from ``conventions`` and are read at call time so the validation
suite can be run against deliberately broken conventions (negative controls).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import conventions
from .integrals.bubble import bubble_analytic
from .kinematics import Kinematics
from .problem import ProblemSpec


def _ri(z: complex) -> dict:
    return {"re": z.real, "im": z.imag}


@dataclass
class AmplitudeResult:
    tree: complex
    one_loop: complex
    channels: dict[str, complex]
    metadata: dict = field(default_factory=dict)

    @property
    def total(self) -> complex:
        return self.tree + self.one_loop

    def to_dict(self) -> dict:
        return {
            "tree": _ri(self.tree),
            "one_loop": _ri(self.one_loop),
            "total": _ri(self.total),
            "channels": {k: _ri(v) for k, v in self.channels.items()},
            "metadata": self.metadata,
        }


def tree_amplitude(problem: ProblemSpec) -> complex:
    return complex(problem.c4, 0.0)


def bubble_channels(kin: Kinematics, mu2: float) -> dict[str, complex]:
    return {
        name: bubble_analytic(value, mu2)
        for name, value in zip(("s", "t", "u"), kin.as_tuple(), strict=True)
    }


def one_loop_amplitude(problem: ProblemSpec, kin: Kinematics) -> tuple[complex, dict[str, complex]]:
    channels = bubble_channels(kin, problem.mu2)
    prefactor = conventions.ONE_LOOP_PREFACTOR * problem.coupling_sq
    return complex(prefactor * sum(channels.values(), 0j)), channels


def assemble(problem: ProblemSpec, kin: Kinematics) -> AmplitudeResult:
    loop, channels = one_loop_amplitude(problem, kin)
    return AmplitudeResult(
        tree=tree_amplitude(problem),
        one_loop=loop,
        channels=channels,
        metadata={
            "one_loop_prefactor": conventions.ONE_LOOP_PREFACTOR,
            "coupling_sq": problem.coupling_sq,
            "mu2": problem.mu2,
            "scheme": problem.scheme,
        },
    )


def forward_one_loop_imag(problem: ProblemSpec, s: float) -> float:
    """Imaginary part of the ASSEMBLED one-loop amplitude at near-forward kinematics.

    Evaluated through ``assemble`` (not a closed-form shortcut), so the optical-theorem
    check genuinely exercises the loop code. The real part diverges as t -> 0 and is
    never returned.
    """
    return assemble(problem, Kinematics.forward(s)).one_loop.imag
