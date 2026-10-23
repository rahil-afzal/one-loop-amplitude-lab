"""Structured description of the one supported Feynman integral.

Deliberately NOT a general loop-integral engine: it describes the two-propagator
massless scalar bubble and refuses anything else.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..conventions import DIMENSION, PRESCRIPTION, SCHEME


@dataclass(frozen=True)
class SupportedIntegral:
    name: str
    n_propagators: int
    internal_masses_squared: tuple[float, ...]
    numerator: str = "1"
    dimension: str = DIMENSION
    prescription: str = PRESCRIPTION
    scheme: str = SCHEME

    def require_supported(self) -> None:
        if self.n_propagators != 2:
            raise NotImplementedError("Only two-propagator bubbles are supported.")
        if any(m2 != 0.0 for m2 in self.internal_masses_squared):
            raise NotImplementedError("Massive internal propagators are not supported.")
        if self.numerator != "1":
            raise NotImplementedError("Tensor numerators are not supported.")
        if self.scheme != SCHEME:
            raise NotImplementedError("Only the finite MSbar part is supported.")
        if self.prescription != PRESCRIPTION:
            raise NotImplementedError("Only the -i0 prescription is supported.")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_propagators": self.n_propagators,
            "internal_masses_squared": list(self.internal_masses_squared),
            "numerator": self.numerator,
            "dimension": self.dimension,
            "prescription": self.prescription,
            "scheme": self.scheme,
            "uv_pole_status": "not computed; finite MSbar part only",
        }


def massless_bubble() -> SupportedIntegral:
    integral = SupportedIntegral(
        name="massless_scalar_bubble",
        n_propagators=2,
        internal_masses_squared=(0.0, 0.0),
    )
    integral.require_supported()
    return integral
