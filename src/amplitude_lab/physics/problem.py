"""Strict, validated problem specification for the supported theory."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from .conventions import DEFAULT_MU2, DEFAULT_S0, SCHEME, SUPPORTED_PROCESS

SUPPORTED_COUPLINGS = ("c4", "c42")


@dataclass(frozen=True)
class ProblemSpec:
    """phi1 phi1 -> phi1 phi1 in the massless two-scalar dim-4 theory, one loop.

    ``c4``  is the tree amplitude of phi1 phi1 -> phi1 phi1 (paper: c_1^[4]).
    ``c42`` is the tree amplitude of phi1 phi1 -> phi2 phi2 (paper: c_1^[42]).
    Invalid specifications cannot be constructed.
    """

    name: str
    process: str = SUPPORTED_PROCESS
    c4: float = 1.0
    c42: float = 0.0
    mu2: float = DEFAULT_MU2
    s0: float = DEFAULT_S0
    loop_order: int = 1
    scheme: str = SCHEME
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Problem name must be non-empty.")
        if self.process != SUPPORTED_PROCESS:
            raise ValueError(f"Unsupported process {self.process!r}; only {SUPPORTED_PROCESS!r}.")
        for label, value in (("c4", self.c4), ("c42", self.c42)):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite.")
        if not math.isfinite(self.mu2) or self.mu2 <= 0.0:
            raise ValueError("mu2 must be finite and positive.")
        if not math.isfinite(self.s0) or self.s0 <= 0.0:
            raise ValueError("s0 must be finite and positive.")
        if self.loop_order != 1:
            raise ValueError("Only one-loop order is supported.")
        if self.scheme != SCHEME:
            raise ValueError(f"Only the {SCHEME} scheme is supported.")

    @property
    def coupling_sq(self) -> float:
        """c4^2 + c42^2: the weight multiplying the one-loop bubble sum."""
        return self.c4 * self.c4 + self.c42 * self.c42

    def with_(self, **changes) -> ProblemSpec:
        """Return a modified copy (re-validated)."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "process": self.process,
            "couplings": {"c4": self.c4, "c42": self.c42},
            "mu2": self.mu2,
            "s0": self.s0,
            "loop_order": self.loop_order,
            "scheme": self.scheme,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProblemSpec:
        couplings = dict(data.get("couplings", {}))
        unknown = set(couplings) - set(SUPPORTED_COUPLINGS)
        if unknown:
            raise ValueError(
                f"Unsupported couplings {sorted(unknown)}; supported: {list(SUPPORTED_COUPLINGS)}."
            )
        if "c4" not in couplings:
            raise ValueError("Problem must define the coupling 'c4'.")
        return cls(
            name=data["name"],
            process=data.get("process", SUPPORTED_PROCESS),
            c4=float(couplings["c4"]),
            c42=float(couplings.get("c42", 0.0)),
            mu2=float(data.get("mu2", DEFAULT_MU2)),
            s0=float(data.get("s0", DEFAULT_S0)),
            loop_order=int(data.get("loop_order", 1)),
            scheme=data.get("scheme", SCHEME),
            description=data.get("description", ""),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ProblemSpec:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
