"""Built-in benchmark problems (JSON copies live in the repository ``benchmarks/``)."""

from __future__ import annotations

from pathlib import Path

from .problem import ProblemSpec


def yhg_dim4() -> ProblemSpec:
    """Dim-4 one-loop sector of the Ye-He-Gu two-scalar EFT positivity benchmark.

    Corresponds to the c_1^[4], c_1^[42] piece only (the Eq. (3.5)/(3.11) subset).
    It is NOT the full Eq. (3.9). Re-verify equation numbers against the paper.
    """
    return ProblemSpec(
        name="yhg_dim4_one_loop",
        c4=0.7,
        c42=0.4,
        mu2=1.0,
        s0=2.0,
        description=(
            "Two massless scalars, dim-4 couplings only; one-loop finite MSbar bubbles "
            "and the dim-4 dispersion contribution (Ye-He-Gu Eq. 3.11 subset)."
        ),
    )


def pure_phi4() -> ProblemSpec:
    """Single massless real scalar with a quartic coupling (c42 = 0)."""
    return ProblemSpec(
        name="pure_phi4_one_loop",
        c4=1.0,
        c42=0.0,
        mu2=1.0,
        s0=0.25,
        description="Single massless real scalar, L_int = c4/4! phi^4, one-loop bubbles.",
    )


BUILTIN = {"yhg_dim4": yhg_dim4, "pure_phi4": pure_phi4}


def resolve(name_or_path: str) -> ProblemSpec:
    """Resolve a built-in benchmark name or a JSON file path."""
    if name_or_path in BUILTIN:
        return BUILTIN[name_or_path]()
    path = Path(name_or_path)
    if path.is_file():
        return ProblemSpec.from_json(path)
    raise FileNotFoundError(
        f"{name_or_path!r} is neither a built-in benchmark ({', '.join(BUILTIN)}) "
        "nor an existing JSON file."
    )
