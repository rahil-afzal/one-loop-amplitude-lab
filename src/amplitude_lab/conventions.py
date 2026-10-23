"""Single source of truth for physics conventions.

Theory (two real massless scalars, dimension-4 sector)::

    L_int = + c4/4! phi1^4  +  c42/4 phi1^2 phi2^2

Amplitude convention: ``i*A`` = sum of amputated diagrams, vertex ``+i*c``,
propagator ``i/(p^2 + i0)``. Consequences:

* ``A(phi1 phi1 -> phi1 phi1)_tree = c4``      (ONE contact diagram, not three channels)
* ``A(phi1 phi1 -> phi2 phi2)_tree = c42``     (this *defines* the meaning of ``c42``)

One-loop bubbles: each of the s/t/u channels has Wick symmetry factor 1/2 and loop
measure factor 1/(16 pi^2), giving the combined prefactor 1/(32 pi^2). In the channel
of ``phi1 phi1 -> phi1 phi1`` both a phi1 loop (weight c4^2) and a phi2 loop
(weight c42^2) contribute::

    A_1loop = (c4^2 + c42^2)/(32 pi^2) * [B(s) + B(t) + B(u)]

Finite bubble (MSbar), physical sheet = upper-half s-plane boundary::

    B(s; mu2) = - Int_0^1 dx Log[(-x(1-x) s - i0) / mu2] = 2 - Log(-s/mu2)

    s < 0 :  B = 2 - ln(-s/mu2)               (real)
    s > 0 :  B = 2 - ln( s/mu2) + i*pi        (NOT -i*pi)

Reference beta function (textbook two-scalar phi^4 RG, mapped to this sign convention
via c = -lambda)::

    beta_c4 = mu dc4/dmu = -3 (c4^2 + c42^2) / (16 pi^2)

Scope limits (see docs/ROADMAP.md): no dim-6/8 operators, no massive propagators,
no triangles/boxes, no tensor numerators, no UV-pole/counterterm matching.
"""

from __future__ import annotations

import math

DEFAULT_MU2 = 1.0
DEFAULT_S0 = 1.0

BUBBLE_SYMMETRY_FACTOR = 0.5
LOOP_MEASURE_FACTOR = 1.0 / (16.0 * math.pi**2)
ONE_LOOP_PREFACTOR = BUBBLE_SYMMETRY_FACTOR * LOOP_MEASURE_FACTOR  # 1/(32 pi^2)

# beta_c4 = BETA_C4_COEFFICIENT * (c4^2 + c42^2)
BETA_C4_COEFFICIENT = -3.0 / (16.0 * math.pi**2)

SUPPORTED_PROCESS = "phi1 phi1 -> phi1 phi1"
SCHEME = "MSbar"
PRESCRIPTION = "-i0"
DIMENSION = "4-2*epsilon"


def describe() -> dict:
    """Machine-readable summary of the conventions (embedded in every report)."""
    return {
        "lagrangian_interaction": "+ c4/4! phi1^4 + c42/4 phi1^2 phi2^2",
        "amplitude_convention": "i*A = amputated diagrams; vertex +i*c; propagator i/(p^2+i0)",
        "tree_phi1phi1_to_phi1phi1": "c4 (single contact diagram)",
        "tree_phi1phi1_to_phi2phi2": "c42 (definition of c42)",
        "bubble_symmetry_factor": BUBBLE_SYMMETRY_FACTOR,
        "one_loop_prefactor": "(c4^2 + c42^2)/(32 pi^2)",
        "bubble": "B(s; mu2) = 2 - Log(-s/mu2), principal log",
        "timelike_bubble_imaginary_part": "+pi",
        "scheme": SCHEME,
        "prescription": PRESCRIPTION,
        "reference_beta_c4": "-3 (c4^2 + c42^2) / (16 pi^2)",
        "supported_process": SUPPORTED_PROCESS,
    }
