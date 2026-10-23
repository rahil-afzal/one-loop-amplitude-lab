# Roadmap toward the Ye–He–Gu benchmark

Each milestone needs its own independent reference check before it is considered done.

| Milestone | Scope | Independent evidence to add |
|---|---|---|
| **M0 (this release)** | massless dim-4, `phi1 phi1 -> phi1 phi1`, bubbles, `Sigma` dim-4 | Wick counting, momentum space, unitarity, bubble dispersion |
| M1 | other channels: `phi1 phi1 -> phi2 phi2`, `phi2 phi2 -> phi2 phi2`; add `phi2^4` coupling | Wick counting per channel, multi-channel unitarity |
| M2 | forward-limit subtraction handling the `ln(-t)` divergence; arcs / smeared amplitudes | analytic subtraction vs. numerical derivative at small `t` |
| M3 | dim-6 and dim-8 operator bases: tree amplitudes, cross sections, symmetry factors | enumerated Feynman rules; tree unitarity |
| M4 | UV poles, counterterms, `beta^[8]` mixing | RG consistency of the full amplitude set |
| M5 | massive propagators, thresholds | threshold behaviour, Landau analysis, comparison with an external library |
| M6 | full two-scalar dispersion relation (Eq. 3.9) with interference terms | comparison with the paper's numerical benchmarks |

Assistant track: tasks for new integrals (massive bubble, triangle) each with a
verifier that does not reuse the proposal's derivation; optional comparison against
Package-X / LoopTools output as an external oracle.

Open verification items: re-check equation numbers and the `c_1^[42]` normalization
against the paper PDF; verify Gemini model IDs against the live documentation.
