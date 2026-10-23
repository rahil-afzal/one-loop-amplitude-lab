# Validation strategy

## Why evidence is stratified

An earlier design used the same derivation to produce an analytic formula and the
"oracle" that checked it. Agreement then proves only internal consistency. Every check
here declares its evidence kind and may not claim more.

## The checks

| Check | Evidence | What it can catch |
|---|---|---|
| `integral_feynman_quadrature` | implementation | wrong constant in `B`, wrong branch in complex plane |
| `integral_momentum_space` | **independent** | wrong closed form (4D integral, no Feynman parameters) |
| `integral_branch_cut` | implementation | wrong sign of `Im B`, wrong continuation; numeric `s+i*eta` is compared to the exact principal-log value `pi - atan(eta/s)` |
| `integral_symmetry_prefactor_link` | implementation | prefactor inconsistent with its documented decomposition |
| `combinatorics_wick_factors` | **independent** | wrong tree coefficient (e.g. 3 channels), wrong symmetry factor, wrong `c42` definition |
| `amplitude_crossing` | consistency | broken s/t/u symmetry |
| `amplitude_scaling` | consistency | wrong log arguments / units of `mu2` |
| `physics_optical_theorem` | **independent** | loop normalization or sign vs. `s * sigma` from tree + phase space (`dsigma/dt` and `dPhi_2` routes) |
| `physics_unitarity_cut` | **independent** | angle dependence, sign of `Im A`, missing phi2 loop |
| `physics_bubble_dispersion` | **independent** | links `Im B` (unitarity) to `B''` (Feynman integral): `B'' = 1/x^2` |
| `physics_sigma_dispersion_dim4` | reference | `Sigma` vs `c^2/(64 pi^2 s0^2)` via three routes (quadrature, assembled-amplitude integral, SymPy) |
| `physics_rg_invariance` | reference | wrong `mu` dependence vs. textbook beta function |

Run `python -m amplitude_lab list-checks` for the live list.

## Negative controls

`tests/test_validation_suite.py` monkeypatches deliberately wrong physics and requires
the suite to fail the right checks: loop prefactor without the symmetry factor, `-i*pi`
on the cut, tree summed over three channels, wrong beta function, dropped phi2 loop.
A validator that cannot fail is not evidence. These controls already found one real
bug during development: a check that imported a constant by value and therefore never
saw the sabotaged value.

## What the checks do not establish

- The massless Feynman-parameter integral factorizes (`x(1-x)` leaves the log), so
  quadrature mostly validates the constant 2 and branch handling. The momentum-space,
  unitarity and dispersion checks carry the weight.
- Optical theorem and unitarity cut are the same physical statement tested through two
  phase-space formulas; they validate normalization and sign, not new physics.
- The beta function is a reference value. The Sigma formula is checked against the
  closed form as stated in the project notes, not against the paper PDF.
- Finite sampling is never a proof of global analytic correctness.
- Nothing here validates dim-6/dim-8 operators, massive thresholds or UV matching.
