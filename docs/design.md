# Design notes

## Setting

One real massless scalar, `L_int = +c4/4! phi^4`, so the tree amplitude is `A = c4` (vertex
`+i c4`, `iA` = amputated diagrams). To one loop in MS-bar,

    A = c4 + c4^2/(32 pi^2) [B(s) + B(t) + B(u)],   B(p2) = 2 - Log(-p2/mu2)

The `1/32 pi^2` is the bubble's symmetry factor 1/2 times the loop factor `1/16 pi^2`.
`tests/helpers.py` recovers the 1 and the 1/2 by counting Wick contractions, and checks the
bubble against a 4d momentum-space integral. The physical region is `s -> s + i0`, which
gives `Im B = +pi` for `s > 0`; unitarity needs `Im A > 0`, which agrees.

## Interface

A candidate is a file defining `amplitude(s, t, u, c4, mu2)`, complex arguments, `s+t+u = 0`.
The verifier never passes real-axis points: it uses generic complex points and, for the
physical region, `s + i*eta` with `u - i*eta`. So a candidate cannot be marked right or
wrong because of a convention about what `log` of a negative real number does. The reference
itself refuses exactly real positive input (the side of the cut is a choice).

## Checks

`checks.py`, in the order the agent sees them:

- crossing: same value for all permutations of (s, t, u)
- real_analyticity: `A(conj z) = conj A(z)`
- coupling_dependence: `A(+-eps)/(+-eps) -> 1`; `A - c4` is even in `c4` and scales as `c4^2`
- scale_covariance: `A(l s, l t, l u, c4, l mu2) = A(s, t, u, c4, mu2)`
- unitarity: `Im A` at physical points equals `c4^2/(32 pi)`, computed in `physics.py` from the
  tree amplitude and two-body phase space (no loop integral involved)
- dispersion: the second s-derivative at fixed t equals
  `(2/pi) Im A * [1/(2 s^2) + 1/(2 u^2)]`, the twice-subtracted dispersion integral over the
  s- and u-channel cuts. Additive constants drop out
- rg_beta: the explicit mu dependence cancels `beta * dA/dc4` with `beta = -3 c4^2/(16 pi^2)`
- reference_match: agreement with `reference.py` to 1e-6 of the one-loop part

Everything except reference_match and the beta coefficient is computed from tree-level data
alone. Beta is a textbook number; it also follows from unitarity plus scale covariance, so
rg_beta is not independent information, just a stricter way to hit the same logs.

## Sampling

The first version used four fixed complex points, positive couplings and `mu2` near 1. A review
found five wrong amplitudes that passed every check (`c4*|c4|` in the loop, `|c4|` for the
tree, the loop dropped for large invariants, zero for tiny ones, an extra term at large
`mu2`). Now every check draws its own points from a seeded generator:

- one common scale per point, stratified over 0.01 to 30 (unitarity: 0.01 to 100), so every
  decade is covered in every submission
- couplings from 0.05 to 3 with both signs; `mu2` from 1e-3 to 1e3
- points keep clear of the cuts; the dispersion check also sets its stencil step from the
  distance to the nearest cut, since a fixed step is not accurate near a cut
- a new random seed per submission; the seed is written to `attempts.jsonl`, never shown to the
  agent, and `verify --seed N` reproduces a check run

Tolerances are written as error/tolerance ratios. On the reference over 1000 seeds the largest
ratio is 0.0135 (dispersion), so every check has a factor 70 of room, and a test requires a
factor 10. The numerically integrated example (`numeric_quad.py`) passes all seeds; the same
integral with too few nodes (`low_accuracy_quad.py`) is rejected.

## What the checks cannot see

Constants. A candidate with `B = 1 - Log(-p2/mu2)` passes everything except
reference_match, because crossing, unitarity, dispersion and the log part of the RG are all
blind to an additive `c4^2` term. In a real calculation that constant is fixed by the
scheme, and the only independent handle is a reference (a paper, or another tool). This
is the main reason calibration papers matter.

Because reference_match is the only thing that fixes the constant, "solved" means "matches
the reference". The seven structural checks add feedback, not acceptance power, so a run
reports a structural outcome and a reference outcome separately. A wrong constant can also be
found by trying 0, 1, 2, ...: each guess costs a submission, and the hint says the constant
is the likely problem. If that matters for an experiment, count constant-only submissions.

Sampling is finite, so a pass is evidence, not proof.

## Feedback

After a failed submission the agent sees the names of the failed checks and the fixed hint for
each, and nothing numerical. The numbers go to `attempts.jsonl` for the experimenter. If a
file crashes, the agent sees the exception type and the line of its own code that raised, not
the message: a message such as `ValueError(repr(args))` would hand the test points back. Import
errors keep their message (it is ours). The statement lists the check names up front; that is
a choice, and could be switched off to see how much it matters.

## Sandbox

`run` blocks and candidate files execute in a subprocess (`python -I`, a scratch directory,
timeout). Candidate files are copied into the scratch directory first, so nothing is written
next to the submitted file. Two layers:

1. An `__import__` wrapper limits `import` statements written in the snippet to python, sympy
   and mpmath. That is the "tools" restriction. It looks at the calling frame, so direct
   `__import__('os')` is caught too.
2. A PEP 578 audit hook refuses the operations that matter, whoever performs them: opening
   anything for reading outside the interpreter's library and the scratch directory (and never
   the amplitude_lab package), writing outside scratch, listing directories outside those,
   process creation, `kill`, network, file-system changes, and ctypes except the `dlopen(None)`
   that sympy's own import performs. Importing `amplitude_lab` is refused.

On POSIX the child also gets limits on memory (2 GiB) and file size.

The first version had only the wrapper, and a review showed the answer key was one
`__import__('amplitude_lab.reference')` away, and that sympy's `import_module` hands out any
module. `tests/test_sandbox.py` now runs those attacks (and open, listdir, system, subprocess,
sockets, ctypes, kill, memory and file-size bombs) and a candidate that imports the reference,
searches the disk for it, or overwrites the result file, and requires every one to fail.

What this is not: a security boundary. Native code can tamper with the interpreter, the
verifier and agent share a machine, and none of it has been tried on Windows. sympy can pull
numpy in through `lambdify`, which the import wrapper does not see. For runs that matter use
a separate interpreter without amplitude_lab or numpy (`AMPLITUDE_LAB_SANDBOX_PYTHON`) and a
container.

## Run logs

`meta.json` records the task, the hash of the statement text, the model settings and the
library versions, so a run can be audited and compared; `events.jsonl` has one row per turn
(kind, timings, sizes), which makes "did it use sympy at all" a count of `run` blocks;
`attempts.jsonl` keeps the verifier's numbers and seed per submission.

## Notes

- sympy 1.14: `integrate(-log(x*(1-x)), (x, 0, 1))` returns `2 - 2*I*pi`, not 2, because of
  the branch its antiderivative of `log(1-x)` uses. A candidate that keeps that complex
  constant fails `real_analyticity`, `unitarity` and `reference_match` (checked over 20 seeds);
  one that wraps it in `float()` crashes and gets the "failed to run" feedback.
  `examples/candidates/correct.py` integrates `log(x)` alone and uses `x -> 1-x`.
- Isolated verification starts a fresh interpreter (and imports sympy) once per check, a few
  seconds per submission. Runs that are not isolated (`--trusted`) take milliseconds. The test
  suite takes about two minutes for the same reason.
