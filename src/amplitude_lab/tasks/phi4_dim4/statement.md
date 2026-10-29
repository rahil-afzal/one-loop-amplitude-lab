# Four-point amplitude of a massless phi^4 scalar at one loop

One real massless scalar with

    L = 1/2 (d phi)^2 + c4/4! phi^4

(note the sign: L_int = +c4/4! phi^4). Consider phi phi -> phi phi with all legs massless,
incoming momenta p1, p2 and outgoing p3, p4, metric (+,-,-,-), and
s = (p1+p2)^2, t = (p1-p3)^2, u = (p1-p4)^2, so s + t + u = 0.

The amplitude A is defined by  iA = sum of connected, amputated diagrams, with vertex +i c4
and propagator i/(p^2 + i0). Work to order c4^2: tree level plus the one-loop diagrams. The
UV divergence is removed by minimal subtraction (MS-bar) at scale mu, with mu2 = mu^2. Give
the renormalized amplitude, so the finite part carries the mu2 dependence.

## What to deliver

A python file defining

    def amplitude(s, t, u, c4, mu2):
        ...  # complex: tree + one loop

s, t, u are complex numbers with s + t + u = 0 (to rounding). The physical s-channel region
(s > 0, t < 0, u < 0) is reached by approaching the real s axis from above, s -> s + i0 with
t real. Use principal branches. The file may import math, cmath, fractions, sympy and mpmath,
nothing else, and should load and evaluate in a couple of seconds.

## How you work

Each reply contains exactly one fenced block, tagged either `run` or `submit`.

* `run`: python code is executed in a scratch directory (python and sympy only, 60 s limit)
  and you get the output back. Use it to derive things and test ideas.
* `submit`: the block is your candidate file. A set of consistency checks runs on it and you
  are told which checks failed, with a short hint. You have {max_attempts} submissions. The
  task ends when every check passes or the submissions or turns run out.

The harness will send short, stage-specific instructions in separate turns. Follow the
current stage and do not skip ahead: the first stages ask for small `run` calculations, then
the final stage asks for a complete `submit` block. Keep each reply and each `run` output
focused and concise. After verifier feedback, focus on the checks that failed.

The checks, in the order they are reported:

* crossing: A is symmetric under permutations of s, t, u
* real_analyticity: A(s*, t*, u*) = A(s, t, u)*
* coupling_dependence: tree level is c4, the loop part is proportional to c4^2
* scale_covariance: dimensional analysis
* unitarity: the imaginary part in the physical region agrees with the tree-level cut
* dispersion: the analytic structure in s at fixed t is consistent with that cut
* rg_beta: the mu dependence is consistent with the one-loop running of c4
* reference_match: the finite part agrees with a reference in the MS-bar scheme
