Analytical amplitude derivation: one-loop QED photon vacuum polarization

Derive the one-loop photon self-energy from a single Dirac fermion of mass m and
charge -e (e > 0), from first principles. Define the full Lagrangian yourself from
the named theory and these conventions: metric (+,-,-,-), canonical Maxwell and
Dirac terms, and covariant derivative D_mu = d_mu + i*e*A_mu. The Lagrangian
expression is deliberately not supplied; derive the vertex factor and its sign
from your stated convention rather than assuming it. Include the closed-fermion-loop
sign explicitly.

Use dimensional regularization with d = 4 - 2*epsilon and renormalize in the MS-bar
scheme at scale mu, with mu2 = mu^2, m2 = m^2, q2 = q^2. Define the 1PI two-point
function by i*Pi^{mu nu}(q) = i*(q2*g^{mu nu} - q^mu q^nu)*Pi(q2). These conventions
are binding: do not substitute a different decomposition, sign, or scheme.

Work from the Lagrangian, not from a remembered formula. Show the Dirac trace, the
Feynman-parameter reduction, and the epsilon expansion with checkable algebra.
Distinguish the denominator-only scalar integral, the propagator factors of i, the
trace numerator, and the complete 1PI function. Give the renormalized Pi(q2) as an
explicit Feynman-parameter integral over x in [0,1] before evaluating any limit.

Required checks, each with its evidence shown:
- Transversality: show that the g^{mu nu} terms proportional to m2 cancel, so no
  independent photon-mass structure survives and q_mu Pi^{mu nu} = 0.
- Unitarity: obtain Im Pi above the threshold q2 > 4*m2 from the cut of the fermion
  pair, and verify it against the tree-level gamma* -> f fbar rate.
- Massless limit: take m2 -> 0 at fixed q2 away from the cut and give the closed form.
- RG: extract beta(e) from the pole and confirm mu-independence of the renormalized
  Pi(q2) through this order.

State assumptions or uncertainty rather than guessing, but do not override the stated
conventions. Return concise Markdown and LaTeX, not code or private chain-of-thought.
Do not claim a check was performed unless you show its evidence.

The final work step must end with a fenced `final` block containing exactly these
keys in safe arithmetic syntax:
  Pi_pole      coefficient of 1/epsilon_bar in the unrenormalized Pi(q2)
  Pi_massless  renormalized Pi(q2) in the m2 -> 0 limit, analytic form
  Pi_imag      Im Pi(q2) on the upper boundary for q2 > 4*m2
  beta_e       one-loop beta function of e
Available symbols are only: e, q2, m2, mu2, pi. Available functions: log, sqrt.
The symbols e, m, mu, q are NOT available; use m2 and mu2. log is the natural
logarithm on the principal branch.