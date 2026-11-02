# Analytical amplitude derivation: massless real scalar phi^4

Derive the analytical 2 -> 2 scattering amplitude through order c4^2 for one real massless
scalar from first principles, starting with

    L = 1/2 (d phi)^2 + c4/4! phi^4

using metric (+,-,-,-). Define iA as the sum of connected, amputated diagrams. Use
dimensional regularization with d = 4 - 2 epsilon and renormalize in the MS-bar scheme at
scale mu, with mu2 = mu^2. The task does not supply the Feynman rules or the target amplitude:
derive them from the Lagrangian and the stated conventions rather than quoting a remembered
formula.

For phi(p1) phi(p2) -> phi(p3) phi(p4), use

    s = (p1+p2)^2, t = (p1-p3)^2, u = (p1-p4)^2, s+t+u = 0.

In the physical s-channel region, s is approached from above (s -> s+i0), while t and u are
negative real. Use the same Feynman prescription consistently when writing the analytic
channel sum, and explain how its boundary values follow in this region.

Determine a suitable derivation plan from the problem itself; no intermediate answers or
prescribed calculation route are supplied. Work from the Lagrangian and stated assumptions.
At loop stages, distinguish the denominator-only scalar integral, propagator factors of i,
and the complete diagram including vertices and symmetry factors. Define notation before
using it, carry signs and factors of i explicitly, and show checkable algebra connecting each
stage. State assumptions or uncertainty rather than guessing.

Return concise Markdown and LaTeX, not code or private chain-of-thought. Include enough
intermediate equations for each result to be checked. Do not claim a check was performed
unless you show its evidence. The final work step must end with a fenced `final` block with
`A_tree`, `A_loop`, and `beta_c4` expressions in the safe arithmetic syntax if those
quantities are applicable to this problem.
