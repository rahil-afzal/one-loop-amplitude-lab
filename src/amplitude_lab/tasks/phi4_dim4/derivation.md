# Analytical amplitude derivation: massless real scalar phi^4

Derive the analytical 2 -> 2 scattering amplitude for one real massless scalar with

    L = 1/2 (d phi)^2 + c4/4! phi^4

using metric (+,-,-,-). The interaction term is +c4/4! phi^4, so the vertex is +i c4 and the
propagator is i/(p^2 + i0). Define iA as the sum of connected, amputated diagrams. Work through
order c4^2, including tree level and the one-loop diagrams. Renormalize in the MS-bar scheme at
scale mu, with mu2 = mu^2.

For phi(p1) phi(p2) -> phi(p3) phi(p4), use

    s = (p1+p2)^2, t = (p1-p3)^2, u = (p1-p4)^2, s+t+u = 0.

In the physical s-channel region, s is approached from above (s -> s+i0), while t and u are
negative real. Use the corresponding analytic continuation of logarithms; state branch choices
explicitly.

## Deliverable

Return a Markdown derivation with a boxed LaTeX expression for A(s,t,u), valid through order
c4^2. Include:

1. The tree contribution and the one-loop channel contributions, with symmetry factors.
2. The scalar bubble integral and its MS-bar subtraction, showing enough intermediate
   equations for another physicist to check the normalization and finite constant.
3. The crossing-symmetric sum and explicit logarithm branches in the physical region.
4. Brief checks of crossing, the physical s-channel imaginary part, and mu dependence.
5. Any assumptions or uncertainty that could change the result.

Do not write Python, executable code, or a ```run/```submit block. Do not claim a check was
performed if it was only asserted. Give the equations and concise verification evidence, not
private chain-of-thought. If conventions are inconsistent or information is insufficient, say
so clearly rather than guessing.
