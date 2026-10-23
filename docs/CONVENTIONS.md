# Conventions

All conventions are defined once in `src/amplitude_lab/conventions.py` and read at call
time. Change one there and the validation suite tells you what you broke.

## Theory

Two real massless scalars, dimension-4 sector:

```
L_int = + c4/4! phi1^4 + c42/4 phi1^2 phi2^2
```

`i*A` is the sum of amputated diagrams; vertex `+i*c`; propagator `i/(p^2 + i0)`.

| Quantity | Value | Comment |
|---|---|---|
| `A(phi1 phi1 -> phi1 phi1)` tree | `c4` | **one** contact diagram; the 4! cancels leg assignments. Do not sum three channels. |
| `A(phi1 phi1 -> phi2 phi2)` tree | `c42` | this is the *definition* of `c42` |
| bubble symmetry factor | `1/2` | two identical internal lines |
| loop measure | `1/(16 pi^2)` | 4D, MSbar-normalized |

Wick enumeration (`combinatorics.py`) reproduces the first, second and third rows
independently of the amplitude code.

## One-loop amplitude

```
A_1loop = (c4^2 + c42^2)/(32 pi^2) [B(s) + B(t) + B(u)]
```

Both a phi1 loop (weight `c4^2`) and a phi2 loop (weight `c42^2`) contribute to each of
the three channels. No cross terms arise: a `c4`-`c42` bubble would need a species
mismatch or a massless tadpole.

## Bubble and branch cut

```
B(s; mu2) = - Int_0^1 dx Log[(-x(1-x) s - i0)/mu2] = 2 - Log(-s/mu2)
s < 0 :  B = 2 - ln(-s/mu2)
s > 0 :  B = 2 - ln(s/mu2) + i*pi
```

The `-i0` places the logarithm's argument just *below* the negative real axis, so
`Log = ln|.| - i*pi`, and the overall minus sign gives `+i*pi`. This is the boundary value
from the **upper** half `s`-plane, the physical sheet. Unitarity independently requires
`Im A > 0`, which fixes the same sign (see `physics_unitarity_cut`).

`B` is analytic in the `s`-plane cut along `s > 0`; `Disc B = B(s+i0) - B(s-i0) = 2*pi*i`.

## Forward limit

As `t -> 0` the real part contains `-ln(-t/mu2)` and diverges. Only the imaginary part is
finite and only it is ever evaluated in the forward limit (`Kinematics.forward` keeps
`t = -rel_eps * s` strictly inside the physical region at any scale).

## Renormalization group

```
beta_c4 = mu dc4/dmu = -3 (c4^2 + c42^2) / (16 pi^2)
```

Textbook two-scalar phi^4 result, mapped with `c = -lambda` (`beta_lambda1 = (3 lambda1^2 + 3
lambda12^2)/(16 pi^2)`). It is a *reference*, not derived by this code. The check verifies
that the explicit `mu` dependence of the loop, `mu dA/dmu = 3 (c4^2 + c42^2)/(16 pi^2)`,
is cancelled by `beta_c4 * dA/dc4` with `dA/dc4 = 1` at this order.

## Cross sections

```
dsigma/dt = |A|^2 / (16 pi s^2)           massless 2 -> 2
sigma(phi1 phi1 -> X) = (c4^2 + c42^2) / (32 pi s)
```

Each final state (`phi1 phi1`, `phi2 phi2`) has two identical particles, hence the factor
1/2. Optical theorem: `Im A(s, t->0) = s * sigma`.

## Dispersion benchmark

```
Sigma = (2/pi) Int_0^inf ds s^4 sigma(s)/(s^2+s0^2)^3 = (c4^2 + c42^2)/(64 pi^2 s0^2)
```

For `c4 = 1`, `c42 = 0`, `s0 = 0.25`: `Sigma = 1/(4 pi^2) ~ 0.025330`.

Equation numbers (3.5, 3.9, 3.11) and the meaning of `c_1^[42]` come from the project
notes and **must be re-checked against the paper PDF** before the benchmark is cited.
