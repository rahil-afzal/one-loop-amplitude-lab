# Conventions

All conventions are defined in `src/amplitude_lab/physics/conventions.py`.

## Theory

Two real massless scalars, dimension-four sector:

```
L_int = + c4/4! phi1^4 + c42/4 phi1^2 phi2^2
```

`i*A` is the sum of amputated diagrams; vertex `+i*c`; propagator `i/(p^2+i0)`.

| Quantity | Value |
|---|---|
| `A(phi1 phi1 -> phi1 phi1)` tree | `c4` |
| bubble symmetry factor | `1/2` |
| loop measure | `1/(16 pi^2)` |

## One-loop amplitude

```
A_1loop = (c4^2 + c42^2)/(32 pi^2) [B(s) + B(t) + B(u)]
```

## Bubble and branch cut

```
B(s; mu2) = 2 - Log(-s/mu2)
s < 0 : B = 2 - ln(-s/mu2)
s > 0 : B = 2 - ln(s/mu2) + i*pi
```

The upper-half-plane boundary gives `+i*pi` for real positive `s`.
`Disc B = B(s+i0)-B(s-i0) = 2*pi*i`.

## Renormalization

```
beta_c4 = mu dc4/dmu = -3 (c4^2 + c42^2)/(16 pi^2)
```

The final-answer verifier compares a candidate expression with this repository convention.
It does not provide an external proof or independent paper validation.
