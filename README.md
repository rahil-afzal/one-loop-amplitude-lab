# amplitude-lab

[![CI](https://github.com/YOUR_USER/one-loop-amplitude-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USER/one-loop-amplitude-lab/actions/workflows/ci.yml)

An agent-assisted framework for computing and **validating** one-loop scalar scattering
amplitudes.

The system represents a well-defined scattering problem, evaluates the supported Feynman
integral analytically and numerically, assembles the amplitude with explicit symmetry
factors and conventions, and validates the result with independent checks: crossing
symmetry, branch-cut behaviour, the optical theorem, unitarity cuts and dispersion
relations. An optional Gemini assistant can propose candidate formulas, but it is never
treated as mathematical ground truth: every proposal is parsed through a strict AST
whitelist and verified by deterministic code.

**Long-term benchmark:** the Ye–He–Gu scalar-EFT positivity paper, starting with the
dim-4 one-loop contribution and extending to the full two-scalar EFT dispersion relation
(see [docs/ROADMAP.md](docs/ROADMAP.md)).

## Supported scope (v0.1)

Two real massless scalars, dimension-4 interactions, process `phi1 phi1 -> phi1 phi1`:

```
L_int = + c4/4! phi1^4 + c42/4 phi1^2 phi2^2

A_tree  = c4                                          (one contact diagram)
A_1loop = (c4^2 + c42^2)/(32 pi^2) * [B(s) + B(t) + B(u)]
B(s; mu2) = 2 - Log(-s/mu2)                           (MSbar, physical sheet)
```

For real `s > 0`, `B = 2 - ln(s/mu2) + i*pi` (note `+i*pi`, from the `-i0` prescription).
`c42 = 0` is the single-scalar phi^4 theory. The dispersion benchmark is the dim-4
contribution `Sigma = (c4^2 + c42^2) / (64 pi^2 s0^2)` (the Eq. 3.11 subset, **not** the
full Eq. 3.9).

**Not supported** (and listed in every report): dim-6/dim-8 operators, massive
propagators, triangles, boxes, tensor numerators, spins, UV-pole/counterterm matching.
Full conventions: [docs/CONVENTIONS.md](docs/CONVENTIONS.md).

## Install

Windows PowerShell (conda):

```powershell
conda create --name amplitude-lab python=3.11 --yes
conda activate amplitude-lab
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"          # physics core + tests, no Gemini needed
python -m pip install -e ".[dev,gemini]"   # add this only if you want the Gemini assistant
```

Linux / macOS: the same `pip` commands in any virtual environment.

## Use

```powershell
python -m pytest                                  # offline, no key, no network
python -m amplitude_lab run                       # benchmark + reports in reports/
python -m amplitude_lab validate                  # checks only, as a table
python -m amplitude_lab list-checks               # what each check proves
python -m amplitude_lab amplitude --s 5 --t -2    # evaluate the amplitude
python -m amplitude_lab describe                  # conventions as JSON
```

Built-in problems: `yhg_dim4` (default, `c4=0.7`, `c42=0.4`) and `pure_phi4`. Pass
`--problem path/to/problem.json` for your own (see `benchmarks/`).

Python API:

```python
from amplitude_lab import benchmarks
from amplitude_lab.amplitude import assemble
from amplitude_lab.kinematics import Kinematics
from amplitude_lab.validation import run_all

problem = benchmarks.yhg_dim4()
print(assemble(problem, Kinematics.from_s_t(5.0, -2.0)).total)
print([r.name for r in run_all(problem) if not r.passed])  # [] when everything passes
```

Exit codes: `0` success, `1` validation failure, `2` configuration or input error.

## What the checks prove

Every check is labelled with the strongest claim it may make:

| Evidence | Meaning |
|---|---|
| `implementation` | two evaluations of the same definition agree |
| `consistency` | the assembled amplitude satisfies internal identities |
| `reference` | agreement with an externally stated formula (paper or textbook) |
| `independent` | obtained by a genuinely different route |

The `independent` checks are the ones that can catch a wrong normalization, sign or
symmetry factor: brute-force **Wick-contraction counting** (tree coefficient 1, bubble
1/2), a **4D momentum-space** integral, the **optical theorem** and **unitarity cut**
computed from tree amplitudes and phase space only, and a **dispersion relation** that
links the unitarity cut to the real part. The test suite also contains **negative
controls**: it sabotages the physics (wrong prefactor, `-i*pi`, tree summed over three
channels, wrong beta function, dropped phi2 loop) and asserts that the suite fails.
Details and honest limits: [docs/VALIDATION.md](docs/VALIDATION.md).

## Optional Gemini assistant

```powershell
Copy-Item .env.example .env        # then edit .env locally; never commit it
python -m amplitude_lab run --assistant gemini
python -m amplitude_lab smoke      # one real request, run manually
```

Offline demos need no key: `--assistant mock-correct`, `mock-wrong_sign`,
`mock-wrong_branch`, `mock-explicit_ipi`, `mock-malicious`, `mock-garbage`.

The assistant is asked for a closed form of the bubble. The reply is parsed by an AST
whitelist (never executed), then compared with the deterministic Feynman-parameter
integral at points off the cut plus boundary points that fix the physical sheet. A
`verified` status means a finite-sample test passed; it is not a proof, and **the
assistant can never change the overall pass/fail status**. See
[docs/ASSISTANT.md](docs/ASSISTANT.md).

> The `google-genai` call pattern in `assistant/client.py` was inspected against
> google-genai 2.33.0, but model IDs and live behaviour were **not** verified. Check
> <https://ai.google.dev/gemini-api/docs> and set `GEMINI_MODEL` yourself.

Credentials: `.env` is git-ignored, `GeminiConfig` redacts the key in `repr`, SDK errors
are sanitized before they reach logs or reports. Check with `git check-ignore -v .env`.
If a key was ever committed, rotate it and purge history (`git filter-repo` or BFG);
deleting the file in a new commit is not enough.

## Layout

```
src/amplitude_lab/
  conventions.py     single source of truth for signs and normalizations
  kinematics.py      massless (s, t, u), crossing permutations, physical points
  problem.py         validated ProblemSpec (invalid specs cannot be constructed)
  integrals/         bubble (analytic, scipy, mpmath), momentum-space route, spec
  combinatorics.py   Wick-contraction enumeration (symmetry factors)
  amplitude.py       tree + one-loop assembly
  crosssection.py    tree cross sections, two-body phase space, unitarity cut
  dispersion.py      Sigma benchmark routes, bubble dispersion relation
  validation/        registry of checks with evidence kinds
  assistant/         optional LLM client, safe parser, tasks, verifier
  reporting.py, pipeline.py, cli.py
tests/               unit tests, negative controls, security tests
docs/                conventions, validation, assistant, roadmap
benchmarks/          problem definitions (JSON)
```

## Development

```powershell
ruff check . ; ruff format --check .
python -m pytest --cov
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under MIT.
