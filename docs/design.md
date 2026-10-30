# Design and boundaries

## Physics core

The package implements the finite one-loop amplitude for the supported massless two-scalar
dimension-four theory. The conventions and formula are defined in
[CONVENTIONS.md](CONVENTIONS.md); the supported scope is tracked in [ROADMAP.md](ROADMAP.md).

The calculation is split by responsibility:

- `problem.py`, `kinematics.py`, and `conventions.py` define inputs and conventions.
- `integrals/` contains the bubble definition and independent numerical routes.
- `amplitude.py` assembles channel contributions.
- `validation/` runs registered checks with explicit evidence labels.
- `reporting.py` serializes results without changing check outcomes.

The deterministic `run` and `validate` commands do not call a model. Validation is finite
sample evidence, not a proof; see [VALIDATION.md](VALIDATION.md).

## Optional LLM workflows

`assistant/` accepts an expression proposal only after parsing it through a strict AST
allowlist, then checks it against deterministic integral implementations. Assistant results
are recorded separately and cannot change the deterministic report status. Its trust model
and limitations are described in [ASSISTANT.md](ASSISTANT.md).

`derive` requests a written derivation and a separate model audit. Those responses are saved
for review, but prose is not executed or treated as validated physics. Use the deterministic
commands and independent sources to verify any claimed result.

## Iterative audit loop (`refine`) and deterministic verification

`refine` runs derive -> audit -> revise for up to `--max-rounds` rounds (`feedback.py`).

- The derivation agent does the physics and must end with a ```final block giving
  `A_tree`, `A_loop` and `beta_c4`.
- The audit agent returns `VERDICT`, `FIRST_ERROR`, `ERROR` and `HINT`. It is told not to
  write a full solution. Its text, not a reference answer, is what the next round sees.
- `verification.py` checks the final block after every round, with no LLM involved. The
  checks are: parseability through the AST allowlist, the tree value, the one-loop
  amplitude against `assemble` at physical `s + i0` points for two `mu2` values,
  crossing invariance at complex points, and the beta-function sign.
  The verdict is saved per round (`roundN_verification.json`) and is never shown to the
  derivation agent. Any gain from round 1 to round N therefore comes from the audit
  feedback alone.
- An audit `PASS` never overrides a failed verification. Both are recorded separately.

`verify FILE` runs only the deterministic layer on a saved derivation.

Limits: the references come from this repository (`amplitude.py`, `conventions.py`), so the
check is consistency with the repo's stated result, not an external proof. The check is
finite-sample. One task (massless phi^4) means few runs say little about accuracy.

## Data and generated files

Task directories contain analytical prompts. CLI reports, prompts, transcripts, and request
metadata are generated under `runs/`, which is intentionally local and ignored by Git.
