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

## Data and generated files

Task directories contain analytical prompts. CLI reports, prompts, transcripts, and request
metadata are generated under `runs/`, which is intentionally local and ignored by Git.
