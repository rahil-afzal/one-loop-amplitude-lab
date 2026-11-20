# Changelog

## Unreleased

- Trimmed the product to checkpointed autonomous derivation, task prompt printing,
  and deterministic saved-answer verification.
- Removed the browser/SSE server, proposal assistant, feedback/legacy derivation
  workflow, registered validation suite, and unrelated CLI commands.
- Added SciPy as a runtime dependency of the retained bubble/amplitude verifier.

## 0.1.0

- Two-scalar massless dim-4 one-loop amplitude `phi1 phi1 -> phi1 phi1` (`c4`, `c42`).
- Bubble: analytic, scipy, mpmath complex continuation, 4D momentum-space route.
- Twelve deterministic checks with stratified evidence kinds; negative-control tests.
- Brute-force Wick-contraction enumeration for symmetry factors.
- Unitarity cut, optical theorem, bubble dispersion relation, dim-4 `Sigma` (3 routes).
- Optional OpenAI-compatible assistant workflows behind strict parsing and deterministic verification;
  offline mock assistants; credential redaction.
- CLI (`run`, `validate`, `amplitude`, `describe`, `list-checks`), JSON + Markdown reports.
