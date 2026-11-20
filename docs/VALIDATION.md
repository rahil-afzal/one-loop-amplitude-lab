# Deterministic final-answer verification

`python -m amplitude_lab verify FILE` extracts the last fenced `final` block and parses
each expression through `amplitude_lab.safe_expr`, a strict AST allowlist. It never
executes LLM-produced text.

The retained checks compare:

- `A_tree` with the repository tree amplitude;
- `A_loop` with `assemble` at several physical-region points and two `mu2` values;
- `A_loop` crossing symmetry over off-axis permutations;
- `beta_c4` with the repository convention value.

These checks are finite-sample consistency checks against the repository's own amplitude,
kinematics, bubble, and convention implementations. They are not an external proof,
paper cross-reference, or global analytic proof.
