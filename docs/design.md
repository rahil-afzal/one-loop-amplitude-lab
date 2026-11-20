# Design and boundaries

## Physics core

The package implements the retained massless two-scalar dimension-four benchmark:
`physics/`, `integrals/bubble.py`, and benchmark/problem data feed the autonomous
workflow and deterministic verifier.

## Parser

`safe_expr.py` converts a strict subset of arithmetic AST nodes into SymPy expressions.
It rejects attributes, calls outside the allowlist, imports, unsafe exponents, and
resource-exhausting inputs. It never executes LLM text.

## Autonomous workflow

`derive` loads a packaged task prompt, asks a derivation model for a plan, derives
independent steps, and audits accepted checkpoints. Rejections revise only the relevant
checkpoint and invalidate dependent work. Terminal progress and JSONL events are optional.
Runs are persisted under `runs/`.

## Deterministic verifier

`verify FILE` extracts the last final block and compares `A_tree`, `A_loop`, and `beta_c4`
against the repository amplitude and conventions. It is finite-sample repository
consistency evidence, not an external proof.

The package also exposes `task` for prompt printing and `derive` for the checkpoint
workflow. Browser, proposal-assistant, legacy-feedback, and registered-validation surfaces
are outside the retained scope.
