# Assistant security and trust model

The assistant is a *proposal generator*. It has no authority.

1. **Ask.** `BubbleClosedFormTask` sends the definition of `B(s; mu2)` and asks for one
   expression in `s` and `mu2`.
2. **Parse, never execute.** The reply goes through `assistant/safe_expr.py`: `ast.parse`
   in expression mode, then a node-by-node whitelist converting to SymPy. Allowed: the
   names `s, mu2, pi, I`; `log, ln, sqrt, exp` with one argument; real numbers up to
   1e6; `+ - * /`; powers with numeric exponents `|n| <= 8`. Rejected: attributes,
   subscripts, imports, other calls, lambdas, comprehensions, strings, complex or boolean
   literals, symbolic exponents, non-finite constants, input over 500 characters or 200
   nodes, and numeric powers above 10^50 (blocks nested `((10**6)**8)**8...` blow-ups).
3. **Verify deterministically.** The expression is evaluated with mpmath at points off
   the cut and compared with the mpmath Feynman-parameter quadrature (tolerance 1e-9),
   at two values of `mu2`. Boundary points at `s + i*1e-9` must give `2 - ln(s/mu2) + i*pi`,
   which pins the physical sheet.
4. **Report.** Statuses: `verified`, `rejected`, `rejected_unparseable`, `unavailable`.
   The overall run status depends only on the deterministic checks.

`verified` means a finite-sample test passed. It is not a proof.

## Credentials

- `GEMINI_API_KEY` comes from the environment; a local `.env` is loaded only by the CLI
  assistant commands, and tests are hermetic (the key and `.env` loading are disabled).
- `GEMINI_TIMEOUT_SECONDS` configures the request timeout in seconds for both Gemini clients.
- `GeminiConfig.__repr__` redacts the key; SDK exceptions are caught, the key is
  replaced by `<REDACTED>`, messages are truncated, and the original exception is not
  chained (`from None`).
- No automatic retries; requests have an explicit timeout.

## Adding a task

Implement the `ProposalTask` protocol in `assistant/tasks.py` (`name`, `system_instruction`,
`prompt`, `parse`, `verify`). `verify` must compare against code that does not depend on
the proposal's own derivation. Add rejection tests for wrong, unsafe and unparseable
proposals before wiring a real model.
