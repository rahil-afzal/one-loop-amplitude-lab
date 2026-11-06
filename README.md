# one-loop-amplitude-lab

A compact, testable lab for one-loop scattering amplitudes. The physics pipeline assembles
the amplitude and checks it against independent numerical and analytic evidence. Optional
LLM workflows either propose a bubble integral expression for safe parsing and deterministic
verification, or produce a written derivation followed by a model review.

The supported benchmark is the massless two-scalar dimension-four sector at one loop in
MS-bar. Scope and open validation items are in [docs/ROADMAP.md](docs/ROADMAP.md);
physics conventions are in [docs/CONVENTIONS.md](docs/CONVENTIONS.md).

## Install and test

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

The deterministic workflow and tests run offline:

```powershell
python -m amplitude_lab run --problem pure_phi4 --out-dir runs/pure-phi4
python -m amplitude_lab validate --problem yhg_dim4
python -m amplitude_lab amplitude --problem pure_phi4 --s 5 --t -2
python -m amplitude_lab list-checks
python examples/quickstart.py
```

`run` writes JSON and Markdown reports. To include a no-network assistant example, use
`--assistant mock-correct`; mock outputs never affect the deterministic overall status.
For Gemini connectivity, install `.[gemini]`, set `GEMINI_API_KEY`, then use
`python -m amplitude_lab smoke` or `run --assistant gemini`.

## Autonomous derivation

`derive` asks the derivation model to formulate the theory's Lagrangian and make a
task-specific plan, then proposes and audits each mathematical step independently. The
Lagrangian is established as the first auditable step. Accepted steps become checkpoints in
a dependency graph. Rejected steps are revised; if an audit finds an earlier checkpoint is
wrong, that step and its dependent work are invalidated and recomputed. Attempts, audit
responses, backtracking, and deterministic checks are saved as Markdown under `runs/`
(`response.md`, `steps.md`, and `result.md`); live events are streamed but not persisted as
a separate JSON Lines file. A rejected or unresolved step is never silently accepted. The
auditor receives no reference amplitude and is asked
for diagnostics rather than a solution. The derivation agent has no web search or external
answer source. Its planner selects a task-specific approach and reports its interpretation,
strategy, and assumptions instead of forcing every problem through the same checklist.

The `derive` command streams state and generated tokens to the terminal. Use
`--jsonl-events` to consume the event stream programmatically. `python -m amplitude_lab serve`
starts a loopback-only browser UI using server-sent events (SSE). The page presents the plan as
step cards, keeps proposals separate from accepted derivation, and shows audit/backtracking
history and deterministic checks in distinct sections. Raw SSE events remain available in a
collapsed technical log for debugging. While the planner is running, the UI shows elapsed wait
time; providers can take time before sending their first visible token, and this does not expose
hidden model reasoning. Derivation steps request intermediate equations, explanations of signs
and factors, and relevant checks so the Markdown response is more useful than a final-formula
dump. The final UI reports the run's `response.md` and
`steps.md` files, saved alongside the full audit report in its run directory.

Install `.[openai-compatible]`, then configure `.env` from `.env.example`. By default, both
roles share `OPENAI_API_KEY` and `OPENAI_BASE_URL`; their model IDs remain independently
configurable as `DERIVATION_MODEL` and `AUDIT_MODEL`. If a role needs different credentials
or an endpoint, optional `DERIVATION_API_KEY` / `DERIVATION_BASE_URL` and
`AUDIT_API_KEY` / `AUDIT_BASE_URL` overrides are supported. `MAX_STEP_RETRIES` configures
correction attempts (default 3); `STREAMING_ENABLED` enables provider token streaming. API
keys remain on the server and are not included in prompts, reports, or SSE messages.

```powershell
python -m amplitude_lab derive
python -m amplitude_lab derive --max-step-retries 4 --jsonl-events
python -m amplitude_lab serve --port 8000
```

Set `LLM_TIMEOUT_SECONDS` to tune request timeouts; `--timeout-seconds` overrides it for one
run. `OPENAI_MAX_TOKENS` controls the provider completion token budget. Streaming depends on
the configured OpenAI-compatible endpoint's support; it does not override provider total
generation limits. For tasks without a configured deterministic reference verifier, the
result is explicitly labeled `completed_unverified`: an AI audit is not proof. For the
included `phi4_dim4` task, the final result is compared against the repository's deterministic
physics verifier. No live-model success is claimed until an actual model run has completed.
Optional `DERIVATION_REASONING_EFFORT` and `AUDIT_REASONING_EFFORT` values are forwarded as
the OpenAI-compatible `reasoning_effort` request parameter; leave them unset for providers
that do not support it.

OpenAI-compatible requests retry transient provider errors up to two times. If all attempts
fail, the run report and UI retain the provider error and suggest checking service status,
model availability, and endpoint configuration before retrying. Audit JSON is shown in
`result.md` and `steps.md` as a readable verdict, target, and explanation; the original
response remains available in `outcome.json`.

## Project layout

- `src/amplitude_lab/physics/`: amplitude model, kinematics, conventions, and benchmark
  problems.
- `src/amplitude_lab/integrals/` and `src/amplitude_lab/validation/`: integral implementations
  and deterministic checks.
- `src/amplitude_lab/assistant/` and `src/amplitude_lab/workflows/`: optional proposal
  assistant and LLM-backed derivation/review workflows.
- `src/amplitude_lab/cli.py`, `pipeline.py`, and `reporting.py`: application entry points,
  orchestration, and reports.
- `src/amplitude_lab/tasks/`: analytical task prompts.
- `examples/quickstart.py`: offline end-to-end example.
- `tests/`: tests grouped under `physics/`, `assistant/`, and `workflows/`.
- `docs/`: conventions, validation evidence, assistant trust model, design, and roadmap.

Runs and generated reports are local outputs and are ignored by Git.
