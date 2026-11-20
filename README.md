# one-loop-amplitude-lab

A small, testable workspace for one-loop scattering amplitudes.

The project:

- runs checkpointed autonomous derivations with separate planning, derivation, and audit models;
- prints packaged analytical task prompts;
- verifies saved final answers against the retained repository amplitude implementation;
- records derivation attempts and audit checkpoints for inspection.

The supported benchmark is the massless two-scalar dimension-four sector at one loop
in MS-bar. See [Scope and roadmap](docs/ROADMAP.md) and
[Physics conventions](docs/CONVENTIONS.md) for limitations.

## Quick start

### 1. Install

Create an editable development install:

```powershell
python -m pip install -e ".[dev]"
```

### 2. Print a task prompt

```powershell
python -m amplitude_lab task --task phi4_dim4
```

### 3. Verify a saved derivation

```powershell
python -m amplitude_lab verify runs/<run-name>/response.md
```

### 4. Run the tests

```powershell
python -m pytest
```

Core task and verify commands run offline. No API key is required for task printing or
deterministic verification.

Keep credentials in environment variables or a local `.env` file. Do not commit API keys.

## Optional LLM derivation

The `derive` workflow sends a task prompt to separate derivation and audit models.
It proposes one checkpoint at a time and audits each step independently. If a missing
substep is discovered, the workflow can insert and audit it without rebuilding the whole
plan. Rejected work is retried at its checkpoint; only dependent checkpoints are invalidated,
while accepted independent work is retained. The workflow runs deterministic verification
when a reference is available.

An AI audit is not a proof:

- `completed_verified` means the final answer passed a repository verifier;
- `completed_unverified` means no deterministic verifier was configured;
- a failed or unresolved step is never silently accepted.

### Set up an OpenAI-compatible provider

```powershell
python -m pip install -e ".[openai-compatible]"
Copy-Item .env.example .env
```

Edit `.env` and set at least:

```dotenv
OPENAI_API_KEY=your-provider-api-key
OPENAI_BASE_URL=https://your-provider.example/v1
DERIVATION_MODEL=your-derivation-model
AUDIT_MODEL=your-audit-model
```

Both roles share `OPENAI_API_KEY` and `OPENAI_BASE_URL` unless overridden with
`DERIVATION_API_KEY`, `DERIVATION_BASE_URL`, `AUDIT_API_KEY`, or `AUDIT_BASE_URL`.

Other useful settings:

- `MAX_STEP_RETRIES`: correction attempts per step, default `3`;
- `LLM_TIMEOUT_SECONDS`: provider timeout, default `600`;
- `OPENAI_MAX_TOKENS`: requested completion limit, default `128000`;
- `STREAMING_ENABLED`: enable provider token streaming;
- `DERIVATION_REASONING_EFFORT` and `AUDIT_REASONING_EFFORT`: optional provider parameters.

### Run a derivation

Terminal output:

```powershell
python -m amplitude_lab derive
python -m amplitude_lab derive --max-step-retries 4
```

Machine-readable events:

```powershell
python -m amplitude_lab derive --jsonl-events
```

### Saved outputs

Derivation runs are written under `runs/`:

- Markdown reports refresh after committed checkpoints and include partial accepted
  substeps, rejected attempts, and workflow failures. `checkpoints.json` records the
  committed state if a run is interrupted.
- `response.md`: generated final response;
- `steps.md`: accepted derivation steps and audit summaries;
- `result.md`: status, deterministic checks, and final report;
- `checkpoints.json`: checkpoint state and attempt history;
- `prompt.md`, `outcome.json`, and `plan.json`: task, result, and planning records.

Verify a saved derivation:

```powershell
python -m amplitude_lab verify runs/<run-name>/response.md
```

## CLI reference

| Command | Purpose |
|---|---|
| `task` | Print an analytical derivation prompt |
| `derive` | Derive and independently audit each step |
| `verify` | Verify the final block of a saved derivation |

Run `python -m amplitude_lab --help` or `python -m amplitude_lab <command> --help`
for all options.

## Documentation

- [Verification strategy](docs/VALIDATION.md): what the final-answer verifier proves and its limits
- [Physics conventions](docs/CONVENTIONS.md): signs, prefactors, kinematics, and units
- [Design](docs/design.md): architecture and workflow
- [Roadmap](docs/ROADMAP.md): supported scope and next milestones
- [Contributing](CONTRIBUTING.md): development and validation rules

## Project layout

- `src/amplitude_lab/physics/`: amplitude model, kinematics, conventions, and benchmarks
- `src/amplitude_lab/integrals/`: retained bubble implementation
- `src/amplitude_lab/workflows/`: autonomous derivation, task loading, audit, provider, and verification
- `src/amplitude_lab/safe_expr.py`: strict final-expression parser
- `tests/`: physics, parser, and workflow tests
- `docs/`: conventions, validation, design, and roadmap
