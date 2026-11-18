# one-loop-amplitude-lab

A small, testable workspace for one-loop scattering amplitudes.

The project:

- assembles a benchmark amplitude and runs deterministic physics checks;
- separates independent numerical, analytic, and consistency evidence;
- optionally uses an LLM to propose or derive expressions;
- records derivation attempts and audits for inspection.

The supported benchmark is the massless two-scalar dimension-four sector at one loop
in MS-bar. See [Scope and roadmap](docs/ROADMAP.md) and
[Physics conventions](docs/CONVENTIONS.md) for limitations.

## Quick start

### 1. Install

Create an editable development install:

```powershell
python -m pip install -e ".[dev]"
```

### 2. Try the offline example

```powershell
python examples/quickstart.py
```

### 3. Run deterministic checks

```powershell
python -m amplitude_lab list-checks
python -m amplitude_lab validate --problem yhg_dim4
python -m amplitude_lab run --problem yhg_dim4 --out-dir runs/yhg_dim4
```

`validate` prints each result directly. `run` writes `run.json` and `run.md` in the
selected output directory.

### 4. Run the tests

```powershell
python -m pytest
```

All core commands and tests run offline. No API key is required for the quick start.

## Common commands

Evaluate the amplitude at one kinematic point:

```powershell
python -m amplitude_lab amplitude --problem yhg_dim4 --s 5 --t -2
```

Run a single named check:

```powershell
python -m amplitude_lab validate --problem yhg_dim4 --only integral_momentum_space
```

Try an offline assistant proposal:

```powershell
python -m amplitude_lab run --problem yhg_dim4 --assistant mock-correct --out-dir runs/mock
```

Mock proposals exercise parsing and verification only.

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

Browser interface:

```powershell
python -m amplitude_lab serve --port 8000
```

The browser UI runs on loopback and uses server-sent events. Providers may take time
before returning the first visible token.

### Saved outputs

Derivation runs are written under `runs/`:

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
| `run` | Run deterministic checks and write JSON/Markdown reports |
| `validate` | Run selected deterministic checks |
| `amplitude` | Evaluate the amplitude at one kinematic point |
| `describe` | Print supported physics conventions |
| `list-checks` | List available validation checks |
| `task` | Print an analytical derivation prompt |
| `derive` | Derive and independently audit each step |
| `refine` | Run a derivation/audit revision loop |
| `serve` | Start the loopback browser interface |
| `verify` | Verify the final block of a saved derivation |

Run `python -m amplitude_lab --help` or `python -m amplitude_lab <command> --help`
for all options.

## Documentation

- [Validation strategy](docs/VALIDATION.md): what each check proves and its evidence type
- [Physics conventions](docs/CONVENTIONS.md): signs, prefactors, kinematics, and units
- [Assistant trust model](docs/ASSISTANT.md): where LLM output is accepted and rejected
- [Design](docs/design.md): architecture and workflow
- [Roadmap](docs/ROADMAP.md): supported scope and next milestones
- [Contributing](CONTRIBUTING.md): development and validation rules

## Project layout

- `src/amplitude_lab/physics/`: amplitude model, kinematics, conventions, and benchmarks
- `src/amplitude_lab/integrals/` and `src/amplitude_lab/validation/`: implementations and checks
- `src/amplitude_lab/assistant/`: optional proposal clients
- `src/amplitude_lab/workflows/`: derivation, audit, and browser workflows
- `examples/quickstart.py`: offline API example
- `tests/`: physics, assistant, and workflow tests
- `docs/`: conventions, validation, design, and roadmap
