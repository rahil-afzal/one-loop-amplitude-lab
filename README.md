# one-loop-amplitude-lab

A compact, testable lab for one-loop scattering amplitudes. The physics pipeline assembles
the amplitude and checks it against independent numerical and analytic evidence. Optional
LLM workflows either propose a bubble integral expression for safe parsing and deterministic
verification, or produce a written derivation followed by a separate model audit.

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

## Analytical derivation

The separate `derive` workflow makes a derivation request and a second model-audit request.
It stores the prompt, responses, metadata, and timings in `runs/`. The audit is model-generated,
not an independent proof, and its text is **not** submitted to the deterministic verifier.

For an OpenAI-compatible provider, install `.[openai-compatible]`, copy `.env.example` to
`.env`, then set `OPENAI_API_KEY` and the provider's exact `LLM_MODEL` ID. The default endpoint
is `https://tokens.bd/v1`; set `OPENAI_BASE_URL` for another compatible service. The default
audit model is `deepseek/deepseek-v4-flash-latest`; override it with `LLM_AUDIT_MODEL` or
`--audit-model`. Select Gemini with `LLM_PROVIDER=gemini` and the Gemini credentials.

```powershell
python -m amplitude_lab task --analytical
python -m amplitude_lab derive --model "<model-id>" --audit-model "<audit-model-id>"
```

Set `LLM_TIMEOUT_SECONDS` to change API request timeouts; `GEMINI_TIMEOUT_SECONDS` remains a
fallback for the older provider setting.

## Project layout

- `src/amplitude_lab/`: physics model, integral implementations, deterministic validation,
  optional proposal assistant, reporting, and CLI.
- `src/amplitude_lab/tasks/`: analytical task prompts.
- `examples/quickstart.py`: offline end-to-end example.
- `tests/`: physics, validation, assistant, CLI, and provider tests.
- `docs/`: conventions, validation evidence, assistant trust model, design, and roadmap.

Runs and generated reports are local outputs and are ignored by Git.
