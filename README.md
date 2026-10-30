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

## Analytical derivation

The separate `derive` workflow requests fourteen focused work steps, then a second-pass review.
Only relevant completed work is supplied to each next step in a fresh request. The report
assembles the saved steps directly, avoiding an extra model call that would only rephrase
them. Prompts, step responses, metadata, and timings are stored in `runs/`, including
completed steps if a later request fails. Both the derivation and review currently use
`qwen/qwen-3.8-max`; the review is a second response
from the same model, not an independent proof, and its text is **not** submitted to the
deterministic verifier.

The fixed model ID is set in `src/amplitude_lab/cli.py`;
`LLM_MODEL` and `LLM_AUDIT_MODEL` do not override them. Install `.[openai-compatible]`,
copy `.env.example` to `.env`, and set `OPENAI_API_KEY`. The default endpoint is
`https://tokens.bd/v1`; set `OPENAI_BASE_URL` for another compatible service.

```powershell
python -m amplitude_lab task --analytical
python -m amplitude_lab derive
python -m amplitude_lab derive --timeout-seconds 300
```

Set `LLM_TIMEOUT_SECONDS` to tune request timeouts in `.env`; `--timeout-seconds` overrides it
for one run. The default is 180 seconds. Set `OPENAI_STREAM=true` to consume streamed chunks
from compatible endpoints; this can prevent idle gateway timeouts while the model is
generating. Set it to `false` if the endpoint does not support streaming. Streaming does not
override a provider's hard total-generation limit, so keep the client timeout longer than the
expected request duration. `OPENAI_MAX_TOKENS` controls the completion token budget (default
4096); increase it only if the endpoint/model allows it and the response ends with
`finish_reason=length`. This is separate from the network timeout and streaming setting. This
fixed-model derivation currently uses the OpenAI-compatible endpoint configured above.

## Project layout

- `src/amplitude_lab/`: physics model, integral implementations, deterministic validation,
  optional proposal assistant, reporting, and CLI.
- `src/amplitude_lab/tasks/`: analytical task prompts.
- `examples/quickstart.py`: offline end-to-end example.
- `tests/`: physics, validation, assistant, CLI, and provider tests.
- `docs/`: conventions, validation evidence, assistant trust model, design, and roadmap.

Runs and generated reports are local outputs and are ignored by Git.
