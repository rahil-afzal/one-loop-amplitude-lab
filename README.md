# one-loop-amplitude-lab

Can an LLM agent derive a one-loop scattering amplitude by itself, if something else checks
the answer?

This repo is a test bed for that question. A task statement goes to the model. The model
works in a python/sympy scratch space and submits candidate solutions. A verifier made of
physics consistency checks (crossing, unitarity, dispersion, RG, ...) runs on each
submission and reports which checks failed. A reference amplitude exists too, but the model
never sees it; it backs one of the checks and is what the verifier itself was calibrated on.

The first task is the four-point amplitude of a massless phi^4 scalar to one loop in MS-bar.
It is deliberately easy: if an agent fails here, the harness is the first suspect, not the
physics. Published results (Ye-He-Gu on one-loop positivity bounds first) will be added as
further tasks to calibrate against.

## Status

Early. Done and tested: the verifier, the sandbox, the agent loop (against a scripted
model), the run logs. Not done: no real model has been run yet, the Gemini wrapper has only
been tested against a stub, and nothing has been run on Windows. The sandbox is hardened with
audit hooks and resource limits but is not a security boundary (see docs/design.md).

An independent review of the first version found real problems (the answer key was
importable from the sandbox, exception text leaked the test points, the checks sampled too
narrow a domain). These are fixed and each has a test; the review is summarised in
docs/design.md.

## Try it

    python -m pip install -e ".[dev]"
    python -m pytest                     # about two minutes: many tests start fresh interpreters

    python -m amplitude_lab task                                   # what the agent is told
    python -m amplitude_lab selftest                               # reference, 20 random seeds
    python -m amplitude_lab verify examples/candidates/correct.py
    python -m amplitude_lab verify examples/candidates/wrong_factor.py --details

`examples/candidates/` has one right solution and a handful of wrong ones (forgotten
symmetry factor, missing channel, wrong finite constant, ...). The tests check that each
wrong one trips the checks it should.

To let a model loose on the task:

    python -m pip install -e ".[gemini]"
    cp .env.example .env        # set GEMINI_API_KEY and GEMINI_MODEL
    python -m amplitude_lab agent

Each run writes `runs/<timestamp>-phi4_dim4/`: `meta.json` (task, statement hash, model and
temperature, library versions), every submission (`attempt_NN.py`), the verifier's results
with numbers and seed (`attempts.jsonl`), one row per turn (`events.jsonl`, including how
often the model used `run` blocks), the transcript and `outcome.json`.

For a run that matters, keep the answer key out of reach: create an interpreter that has only
sympy and mpmath (`python -m venv sandbox-env && sandbox-env/bin/pip install sympy mpmath`) and
point `AMPLITUDE_LAB_SANDBOX_PYTHON` at it, ideally inside a container.

## How a submission is judged

The model submits a python file with `amplitude(s, t, u, c4, mu2)`. The verifier evaluates it
at random complex points (wide ranges of scale and `mu2`, couplings of both signs, a fresh
seed for every submission, recorded in the log) and sees the candidate through nothing but
its values. The model is told which checks failed and gets a short fixed hint. It is never
shown numbers; if the file crashes it sees the exception type and the line, not the message,
because the message can quote the test points. If everything passes except the comparison
with the reference, the hint says so, and names both possible causes (a finite constant,
or numerical accuracy).

A run reports two outcomes separately: every structural check passed, and everything passed
including the reference.

The checks can't pin down everything: crossing, unitarity, the dispersion relation and RG
fix the logarithms and the imaginary part, not the scheme-dependent constant. That is what
`reference_match` is for. More in docs/design.md.

## Layout

    src/amplitude_lab/
      tasks/phi4_dim4/   statement.md (what the agent sees) and task.json
      checks.py          the consistency checks
      verify.py          runs them, builds the feedback
      candidate.py       loads a submission, in-process or in a subprocess
      sandbox.py         subprocess runner: import limits, audit hook, resource limits
      agent.py           the turn loop
      llm.py             Gemini client and a scripted one for tests
      log.py             run directories
      physics.py         tree-level inputs (phase space, beta coefficient)
      reference.py       the answer key
    examples/candidates/ right and wrong solutions
    tests/

## To do

- second task with derivative interactions (dim-6) once the paper is in
- `physics.py` hard-codes the single-scalar tree data; for new models it has to come from
  the task spec
- real runs: several seeds per task, with and without feedback
- test the sandbox on Windows; run agents in a container
- an independent route for new tasks (FeynCalc or Package-X) to catch typos in a reference

MIT licensed.
