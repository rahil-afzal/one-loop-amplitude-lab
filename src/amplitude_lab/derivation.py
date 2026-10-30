"""Stepwise analytical derivation and model-audit workflow."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .llm import LLMError

DERIVATION_SYSTEM = (
    "You are a careful quantum field theory researcher. For each requested step, start from "
    "the stated conventions and show the short algebraic chain that supports the result; do "
    "not quote a remembered final formula or jump ahead to later results. Keep notation "
    "consistent: distinguish a scalar integral containing only denominator factors from "
    "the complete diagram containing vertices and propagator factors, and carry every i and "
    "minus sign explicitly. Treat prior model work as fallible; if it conflicts with the "
    "conventions, recompute from the definitions and flag the conflict. Give checkable "
    "equations, not private chain-of-thought. State assumptions or uncertainty rather than "
    "guessing. Return concise Markdown and LaTeX only."
)

DERIVATION_CONTEXT = (
    "Task: massless real scalar phi^4 with L = 1/2 (d phi)^2 + c4 phi^4/4!, metric (+,-,-,-). "
    "Define iA as the sum of connected amputated diagrams. Derive the Feynman rules from L; "
    "they are not supplied here. Work in dimensional regularization with d=4-2 epsilon and "
    "the MS-bar scale mu, absorbing ln(4 pi)-gamma_E into the scale. Use "
    "s=(p1+p2)^2, t=(p1-p3)^2, u=(p1-p4)^2, s+t+u=0; physical region s+i0>0, t,u<0. "
    "Keep the propagator i factors outside any scalar denominator-only loop integral."
)

AUDIT_SYSTEM = (
    "You are a skeptical quantum field theory reviewer. Recheck the key normalizations and "
    "checks from the problem conventions; do not assume the proposed derivation is correct "
    "and do not merely paraphrase it. Give concise equations and checkable evidence, not "
    "private chain-of-thought. Do not write code. State whether the result is consistent, "
    "has specific errors, or cannot be verified."
)

AUDIT_REQUEST = """Review the derivation above against the original problem statement.

Check whether the derivation keeps the denominator-only loop integral distinct from the
propagator factors and complete diagram. Recompute the vertex and tree signs, the loop
integral's i and pole/finite terms, the bubble symmetry factor and channel coefficient, the
crossing sum, the physical s-channel branch and imaginary part, and the scale derivative.
For every PASS, provide the decisive equation or arithmetic; do not infer correctness just
because the final expression resembles a known result. Report PASS, FAIL, or UNCERTAIN for
each item. If the derivation omits a needed intermediate step, mark it UNCERTAIN rather than
filling the gap with an unsupported assertion. If you find an error, provide the corrected
formula and identify the first step where it enters. This model audit is a review, not a
mathematical proof or a substitute for the deterministic checks."""

DERIVATION_STEPS = (
    (
        "conventions",
        "Step 1 — Derive the four-point vertex and free propagator from the interaction and "
        "quadratic terms in the Lagrangian. Show how the 4! and the factors of i arise; do "
        "not merely list the rules. Limit the response to 80 words.",
    ),
    (
        "tree amplitude",
        "Step 2 — Describe the tree-level topology and multiply its vertex rule. Use the "
        "definition of iA to derive A_tree, including the sign, and explain why there is "
        "one contact diagram rather than a sum over three channels. Limit to 70 words.",
    ),
    (
        "bubble integral",
        "Step 3 — Define J_0(q^2) to be the scalar denominator-only loop integral for one "
        "massless bubble. Write its d=4-2 epsilon measure and both denominators. State "
        "explicitly whether the two propagator factors of i are included in J_0 or are "
        "reserved for the diagram. Do not evaluate the integral. Limit to 90 words.",
    ),
    (
        "Feynman parameter shift",
        "Step 4 — Starting from the J_0 definition, apply the Feynman-parameter identity "
        "and complete the square explicitly. Show the shifted momentum, the Jacobian, and "
        "how the sign of q^2 and the i0 prescription enter the denominator. Stop before "
        "integrating over loop momentum. Limit to 120 words.",
    ),
    (
        "loop momentum integral",
        "Step 5 — Evaluate the shifted loop-momentum integral using the dimensionally "
        "regulated Minkowski formula. Show the general Gamma(2-d/2) dependence before "
        "substituting d=4-2 epsilon, then substitute the shifted mass parameter into J_0. "
        "Track the factor of i and do not expand in epsilon. Limit to 140 words.",
    ),
    (
        "epsilon expansion",
        "Step 6 — Expand J_0 through O(epsilon^0), showing separately the Gamma/scale "
        "expansion and the parameter-log integral. Derive the parameter integral from "
        "ln[x(1-x)]=ln x+ln(1-x), and show how it contributes to the finite constant. Keep "
        "the pole, i factor, and logarithm branch visible. Limit to 170 words.",
    ),
    (
        "MS-bar subtraction",
        "Step 7 — Apply the specified MS-bar subtraction to J_0. State exactly which pole "
        "combination is removed under the scale convention, define the finite J_R(q^2), "
        "then define the dimensionless F by J_R=iF/(16 pi^2). Do not conflate J_R, F, and "
        "the full diagram. Limit to 130 words.",
    ),
    (
        "channel symmetry factor",
        "Step 8 — Count the identical internal-line interchange for one channel to obtain "
        "its symmetry factor. Then write the channel diagram before simplifying it, showing "
        "the two vertices, both propagator i factors, and J_R. Simplify only after displaying "
        "all factors; keep the distinction between iA_q and A_q explicit. Limit to 130 words.",
    ),
    (
        "channel contributions",
        "Step 9 — Substitute q^2=s, t, and u separately into the one-channel expression. "
        "Show all three contributions in terms of the same F function, preserving the "
        "analytic i0 prescription in each channel. Do not yet combine the terms or choose "
        "physical-region logarithm branches. Limit to 120 words.",
    ),
    (
        "crossing sum",
        "Step 10 — Add the three one-loop diagrams and the tree diagram. First show the "
        "sum for iA, then divide by i to obtain A. Collect the common coefficient and keep "
        "the logarithms in their unspecialized analytic form; do not select branches yet. "
        "Limit to 130 words.",
    ),
    (
        "logarithm branches",
        "Step 11 — Starting from the principal logarithm and the stated s+i0 prescription, "
        "derive the boundary values of the s-, t-, and u-channel logarithms in the physical "
        "region. Explain which arguments lie on the cut; do not quote branch values without "
        "showing how their arguments approach it. Limit to 120 words.",
    ),
    (
        "crossing check",
        "Step 12 — Check crossing symmetry on the analytic, unspecialized channel sum. "
        "Write the effect of an arbitrary permutation of s,t,u and explain why the same "
        "channel function and prescription are used. Distinguish this statement from the "
        "single physical-region branch choice. Limit to 90 words.",
    ),
    (
        "imaginary part check",
        "Step 13 — Extract Im A from the s-channel boundary value and show the coefficient "
        "multiplication that fixes its sign and normalization. As a cross-check, compare "
        "with the two-body unitarity cut only if you can state its phase-space and "
        "identical-particle factors explicitly; otherwise label that comparison unverified. "
        "Limit to 130 words.",
    ),
    (
        "scale dependence check",
        "Step 14 — Differentiate each channel term with respect to ln(mu) at fixed c4, "
        "showing the derivative of the logarithm before summing channels. Then infer the "
        "one-loop running of c4 required to cancel the explicit scale dependence. Do not "
        "insert a memorized beta function. Limit to 120 words.",
    ),
    (
        "final synthesis",
        "Final step — Reconstruct the derivation only from the completed steps: Feynman "
        "rules, tree graph, definition and evaluation of J_0, subtraction, channel factors, "
        "sum, branches, and checks. Keep J_R distinct from F and from the full diagram. "
        "Include the key equations that connect these stages and one boxed amplitude through "
        "O(c4^2). If any earlier steps conflict, resolve the conflict from the stated "
        "conventions and identify it; do not silently select a preferred answer. Do not add "
        "new checks or unsupported claims. Limit to 700 words.",
    ),
)

STEP_CONTEXT = {
    2: (0,),
    3: (0,),
    4: (2,),
    5: (2, 3),
    6: (4,),
    7: (5,),
    8: (0, 6),
    9: (6, 7),
    10: (1, 8),
    11: (9,),
    12: (9,),
    13: (9, 10),
    14: (9,),
    15: (0, 1, 6, 7, 8, 9, 10, 11, 12, 13),
}


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m {seconds}s" if minutes else f"{seconds}s"


class DerivationLog:
    def __init__(self, base: str | Path, task, llm, audit_llm=None):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(base) / f"{stamp}-{task.name}-derivation"
        suffix = 1
        while self.dir.exists():
            suffix += 1
            self.dir = Path(base) / f"{stamp}-{task.name}-derivation-{suffix}"
        self.dir.mkdir(parents=True)
        self.prompt = task.derivation_prompt
        meta = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": task.name,
            "workflow": "derive_then_model_audit",
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "llm": llm.describe() if hasattr(llm, "describe") else {"name": str(llm)},
            "audit_llm": (
                audit_llm.describe()
                if audit_llm is not None and hasattr(audit_llm, "describe")
                else {"name": getattr(audit_llm or llm, "name", "unknown")}
            ),
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (self.dir / "prompt.md").write_text(self.prompt, encoding="utf-8")

    def say(self, who: str, text: str) -> None:
        with (self.dir / "transcript.md").open("a", encoding="utf-8") as stream:
            stream.write(f"## {who}\n\n{text.rstrip()}\n\n")

    def event(self, **row) -> None:
        row["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with (self.dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def finish(
        self,
        result: "DerivationResult",
        derivation: str = "",
        audit: str = "",
        completed_steps: list[tuple[str, str]] | None = None,
    ) -> None:
        sections = ["# Analytical amplitude derivation", ""]
        if completed_steps:
            sections.extend(["## Completed work steps", ""])
            for label, response in completed_steps:
                sections.extend([f"### {label.title()}", "", response.rstrip(), ""])
        if derivation:
            sections.extend(["## Model derivation", "", derivation.rstrip(), ""])
        if audit:
            sections.extend(
                [
                    "## Audit response",
                    "",
                    (
                        "> This is a second response from the same model, not an independent "
                        "proof."
                        if result.model == result.audit_model
                        else "> This is a separate model response, not an independent proof."
                    ),
                    "",
                    audit.rstrip(),
                    "",
                ]
            )
        if result.error:
            sections.extend(["## Workflow error", "", result.error, ""])
        sections.extend(
            [
                "## Status",
                "",
                f"- Workflow: {result.status}",
                f"- Total elapsed: {_format_elapsed(result.elapsed_seconds)}",
                "",
                "A successful workflow means all derivation steps and the audit response were "
                "received. It does not "
                "establish that the amplitude is correct.",
                "",
            ]
        )
        (self.dir / "result.md").write_text("\n".join(sections), encoding="utf-8")
        (self.dir / "outcome.json").write_text(
            json.dumps(asdict(result), indent=2), encoding="utf-8"
        )


@dataclass
class DerivationResult:
    task: str
    model: str
    status: str
    run_dir: str
    audit_model: str = ""
    elapsed_seconds: float = 0.0
    derivation_seconds: float = 0.0
    audit_seconds: float = 0.0
    error: str = ""


def _request(llm, system: str, messages: list[dict[str, str]], label: str, progress):
    started = time.monotonic()
    done = threading.Event()

    def report_wait():
        while not done.wait(10):
            elapsed = time.monotonic() - started
            progress(f"Still waiting for {label} ({_format_elapsed(elapsed)})")

    progress(f"Requesting {label}")
    reporter = threading.Thread(target=report_wait, daemon=True)
    reporter.start()
    failure = None
    try:
        response = llm.complete(system, messages)
    except LLMError as exc:
        failure = exc
    finally:
        done.set()
        reporter.join()
    elapsed = time.monotonic() - started
    if failure:
        progress(f"{label.capitalize()} failed after {_format_elapsed(elapsed)}")
        raise failure
    progress(f"Received {label} after {_format_elapsed(elapsed)}")
    return response, elapsed


def run_derivation(task, llm, runs_dir="runs", echo=None, audit_llm=None) -> DerivationResult:
    """Request a stepwise derivation, then a second-pass model review; save all responses."""
    if not task.derivation_prompt.strip():
        raise ValueError(f"task {task.name!r} has no analytical derivation prompt")

    audit_llm = audit_llm or llm
    log = DerivationLog(runs_dir, task, llm, audit_llm)
    say = echo or (lambda message: None)
    started = time.monotonic()
    result = DerivationResult(
        task=task.name,
        model=getattr(llm, "name", "unknown"),
        status="failed",
        run_dir=str(log.dir),
        audit_model=getattr(audit_llm, "name", "unknown"),
    )
    derivation = ""
    audit = ""
    completed_steps: list[tuple[str, str]] = []
    current_stage = "derivation_step_1"

    try:
        for index, (label, instruction) in enumerate(DERIVATION_STEPS, start=1):
            current_stage = f"derivation_step_{index}"
            prompt_parts = [
                task.derivation_prompt if index == 1 else DERIVATION_CONTEXT,
                f"\n\nCurrent work request ({index}/{len(DERIVATION_STEPS)}):\n{instruction}",
            ]
            context_indices = STEP_CONTEXT.get(index, ())
            if context_indices:
                prior_work = "\n\n".join(
                    f"### Completed step: {completed_steps[prior_index][0]}\n"
                    f"{completed_steps[prior_index][1]}"
                    for prior_index in context_indices
                )
                prompt_parts.insert(1, f"\n\nCompleted prior work:\n{prior_work}")
            messages = [{"role": "user", "text": "".join(prompt_parts)}]
            response, elapsed = _request(
                llm, DERIVATION_SYSTEM, messages, f"step {index}: {label}", say
            )
            if not response.strip():
                raise LLMError(f"Derivation step {index} ({label}) returned an empty response.")
            result.derivation_seconds += elapsed
            completed_steps.append((label, response))
            log.say(f"derivation step {index}: {label}", response)
            log.event(
                stage=current_stage,
                kind="response",
                elapsed_seconds=round(elapsed, 2),
                response_chars=len(response),
            )

        derivation = completed_steps[-1][1]

        current_stage = "audit"
        derivation_record = "\n\n".join(
            f"### Step {index}: {label}\n{response.rstrip()}"
            for index, (label, response) in enumerate(completed_steps, start=1)
        )
        audit_messages = [
            {"role": "user", "text": task.derivation_prompt},
            {"role": "assistant", "text": derivation_record},
            {"role": "user", "text": AUDIT_REQUEST},
        ]
        audit, result.audit_seconds = _request(
            audit_llm, AUDIT_SYSTEM, audit_messages, "second-pass model review", say
        )
        if not audit.strip():
            raise LLMError("Second-pass model review returned an empty response.")
        log.say("model audit", audit)
        log.event(
            stage="audit",
            kind="response",
            elapsed_seconds=round(result.audit_seconds, 2),
            response_chars=len(audit),
        )
        result.status = "completed"
    except LLMError as exc:
        result.error = str(exc)
        log.say("error", result.error)
        log.event(
            stage=current_stage,
            kind="llm_error",
            error=result.error,
        )
        say(f"Request failed: {result.error}")
    finally:
        result.elapsed_seconds = time.monotonic() - started
        log.finish(result, derivation, audit, completed_steps)
        say(f"Saved analytical report to {log.dir}")

    return result
