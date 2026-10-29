"""Two-pass analytical derivation and model-audit workflow."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .llm import LLMError

DERIVATION_SYSTEM = (
    "You are a careful quantum field theory researcher. Produce a concise, auditable "
    "derivation with equations needed to check the result, not private chain-of-thought. "
    "Follow the stated conventions exactly. Distinguish derived results from assumptions "
    "and state uncertainty instead of guessing. Return Markdown and LaTeX only."
)

AUDIT_SYSTEM = (
    "You are an independent skeptical quantum field theory auditor. Recompute the key "
    "normalizations and checks from the problem conventions; do not assume the proposed "
    "derivation is correct and do not merely paraphrase it. Give concise equations and "
    "checkable evidence, not private chain-of-thought. Do not write code. State clearly "
    "whether the result is consistent, has specific errors, or cannot be verified."
)

AUDIT_REQUEST = """Audit the derivation above against the original problem statement.

Independently check the vertex/sign convention, each bubble symmetry factor and coefficient,
the MS-bar finite term, the crossing-symmetric channel sum, the physical s-channel logarithm
branch and imaginary part, and the renormalization-scale dependence. Do not simply accept
claims made in the derivation. Report each check as PASS, FAIL, or UNCERTAIN with a concise
equation or reason. If you find an error, provide the corrected boxed amplitude and explain
which step changed it. If you cannot establish a check, say so. This model audit is not a
mathematical proof or a substitute for an independent calculation."""


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m {seconds}s" if minutes else f"{seconds}s"


class DerivationLog:
    def __init__(self, base: str | Path, task, llm):
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

    def finish(self, result: "DerivationResult", derivation: str = "", audit: str = "") -> None:
        sections = ["# Analytical amplitude derivation", ""]
        if derivation:
            sections.extend(["## Model derivation", "", derivation.rstrip(), ""])
        if audit:
            sections.extend(
                [
                    "## Separate model audit",
                    "",
                    "> This is a second model response, not an independent proof.",
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
                "A successful workflow means both model responses were received. It does not "
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


def run_derivation(task, llm, runs_dir="runs", echo=None) -> DerivationResult:
    """Request a derivation and then a separate model audit; save both in a report."""
    if not task.derivation_prompt.strip():
        raise ValueError(f"task {task.name!r} has no analytical derivation prompt")

    log = DerivationLog(runs_dir, task, llm)
    say = echo or (lambda message: None)
    started = time.monotonic()
    result = DerivationResult(
        task=task.name,
        model=getattr(llm, "name", "unknown"),
        status="failed",
        run_dir=str(log.dir),
    )
    derivation = ""
    audit = ""
    messages = [{"role": "user", "text": task.derivation_prompt}]

    try:
        derivation, result.derivation_seconds = _request(
            llm, DERIVATION_SYSTEM, messages, "analytical derivation", say
        )
        if not derivation.strip():
            raise LLMError("Analytical derivation returned an empty response.")
        log.say("model derivation", derivation)
        log.event(
            stage="derivation",
            kind="response",
            elapsed_seconds=round(result.derivation_seconds, 2),
            response_chars=len(derivation),
        )

        audit_messages = [
            *messages,
            {"role": "assistant", "text": derivation},
            {"role": "user", "text": AUDIT_REQUEST},
        ]
        audit, result.audit_seconds = _request(
            llm, AUDIT_SYSTEM, audit_messages, "separate model audit", say
        )
        if not audit.strip():
            raise LLMError("Separate model audit returned an empty response.")
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
            stage="audit" if derivation else "derivation",
            kind="llm_error",
            error=result.error,
        )
        say(f"Request failed: {result.error}")
    finally:
        result.elapsed_seconds = time.monotonic() - started
        log.finish(result, derivation, audit)
        say(f"Saved analytical report to {log.dir}")

    return result
