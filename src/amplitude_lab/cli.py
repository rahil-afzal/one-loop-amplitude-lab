"""Command-line entry points for the deterministic lab and analytical workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .assistant.client import GeminiClient
from .assistant.config import GeminiConfig
from .assistant.errors import (
    AssistantConfigError,
    AssistantRequestError,
    AssistantUnavailableError,
)
from .assistant.mock import mock_client
from .physics import conventions
from .physics.amplitude import assemble
from .physics.benchmarks import resolve
from .physics.kinematics import Kinematics
from .pipeline import run_pipeline
from .reporting import write_json, write_markdown
from .validation import list_checks, run_all
from .workflows.autonomous import run_autonomous_derivation
from .workflows.feedback import run_feedback_loop
from .workflows.llm import (
    DEFAULT_AUDIT_MODEL,
    DEFAULT_DERIVATION_MODEL,
    LLMError,
    role_llm_from_env,
)
from .workflows.server import serve
from .workflows.task import Task
from .workflows.verification import verify_final_answer


def _assistant_from_name(name):
    if name is None:
        return None
    if name == "gemini":
        config = GeminiConfig.from_env(load_dotenv=True)
        config.require_key()
        return GeminiClient(config)
    if name.startswith("mock-"):
        return mock_client(name.removeprefix("mock-"))
    raise ValueError(f"Unknown assistant {name!r}; use 'gemini' or 'mock-<proposal>'.")


def cmd_run(args):
    try:
        problem = resolve(args.problem)
        assistant = _assistant_from_name(args.assistant)
        report = run_pipeline(problem, assistant)
        out_dir = Path(args.out_dir)
        write_json(report, out_dir / "run.json")
        write_markdown(report, out_dir / "run.md")
    except (AssistantConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Overall status: {report['overall_status'].upper()}")
    print(f"report: {out_dir / 'run.md'}")
    return 0 if report["overall_status"] == "pass" else 1


def cmd_smoke(_args):
    try:
        client = GeminiClient(GeminiConfig.from_env(load_dotenv=True))
        print(client.smoke_test())
    except (
        AssistantConfigError,
        AssistantRequestError,
        AssistantUnavailableError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_validate(args):
    try:
        problem = resolve(args.problem)
        results = run_all(problem, only=args.only)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name} [{result.evidence}]: {result.message}")
    return 0 if all(result.passed for result in results) else 1


def cmd_amplitude(args):
    try:
        problem = resolve(args.problem)
        result = assemble(problem, Kinematics.from_s_t(args.s, args.t))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_describe(_args):
    print(json.dumps(conventions.describe(), indent=2))
    return 0


def cmd_list_checks(_args):
    for check in list_checks():
        print(f"{check.name} [{check.evidence}]: {check.description}")
    return 0


def cmd_task(args):
    try:
        task = Task.load(args.task)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not task.derivation_prompt.strip():
        print(f"error: task {task.name!r} has no analytical derivation prompt", file=sys.stderr)
        return 2
    print(task.derivation_prompt)
    return 0


def cmd_derive(args):
    try:
        task = Task.load(args.task)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not task.derivation_prompt.strip():
        print(f"error: task {task.name!r} has no analytical derivation prompt", file=sys.stderr)
        return 2
    try:
        derivation_llm = role_llm_from_env(
            "derivation",
            DEFAULT_DERIVATION_MODEL,
            temperature=args.temperature,
            timeout=args.timeout_seconds,
        )
        audit_llm = role_llm_from_env(
            "audit",
            DEFAULT_AUDIT_MODEL,
            temperature=args.temperature,
            timeout=args.timeout_seconds,
        )
        max_retries = (
            args.max_step_retries
            if args.max_step_retries is not None
            else int(os.environ.get("MAX_STEP_RETRIES", "3"))
        )
        if max_retries < 0:
            raise ValueError("MAX_STEP_RETRIES must be non-negative.")
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def show_event(event):
        if args.jsonl_events:
            print(json.dumps(event, ensure_ascii=False), flush=True)
        elif event["type"] == "token":
            print(event.get("text", ""), end="", flush=True)
        else:
            detail = event.get("diagnostic") or event.get("message") or event["type"]
            step = f" ({event['step_id']})" if event.get("step_id") else ""
            print(f"\n[{event['state']}] {detail}{step}", flush=True)

    result = run_autonomous_derivation(
        task,
        derivation_llm,
        audit_llm,
        runs_dir=args.runs_dir,
        max_step_retries=max_retries,
        on_event=show_event,
    )
    if not args.jsonl_events:
        print(f"\nDerivation workflow {result.status}.")
        print(f"Report: {Path(result.run_dir) / 'result.md'}")
    return 0 if result.status in ("completed_verified", "completed_unverified") else 1


def cmd_refine(args):
    try:
        task = Task.load(args.task)
        llm = role_llm_from_env(
            "derivation",
            DEFAULT_DERIVATION_MODEL,
            temperature=args.temperature,
            timeout=args.timeout_seconds,
        )
        audit_llm = role_llm_from_env(
            "audit",
            DEFAULT_AUDIT_MODEL,
            temperature=args.temperature,
            timeout=args.timeout_seconds,
        )
    except (FileNotFoundError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"task {task.name}, model {llm.name}; audit model {audit_llm.name}")
    result = run_feedback_loop(
        task, llm, audit_llm, runs_dir=args.runs_dir, max_rounds=args.max_rounds, echo=print
    )
    print(f"feedback loop {result.status}; rounds {len(result.rounds)}; report: {result.run_dir}")
    print(
        f"deterministic verification of final round: "
        f"{'PASS' if result.final_verification_passed else 'FAIL'}"
    )
    return 0 if result.status != "failed" and result.final_verification_passed else 1


def cmd_serve(args):
    try:
        serve(port=args.port)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_verify(args):
    try:
        text = Path(args.file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = verify_final_answer(text)
    print(report.summary())
    return 0 if report.passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="amplitude-lab")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run deterministic checks and write a report")
    run.add_argument("--problem", default="yhg_dim4")
    run.add_argument("--assistant", default=None, help="optional Gemini or mock assistant")
    run.add_argument("--out-dir", default="runs/latest")
    run.set_defaults(func=cmd_run)

    smoke = sub.add_parser("smoke", help="test Gemini API connectivity")
    smoke.set_defaults(func=cmd_smoke)

    validate = sub.add_parser("validate", help="run selected deterministic checks")
    validate.add_argument("--problem", default="yhg_dim4")
    validate.add_argument("--only", action="append", default=None)
    validate.set_defaults(func=cmd_validate)

    amplitude = sub.add_parser("amplitude", help="evaluate the amplitude at one kinematic point")
    amplitude.add_argument("--problem", default="yhg_dim4")
    amplitude.add_argument("--s", type=float, required=True)
    amplitude.add_argument("--t", type=float, required=True)
    amplitude.set_defaults(func=cmd_amplitude)

    describe = sub.add_parser("describe", help="print the supported physics conventions")
    describe.set_defaults(func=cmd_describe)

    checks = sub.add_parser("list-checks", help="list deterministic checks")
    checks.set_defaults(func=cmd_list_checks)

    task = sub.add_parser("task", help="print the analytical derivation prompt")
    task.add_argument("--task", default="phi4_dim4")
    task.add_argument("--analytical", action="store_true", help=argparse.SUPPRESS)
    task.set_defaults(func=cmd_task)

    derive = sub.add_parser(
        "derive", help="autonomously derive and audit each step with live progress"
    )
    derive.add_argument("--task", default="phi4_dim4")
    derive.add_argument("--temperature", type=float, default=0.2)
    derive.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="request timeout; overrides LLM_TIMEOUT_SECONDS",
    )
    derive.add_argument("--runs-dir", default="runs")
    derive.add_argument(
        "--max-step-retries",
        type=int,
        default=None,
        help="correction attempts per step; defaults to MAX_STEP_RETRIES or 3",
    )
    derive.add_argument(
        "--jsonl-events",
        action="store_true",
        help="write the live event stream as JSON Lines to stdout",
    )
    derive.set_defaults(func=cmd_derive)

    web = sub.add_parser("serve", help="start the loopback live derivation SSE frontend")
    web.add_argument("--port", type=int, default=8000)
    web.set_defaults(func=cmd_serve)

    refine = sub.add_parser(
        "refine", help="derive, audit, revise in a loop; verify deterministically each round"
    )
    refine.add_argument("--task", default="phi4_dim4")
    refine.add_argument("--temperature", type=float, default=0.2)
    refine.add_argument("--timeout-seconds", type=float, default=None)
    refine.add_argument("--max-rounds", type=int, default=3)
    refine.add_argument("--runs-dir", default="runs")
    refine.set_defaults(func=cmd_refine)

    verify = sub.add_parser("verify", help="check the final block of a saved derivation file")
    verify.add_argument("file")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
