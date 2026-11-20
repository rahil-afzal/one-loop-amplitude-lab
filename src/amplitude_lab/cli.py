"""Command-line entry points for autonomous derivation and verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .workflows.autonomous import run_autonomous_derivation
from .workflows.llm import (
    DEFAULT_AUDIT_MODEL,
    DEFAULT_DERIVATION_MODEL,
    LLMError,
    role_llm_from_env,
)
from .workflows.task import Task
from .workflows.verification import verify_final_answer


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
        print(f"Response: {Path(result.run_dir) / 'response.md'}")
    return 0 if result.status in ("completed_verified", "completed_unverified") else 1


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

    verify = sub.add_parser("verify", help="check the final block of a saved derivation file")
    verify.add_argument("file")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
