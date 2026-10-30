"""Command-line entry points for the deterministic lab and analytical workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, conventions
from .amplitude import assemble
from .assistant.client import GeminiClient
from .assistant.config import GeminiConfig
from .assistant.errors import (
    AssistantConfigError,
    AssistantRequestError,
    AssistantUnavailableError,
)
from .assistant.mock import mock_client
from .benchmarks import resolve
from .derivation import run_derivation
from .kinematics import Kinematics
from .llm import LLMError, llm_from_env
from .pipeline import run_pipeline
from .reporting import write_json, write_markdown
from .task import Task
from .validation import list_checks, run_all

DERIVATION_MODEL = "anthropic/claude-sonnet-5.5"
AUDIT_MODEL = "anthropic/claude-opus-5-5"


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
        llm = llm_from_env(
            provider="openai-compatible",
            model=DERIVATION_MODEL,
            temperature=args.temperature,
            timeout=args.timeout_seconds,
        )
        audit_llm = llm_from_env(
            provider="openai-compatible",
            model=AUDIT_MODEL,
            temperature=args.temperature,
            timeout=args.timeout_seconds,
        )
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"task {task.name}, model {llm.name}; audit model {audit_llm.name}")
    result = run_derivation(task, llm, runs_dir=args.runs_dir, echo=print, audit_llm=audit_llm)
    minutes, seconds = divmod(int(result.elapsed_seconds), 60)
    elapsed_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    print(f"derivation workflow {result.status}; elapsed {elapsed_text}")
    print(f"report: {Path(result.run_dir) / 'result.md'}")
    if result.error:
        return 1
    print(
        "All derivation steps and the audit response were received; "
        "independently validate the result."
    )
    return 0


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
        "derive", help="derive an amplitude and request a second-pass model review"
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
    derive.set_defaults(func=cmd_derive)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
