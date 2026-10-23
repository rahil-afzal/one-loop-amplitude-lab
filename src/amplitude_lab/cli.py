"""Command-line interface.

Exit codes: 0 success, 1 validation (or request) failure, 2 configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, benchmarks, conventions
from .amplitude import assemble
from .assistant.config import GeminiConfig
from .assistant.errors import AssistantConfigError, AssistantRequestError, AssistantUnavailableError
from .assistant.mock import MOCK_PROPOSALS, mock_client
from .kinematics import Kinematics
from .pipeline import run_pipeline
from .reporting import write_json, write_markdown
from .validation import list_checks, run_all

ASSISTANT_CHOICES = ["none", "gemini", *[f"mock-{k}" for k in MOCK_PROPOSALS]]


def _problem_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--problem",
        default="yhg_dim4",
        help=f"built-in name ({', '.join(benchmarks.BUILTIN)}) or JSON path (default: yhg_dim4)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amplitude-lab", description="Validated one-loop scalar scattering amplitudes."
    )
    parser.add_argument("--version", action="version", version=f"amplitude-lab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Full pipeline: amplitudes, checks, JSON + Markdown reports.")
    _problem_arg(run)
    run.add_argument("--out-dir", default="reports")
    run.add_argument("--assistant", choices=ASSISTANT_CHOICES, default="none")

    val = sub.add_parser("validate", help="Run the deterministic checks and print a table.")
    _problem_arg(val)
    val.add_argument("--only", nargs="*", help="run only the named checks")

    amp = sub.add_parser("amplitude", help="Evaluate the one-loop amplitude at (s, t).")
    _problem_arg(amp)
    amp.add_argument("--s", type=float, required=True)
    amp.add_argument("--t", type=float, required=True)

    sub.add_parser("describe", help="Print the physics conventions as JSON.")
    sub.add_parser("list-checks", help="List the validation checks and their evidence kinds.")

    smoke = sub.add_parser("smoke", help="Manual, real Gemini connectivity test (network).")
    smoke.add_argument("--model", default=None)
    return parser


def _make_assistant(choice: str):
    if choice == "none":
        return None
    if choice.startswith("mock-"):
        return mock_client(choice.removeprefix("mock-"))
    from .assistant.client import GeminiClient

    config = GeminiConfig.from_env(load_dotenv=True)
    config.require_key()
    return GeminiClient(config)


def _cmd_run(args: argparse.Namespace) -> int:
    problem = benchmarks.resolve(args.problem)
    try:
        assistant = _make_assistant(args.assistant)
    except AssistantConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    report = run_pipeline(problem, assistant)
    out = Path(args.out_dir)
    write_json(report, out / "run.json")
    write_markdown(report, out / "run.md")

    print(f"Problem: {problem.name}")
    print(f"Overall status: {report['overall_status'].upper()}")
    for check in report["checks"]:
        print(f"  [{check['status'].upper():5s}] {check['name']} ({check['evidence']})")
    for item in report["assistant"]:
        print(f"  assistant {item['task']}: {item['status']}")
    print(f"Reports written to {out / 'run.json'} and {out / 'run.md'}")
    return 0 if report["overall_status"] == "pass" else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    problem = benchmarks.resolve(args.problem)
    results = run_all(problem, only=args.only)
    for r in results:
        print(f"[{r.status.upper():5s}] {r.name:34s} {r.evidence:15s} {r.duration_s:6.2f}s")
        if not r.passed:
            print(f"        {r.message}")
    return 0 if all(r.passed for r in results) else 1


def _cmd_amplitude(args: argparse.Namespace) -> int:
    problem = benchmarks.resolve(args.problem)
    result = assemble(problem, Kinematics.from_s_t(args.s, args.t))
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from .assistant.client import GeminiClient

    try:
        config = GeminiConfig.from_env(load_dotenv=True)
        if args.model:
            config.model = args.model
        text = GeminiClient(config).smoke_test()
    except AssistantConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except (AssistantUnavailableError, AssistantRequestError) as exc:
        print(f"Gemini smoke test failed: {exc}", file=sys.stderr)
        return 1
    print(f"Gemini smoke test OK. Response: {text[:120]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "amplitude":
            return _cmd_amplitude(args)
        if args.command == "describe":
            print(json.dumps(conventions.describe(), indent=2))
            return 0
        if args.command == "list-checks":
            for spec in list_checks():
                print(f"{spec.name:34s} {spec.evidence:15s} {spec.description}")
            return 0
        if args.command == "smoke":
            return _cmd_smoke(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
