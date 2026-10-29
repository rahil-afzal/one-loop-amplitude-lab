import argparse
import sys

from . import reference
from .agent import run_agent
from .candidate import Candidate
from .llm import GeminiLLM, LLMError
from .task import Task
from .verify import verify


def _print_report(report, details):
    print(report.feedback())
    if details:
        print(f"seed {report.seed}")
        for r in report.results:
            print(f"  {r.name}: {r.detail}")


def cmd_task(args):
    print(Task.load(args.task).statement)
    return 0


def cmd_verify(args):
    task = Task.load(args.task)
    candidate = Candidate(path=args.file, isolated=not args.trusted, timeout=task.eval_timeout)
    report = verify(candidate, suite=task.suite, seed=args.seed)
    _print_report(report, args.details)
    return 0 if report.passed else 1


def cmd_selftest(args):
    task = Task.load(args.task)
    candidate = Candidate.from_function(reference.amplitude)
    bad = 0
    for seed in range(args.seeds):
        report = verify(candidate, suite=task.suite, seed=seed)
        bad += not report.passed
        if args.details or not report.passed:
            _print_report(report, args.details)
    print(f"reference passed {args.seeds - bad} of {args.seeds} seeds")
    return 0 if bad == 0 else 1


def cmd_agent(args):
    task = Task.load(args.task)
    try:
        llm = GeminiLLM.from_env(model=args.model, temperature=args.temperature)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"task {task.name}, model {llm.name}")
    result = run_agent(task, llm, runs_dir=args.runs_dir, echo=print)
    state = "solved" if result.solved else "not solved"
    elapsed = result.elapsed_seconds
    minutes, seconds = divmod(int(elapsed), 60)
    elapsed_text = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    print(
        f"{state} ({result.reason}); {result.attempts} submissions, {result.turns} turns; "
        f"elapsed {elapsed_text}"
    )
    if result.structural_solved and not result.solved:
        print("every structural check passed at some point; the reference match did not")
    print(f"log: {result.run_dir}")
    return 0 if result.solved else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="amplitude-lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("task", help="print the task statement the agent sees")
    t.add_argument("--task", default="phi4_dim4")
    t.set_defaults(func=cmd_task)

    v = sub.add_parser("verify", help="run the checks on a candidate file")
    v.add_argument("file")
    v.add_argument("--task", default="phi4_dim4")
    v.add_argument("--seed", type=int, default=None, help="fix the random check points")
    v.add_argument("--trusted", action="store_true", help="import in-process (your own code only)")
    v.add_argument(
        "--details", action="store_true", help="also print numbers (not shown to agents)"
    )
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("selftest", help="run the checks on the reference amplitude, several seeds")
    s.add_argument("--task", default="phi4_dim4")
    s.add_argument("--seeds", type=int, default=20)
    s.add_argument("--details", action="store_true")
    s.set_defaults(func=cmd_selftest)

    a = sub.add_parser("agent", help="let a Gemini model work on the task")
    a.add_argument("--task", default="phi4_dim4")
    a.add_argument("--model", default=None)
    a.add_argument("--temperature", type=float, default=0.7)
    a.add_argument("--runs-dir", default="runs")
    a.set_defaults(func=cmd_agent)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
