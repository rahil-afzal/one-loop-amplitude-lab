"""End-to-end pipeline: problem -> amplitudes -> deterministic checks -> report."""

from __future__ import annotations

from .amplitude import assemble
from .assistant.client import AssistantClient
from .assistant.orchestrator import run_task
from .assistant.tasks import BubbleClosedFormTask
from .kinematics import BENCHMARK_POINTS
from .problem import ProblemSpec
from .reporting import build_report
from .validation import run_all


def run_pipeline(problem: ProblemSpec, assistant: AssistantClient | None = None) -> dict:
    """Run everything. The overall status depends ONLY on the deterministic checks."""
    amplitudes = [
        {
            "kinematics": {"s": kin.s, "t": kin.t, "u": kin.u},
            "result": assemble(problem, kin).to_dict(),
        }
        for kin in BENCHMARK_POINTS
    ]
    checks = run_all(problem)
    assistant_reports = []
    if assistant is not None:
        assistant_reports.append(run_task(assistant, BubbleClosedFormTask(), problem).to_dict())
    return build_report(problem, amplitudes, checks, assistant_reports)
