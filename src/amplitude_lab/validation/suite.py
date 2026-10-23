"""Run the full deterministic validation suite."""

from __future__ import annotations

import importlib

from ..problem import ProblemSpec
from .base import REGISTRY, CheckResult, CheckSpec

# Explicit import order fixes the (reader-friendly) order of the registry.
for _module in ("checks_integral", "checks_amplitude", "checks_physics"):
    importlib.import_module(f"{__package__}.{_module}")


def list_checks() -> list[CheckSpec]:
    return list(REGISTRY)


def run_all(problem: ProblemSpec, only: list[str] | None = None) -> list[CheckResult]:
    selected = REGISTRY
    if only:
        unknown = set(only) - {spec.name for spec in REGISTRY}
        if unknown:
            raise ValueError(f"Unknown checks: {sorted(unknown)}")
        selected = [spec for spec in REGISTRY if spec.name in only]
    return [spec.run(problem) for spec in selected]
