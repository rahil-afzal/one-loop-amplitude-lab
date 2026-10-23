"""Machine-readable (JSON) and human-readable (Markdown) reports."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import mpmath
import numpy
import scipy
import sympy

from . import __version__, conventions
from .integrals.spec import massless_bubble
from .problem import ProblemSpec
from .validation.base import CheckResult

SCHEMA_VERSION = "1.0"

LIMITATIONS = [
    "Only the massless two-scalar dim-4 sector, process phi1 phi1 -> phi1 phi1, is supported.",
    "Only the finite MSbar one-loop bubble amplitude is computed; no UV-pole matching.",
    "No dim-6/dim-8 operators, no massive propagators, triangles, boxes, tensor numerators.",
    "The dispersion benchmark is the dim-4 contribution (Eq. 3.11 subset), not the full Eq. (3.9).",
    "The forward real amplitude diverges like ln(-t); only its imaginary part is used.",
    "Finite numerical agreement is not a proof of global analytic correctness.",
    "The beta function is a textbook reference, not derived independently by this code.",
    "Assistant proposals are never ground truth; 'verified' means a finite-sample test passed.",
    "Gemini model IDs and live API behaviour were not verified by the authors of this release.",
]

UNVERIFIED = [
    "Equation numbers and the meaning of c_1^[42] against the Ye-He-Gu paper (re-check the PDF).",
    "Bare-to-renormalized matching and counterterm derivation.",
    "One-loop dim-8 mixing and beta_1^[8] contributions to Sigma.",
    "Any claim about general one-loop 2 -> 2 amplitudes beyond the supported topology.",
]


def environment() -> dict:
    return {
        "amplitude_lab": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sympy": sympy.__version__,
        "mpmath": mpmath.__version__,
        "platform": platform.platform(),
    }


def build_report(
    problem: ProblemSpec,
    amplitudes: list[dict],
    checks: list[CheckResult],
    assistant: list[dict] | None = None,
) -> dict:
    failed = [c.name for c in checks if not c.passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment(),
        "problem": problem.to_dict(),
        "conventions": conventions.describe(),
        "integral": massless_bubble().to_dict(),
        "amplitudes": amplitudes,
        "checks": [c.to_dict() for c in checks],
        "overall_status": "fail" if failed or not checks else "pass",
        "failed_checks": failed,
        "assistant": assistant if assistant else [],
        "limitations": LIMITATIONS,
        "unverified_claims": UNVERIFIED,
    }


def write_json(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def to_markdown(report: dict) -> str:
    problem = report["problem"]
    lines = [
        "# amplitude-lab validation report",
        "",
        f"**Overall status: {report['overall_status'].upper()}**",
        "",
        f"- Problem: `{problem['name']}`  ({problem['process']})",
        f"- Couplings: c4 = {problem['couplings']['c4']}, c42 = {problem['couplings']['c42']}",
        f"- mu2 = {problem['mu2']}, s0 = {problem['s0']}, scheme = {problem['scheme']}",
        f"- amplitude-lab {report['environment']['amplitude_lab']}, "
        f"Python {report['environment']['python']}",
        "",
        "## Deterministic checks",
        "",
        "| Check | Status | Evidence | Message |",
        "|---|---|---|---|",
    ]
    for c in report["checks"]:
        lines.append(f"| `{c['name']}` | {c['status']} | {c['evidence']} | {c['message']} |")
    lines += [
        "",
        "## Amplitudes at benchmark points",
        "",
        "| s | t | u | total |",
        "|---|---|---|---|",
    ]
    for a in report["amplitudes"]:
        k, r = a["kinematics"], a["result"]["total"]
        lines.append(f"| {k['s']} | {k['t']} | {k['u']} | {r['re']:.8f} {r['im']:+.8f}i |")
    lines += ["", "## Assistant", ""]
    if not report["assistant"]:
        lines.append("Not run.")
    for item in report["assistant"]:
        lines.append(
            f"- `{item['task']}` via `{item['client']}`: **{item['status']}**"
            + (f" (`{item['expression']}`)" if item.get("expression") else "")
        )
    lines += [
        "",
        "Assistant output is unverified commentary unless a deterministic check passed.",
        "",
    ]
    lines += ["## Limitations", ""] + [f"- {item}" for item in report["limitations"]]
    lines += ["", "## Explicitly unverified", ""] + [f"- {i}" for i in report["unverified_claims"]]
    lines.append("")
    return "\n".join(lines)


def write_markdown(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(report), encoding="utf-8")
