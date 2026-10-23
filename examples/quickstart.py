"""Minimal tour of the API. Run: python examples/quickstart.py"""

from amplitude_lab import benchmarks
from amplitude_lab.amplitude import assemble
from amplitude_lab.assistant import BubbleClosedFormTask, run_task
from amplitude_lab.assistant.mock import mock_client
from amplitude_lab.kinematics import Kinematics
from amplitude_lab.validation import run_all

problem = benchmarks.yhg_dim4()

result = assemble(problem, Kinematics.from_s_t(5.0, -2.0))
print("tree      :", result.tree)
print("one-loop  :", result.one_loop)
print(
    "channels  :",
    {k: complex(round(v.real, 6), round(v.imag, 6)) for k, v in result.channels.items()},
)

print("\nDeterministic checks")
for check in run_all(problem):
    print(f"  [{check.status.upper():5s}] {check.name:34s} ({check.evidence})")

print("\nAssistant proposals (offline mocks, no key needed)")
for kind in ("correct", "wrong_sign", "malicious"):
    outcome = run_task(mock_client(kind), BubbleClosedFormTask(), problem)
    print(f"  {kind:10s} -> {outcome.status}")
