"""Offline scripted assistants for tests and demos (no network, no key)."""

from __future__ import annotations

MOCK_PROPOSALS: dict[str, str] = {
    "correct": "2 - log(-s/mu2)",
    "wrong_sign": "2 + log(-s/mu2)",
    "wrong_branch": "2 - log(s/mu2)",
    "explicit_ipi": "2 - log(-s/mu2) + I*pi",
    "malicious": "__import__('os').system('echo unsafe')",
    "garbage": "I think the answer is probably two minus a logarithm.",
}


class ScriptedClient:
    """Returns a fixed proposal regardless of the prompt."""

    def __init__(self, proposal: str, name: str = "mock"):
        self.proposal = proposal
        self.name = name
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.calls.append((prompt, system_instruction))
        return self.proposal


def mock_client(kind: str) -> ScriptedClient:
    if kind not in MOCK_PROPOSALS:
        raise ValueError(f"Unknown mock kind {kind!r}; choose from {sorted(MOCK_PROPOSALS)}.")
    return ScriptedClient(MOCK_PROPOSALS[kind], name=f"mock:{kind}")
