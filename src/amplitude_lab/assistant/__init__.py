"""Optional LLM assistance. Never a source of mathematical truth.

Every proposal is (1) parsed through a strict AST whitelist, never executed as Python,
and (2) verified against deterministic reference code before being reported.
"""

from .errors import AssistantConfigError, AssistantRequestError, AssistantUnavailableError
from .orchestrator import AssistantOutcome, run_task
from .tasks import BubbleClosedFormTask

__all__ = [
    "AssistantConfigError",
    "AssistantOutcome",
    "AssistantRequestError",
    "AssistantUnavailableError",
    "BubbleClosedFormTask",
    "run_task",
]
