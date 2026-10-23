class AssistantConfigError(RuntimeError):
    """The assistant cannot be configured (missing key, bad setting)."""


class AssistantUnavailableError(RuntimeError):
    """An optional dependency (google-genai) is not installed."""


class AssistantRequestError(RuntimeError):
    """The request failed. Messages are sanitized; credentials never appear."""
