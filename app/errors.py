from __future__ import annotations


class ProviderError(RuntimeError):
    """Raised when an upstream AI/media provider request fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        generation_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.generation_id = generation_id


class ProviderOutputError(ProviderError):
    """Raised when an upstream provider succeeds but returns unusable output."""
