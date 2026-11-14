"""Exception types used to classify why a chain was dropped.

Every one of these is a ``ValueError`` so that existing callers keep
working, but they carry a ``reason`` slug so that the notebook can tabulate
failures by cause instead of printing free text.
"""

from __future__ import annotations

__all__ = [
    "ChainError",
    "InsufficientQuotesError",
    "ChainQualityError",
    "SolverError",
    "failure_reason",
]


class ChainError(ValueError):
    """Base class for a chain that cannot be used."""

    reason = "chain_error"

    def __init__(self, message: str, reason: str | None = None):
        super().__init__(message)
        if reason is not None:
            self.reason = reason


class InsufficientQuotesError(ChainError):
    """Too few usable quotes remain after filtering."""

    reason = "insufficient_quotes"


class ChainQualityError(ChainError):
    """The chain parses but fails a data-quality threshold."""

    reason = "failed_quality_check"


class SolverError(ChainError):
    """The convex program did not return a usable solution."""

    reason = "solver_failure"


def failure_reason(exc: BaseException) -> str:
    """Return a short slug describing why a chain failed.

    Falls back to the exception class name so that unexpected errors are
    still grouped rather than lost.
    """
    return getattr(exc, "reason", type(exc).__name__)
