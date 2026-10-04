"""Exception hierarchy with stable `kind` values.

`CorruptionError` is separate from `ParseError` on purpose: a torn WAL tail is expected after a
crash and is repaired, while a corrupt *sealed* table is not recoverable and must stop the caller.
"""

from __future__ import annotations


class KVError(Exception):
    """Base class for every error this package raises on purpose."""

    kind = "kv_error"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = {key: value for key, value in context.items() if value is not None}

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"error": self.kind, "message": self.message}
        document.update(self.context)
        return document


class ParseError(KVError):
    """Malformed input handed to the CLI or to a record decoder."""

    kind = "parse_error"

    def __init__(self, message: str, *, line: int | None = None, **context: object) -> None:
        super().__init__(message, line=line, **context)


class ValidationError(KVError):
    """A request that is well-formed but not allowed: bad key, bad range, bad limit."""

    kind = "validation_error"


class CorruptionError(KVError):
    """A sealed artifact failed its checksum; recovery is not safe to continue."""

    kind = "corruption_error"


class OutputError(KVError):
    """A path or write that cannot be completed without risking existing data."""

    kind = "output_error"
