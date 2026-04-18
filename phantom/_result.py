"""Result - Tool call result types for LLM integration."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ._ref import Ref

if TYPE_CHECKING:
    from ._errors import ResolutionError


def _json_default(obj: object) -> Any:
    """Handle non-standard types that appear in DuckDB results."""
    if isinstance(obj, datetime.date | datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class ToolResult:
    """
    Result from handling an LLM tool call.

    Encapsulates the response that should be sent back to the LLM,
    with metadata about what kind of operation was performed.

    Attributes:
        kind: Type of result - "ref" for lazy operations, "peek" for inspection
        data: The dict to serialize back to the LLM
        ref: The Ref object (only for kind="ref")
    """

    kind: str
    data: dict[str, Any]
    ref: Ref[Any] | None = None

    @property
    def is_error(self) -> bool:
        """True if this result represents an error."""
        return self.kind == "error"

    def to_dict(self) -> dict[str, Any]:
        """Return the data dict for LLM consumption."""
        return self.data

    def to_json(self) -> str:
        """Serialize to JSON string for LLM consumption."""
        return json.dumps(self.data, default=_json_default)

    @classmethod
    def from_ref(cls, ref: Ref[Any]) -> ToolResult:
        """Create result for a lazy operation that created a ref."""
        return cls(
            kind="ref",
            data={"ref": ref.id, "op": ref.op, "args": ref.serialized_args()},
            ref=ref,
        )

    @classmethod
    def from_peek(cls, peek_data: dict[str, Any]) -> ToolResult:
        """Create result for a peek inspection."""
        return cls(
            kind="peek",
            data=peek_data,
            ref=None,
        )

    @classmethod
    def from_error(cls, error: ResolutionError) -> ToolResult:
        """Create result for a resolution error."""
        return cls(
            kind="error",
            data={
                "error": True,
                "type": (
                    type(error.cause).__name__ if error.cause else "ResolutionError"
                ),
                "message": str(error.cause) if error.cause else str(error),
                "ref": error.ref.id,
                "op": error.ref.op,
                "chain": error.chain,
            },
            ref=error.ref,
        )

    @classmethod
    def from_exception(cls, error: Exception) -> ToolResult:
        """Create result for an arbitrary exception (no ref context)."""
        return cls(
            kind="error",
            data={
                "error": True,
                "type": type(error).__name__,
                "message": str(error),
            },
            ref=None,
        )
