"""Path resolution utilities for Phantom."""

from __future__ import annotations

import os
import re
from pathlib import Path

_UNEXPANDED_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def resolve_path(
    p: str | Path,
    *,
    relative_to: Path | None = None,
) -> Path:
    """Expand and resolve a user-supplied path.

    Expansion order (mirrors shell behaviour):

    1. **Environment variables** — ``$VAR`` and ``${VAR}`` via
       :func:`os.path.expandvars`.
    2. **Tilde** — ``~`` and ``~user`` via :meth:`Path.expanduser`.
    3. **Relative anchoring** — if the result is still relative and
       *relative_to* is provided, the path is resolved against that
       directory instead of the current working directory.
    4. **Canonicalization** — :meth:`Path.resolve` produces an absolute,
       symlink-resolved path.

    Args:
        p: A path string or :class:`~pathlib.Path`.
        relative_to: Optional anchor directory for relative paths.
            When ``None``, relative paths resolve against the current
            working directory (the default ``Path.resolve()`` behaviour).

    Returns:
        An absolute, fully resolved :class:`~pathlib.Path`.

    Raises:
        ValueError: If the path references an undefined environment
            variable (e.g. ``$UNDEFINED``).
    """
    raw = str(p)
    expanded = os.path.expandvars(raw)

    m = _UNEXPANDED_RE.search(expanded)
    if m and m.group(0) in raw:
        raise ValueError(
            f"Undefined environment variable ${m.group(1)} in path {raw!r}"
        )

    result = Path(expanded).expanduser()

    if not result.is_absolute() and relative_to is not None:
        result = relative_to / result

    return result.resolve()
