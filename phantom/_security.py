"""Security policy infrastructure for Phantom operations.

Provides composable guards that validate operation arguments before execution,
preventing path traversal, ReDoS, and resource exhaustion attacks.

Usage:
    from phantom import Session, SecurityPolicy, PathGuard

    policy = SecurityPolicy()
    policy.bind(PathGuard(allowed_dirs=["/data"]), ops=["read_text"], args=["path"])

    session = Session(policy=policy)
"""

from __future__ import annotations

import fnmatch
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ._paths import resolve_path

DEFAULT_DENY_PATTERNS: list[str] = [
    "*.env",
    ".git",
    ".ssh",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    ".aws",
    "credentials*",
    ".netrc",
    ".npmrc",
    "*.secret",
    "*.secrets",
]


class SecurityError(ValueError):
    """Raised when a security guard blocks an operation.

    Attributes:
        op_name: The operation that was blocked.
        arg_name: The argument that failed validation.
        guard_name: The guard class that triggered the block.
    """

    def __init__(
        self,
        message: str,
        *,
        op_name: str,
        arg_name: str,
        guard_name: str,
    ) -> None:
        self.op_name = op_name
        self.arg_name = arg_name
        self.guard_name = guard_name
        super().__init__(
            f"[{guard_name}] Blocked {op_name}(…{arg_name}=…): {message}"
        )


# =============================================================================
# Guard base class
# =============================================================================


class Guard(ABC):
    """Base class for security guards.

    Subclasses implement ``check()`` to validate a single argument value.
    Raise ``SecurityError`` to block the operation.
    """

    @abstractmethod
    def check(self, value: Any, *, op_name: str, arg_name: str) -> None:
        """Validate *value*. Raise ``SecurityError`` if not allowed."""


# =============================================================================
# Built-in guards
# =============================================================================


class PathGuard(Guard):
    """Restrict file paths to allowed directories and/or specific files.

    Resolves paths to absolute form (following symlinks) and verifies they
    fall within at least one of the *allowed_dirs* or match one of the
    *allowed_paths* exactly.  Optionally rejects paths matching
    *deny_patterns* (fnmatch globs checked against all path components,
    not just the filename).

    Args:
        allowed_dirs: Directories the operation may access.
        allowed_paths: Specific file paths the operation may access.
        deny_patterns: Glob patterns checked against every path component
            (e.g. ``["*.env", ".git"]`` blocks both ``prod.env`` and
            ``.git/config``).

    When neither *allowed_dirs* nor *allowed_paths* is provided, no
    location restriction is applied (deny patterns and other guards
    still run).
    """

    def __init__(
        self,
        allowed_dirs: list[str | Path] | None = None,
        deny_patterns: list[str] | None = None,
        *,
        allowed_paths: list[str | Path] | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self._allowed_dirs = (
            [resolve_path(d, relative_to=base_dir) for d in allowed_dirs]
            if allowed_dirs else []
        )
        self._allowed_paths = (
            {resolve_path(p, relative_to=base_dir) for p in allowed_paths}
            if allowed_paths else set()
        )
        self._deny = deny_patterns or []
        self._has_restrictions = bool(self._allowed_dirs or self._allowed_paths)

    def check(self, value: Any, *, op_name: str, arg_name: str) -> None:
        try:
            resolved = resolve_path(value)
        except (TypeError, OSError) as exc:
            raise SecurityError(
                f"Invalid path: {exc}",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="PathGuard",
            ) from exc

        for pattern in self._deny:
            for component in resolved.parts:
                if fnmatch.fnmatch(component, pattern):
                    raise SecurityError(
                        f"Path matches deny pattern '{pattern}': {value}",
                        op_name=op_name,
                        arg_name=arg_name,
                        guard_name="PathGuard",
                    )

        if not self._has_restrictions:
            return

        if resolved in self._allowed_paths:
            return

        for allowed in self._allowed_dirs:
            try:
                resolved.relative_to(allowed)
                return
            except ValueError:
                continue

        locations = ", ".join(
            [str(d) for d in self._allowed_dirs]
            + [str(p) for p in sorted(self._allowed_paths)]
        )
        raise SecurityError(
            f"Path '{value}' is outside allowed locations: [{locations}]",
            op_name=op_name,
            arg_name=arg_name,
            guard_name="PathGuard",
        )


class FileSizeGuard(Guard):
    """Prevent reading oversized files.

    Checks the file size (via ``os.path.getsize``) before the read operation
    executes.

    Args:
        max_bytes: Maximum file size in bytes (default: 50 MB).
    """

    def __init__(self, max_bytes: int = 50_000_000) -> None:
        self._max_bytes = max_bytes

    def check(self, value: Any, *, op_name: str, arg_name: str) -> None:
        path = str(value)
        try:
            size = os.path.getsize(path)
        except OSError:
            return

        if size > self._max_bytes:
            mb = self._max_bytes / (1024 * 1024)
            actual_mb = size / (1024 * 1024)
            raise SecurityError(
                f"File size ({actual_mb:.1f} MB) exceeds limit ({mb:.1f} MB)",
                op_name=op_name,
                arg_name=arg_name,
                guard_name="FileSizeGuard",
            )


# =============================================================================
# SecurityPolicy
# =============================================================================


class SecurityPolicy:
    """Composable security policy that binds guards to operations.

    Guards are checked at resolution time, before the operation executes.
    If any guard raises ``SecurityError``, the operation is blocked.

    Example:
        policy = SecurityPolicy()
        policy.bind(PathGuard(["/data"]), ops=["read_text"], args=["path"])
        policy.bind(RegexGuard(), ops=["search_text"], args=["pattern"])

        session = Session(policy=policy)
    """

    def __init__(self) -> None:
        self._bindings: list[tuple[Guard, frozenset[str], frozenset[str]]] = []

    def bind(
        self,
        guard: Guard,
        *,
        ops: list[str],
        args: list[str],
    ) -> SecurityPolicy:
        """Bind a guard to specific operations and argument names.

        Args:
            guard: The guard instance to apply.
            ops: Operation names this guard applies to.
            args: Argument names this guard validates.

        Returns:
            self (for method chaining).
        """
        self._bindings.append((guard, frozenset(ops), frozenset(args)))
        return self

    def check(self, op_name: str, arguments: dict[str, Any]) -> None:
        """Run all applicable guards against the given arguments.

        Raises:
            SecurityError: If any guard blocks the operation.
        """
        for guard, ops, arg_names in self._bindings:
            if op_name not in ops:
                continue
            for arg_name in arg_names:
                if arg_name in arguments:
                    guard.check(
                        arguments[arg_name],
                        op_name=op_name,
                        arg_name=arg_name,
                    )

    def __or__(self, other: SecurityPolicy) -> SecurityPolicy:
        """Merge two policies: ``combined = policy_a | policy_b``."""
        merged = SecurityPolicy()
        merged._bindings = list(self._bindings) + list(other._bindings)
        return merged

    def __repr__(self) -> str:
        guards = [type(g).__name__ for g, _, _ in self._bindings]
        return f"SecurityPolicy(guards={guards})"


__all__ = [
    "DEFAULT_DENY_PATTERNS",
    "SecurityError",
    "Guard",
    "PathGuard",
    "FileSizeGuard",
    "SecurityPolicy",
]
