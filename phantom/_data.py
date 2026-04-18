"""Built-in data engine backed by DuckDB.

Provides five core operations (read_csv, read_parquet, read_json, query, export)
registered automatically when a Session is created.  DuckDB connections are
created lazily and hardened with sandboxed configuration.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from ._paths import resolve_path
from ._security import (
    DEFAULT_DENY_PATTERNS,
    FileSizeGuard,
    PathGuard,
    SecurityPolicy,
)

if TYPE_CHECKING:
    from ._session import Session

DuckDBPyRelation = duckdb.DuckDBPyRelation
DuckDBPyConnection = duckdb.DuckDBPyConnection

_IO_OPS = ["read_csv", "read_parquet", "read_json"]

_BASE_CONFIG: dict[str, str] = {
    "autoinstall_known_extensions": "false",
    "autoload_known_extensions": "false",
}


_COMMENT_RE = re.compile(
    r"--[^\n]*"        # line comments
    r"|/\*.*?\*/",     # block comments
    re.DOTALL,
)

_ALLOWED_STARTS = frozenset({"select", "with"})


def _validate_sql(sql: str) -> None:
    """Validate that *sql* is a read-only query (SELECT or WITH/CTE).

    Strips comments before checking.  Rejects administrative statements
    (COPY, INSTALL, LOAD, ATTACH, CREATE, DROP, INSERT, UPDATE, DELETE,
    SET, PRAGMA, CALL, etc.) which have no place in a sandboxed analytics
    engine.

    Raises:
        ValueError: If the statement is not a SELECT or WITH query.
    """
    stripped = _COMMENT_RE.sub(" ", sql).strip()
    if not stripped:
        raise ValueError("Empty SQL query")
    first_word = stripped.split(None, 1)[0].lower().rstrip("(")
    if first_word not in _ALLOWED_STARTS:
        raise ValueError(
            f"Only SELECT and WITH (CTE) queries are allowed, "
            f"got: {first_word.upper()}"
        )


def _validate_identifier(name: str) -> None:
    """Validate a SQL identifier for use as a view name.

    Allows only alphanumeric characters and underscores.  This is
    deliberately restrictive — DuckDB's ``register()`` handles quoting
    internally, but we reject unusual names to prevent surprises.
    """
    if not name:
        raise ValueError("Identifier must not be empty")
    if not name.replace("_", "").isalnum():
        raise ValueError(
            f"Invalid identifier {name!r}: "
            "only alphanumeric characters and underscores are allowed"
        )


def _secure_connect(
    *,
    allowed_dirs: list[str | Path] | None = None,
    allowed_paths: list[str | Path] | None = None,
    base_dir: Path | None = None,
) -> DuckDBPyConnection:
    """Create a hardened in-memory DuckDB connection.

    When *allowed_dirs* or *allowed_paths* are provided, external access
    stays enabled but is restricted to those locations via DuckDB's native
    allowlists.  When neither is provided, external access is disabled
    entirely.  Configuration is always locked at the end so injected SQL
    cannot reverse these settings.
    """
    needs_file_access = bool(allowed_dirs or allowed_paths)

    config = dict(_BASE_CONFIG)
    if not needs_file_access:
        config["enable_external_access"] = "false"

    conn = duckdb.connect(database=":memory:", config=config)

    if allowed_dirs:
        dirs = [str(resolve_path(d, relative_to=base_dir)) for d in allowed_dirs]
        conn.execute("SET allowed_directories = $dirs", {"dirs": dirs})
    if allowed_paths:
        paths = [str(resolve_path(p, relative_to=base_dir)) for p in allowed_paths]
        conn.execute("SET allowed_paths = $paths", {"paths": paths})

    conn.execute("SET lock_configuration = true")
    return conn


def _data_policy(
    allowed_dirs: list[str | Path] | None = None,
    *,
    allowed_paths: list[str | Path] | None = None,
    deny_patterns: list[str] | None = None,
    max_file_bytes: int = 50_000_000,
    base_dir: Path | None = None,
) -> SecurityPolicy:
    """Build the security policy for built-in data operations."""
    if deny_patterns is None:
        deny_patterns = list(DEFAULT_DENY_PATTERNS)

    path_guard = PathGuard(
        allowed_dirs, deny_patterns=deny_patterns, allowed_paths=allowed_paths,
        base_dir=base_dir,
    )
    policy = SecurityPolicy()
    policy.bind(path_guard, ops=_IO_OPS, args=["path"])
    policy.bind(
        FileSizeGuard(max_bytes=max_file_bytes),
        ops=_IO_OPS,
        args=["path"],
    )
    return policy

def _inspect_relation(relation: DuckDBPyRelation) -> dict[str, Any]:
    """Relation inspector for LLM context."""
    columns = relation.columns
    types = relation.types
    sample = relation.limit(5).fetchall()
    return {
        "type": "duckdb.Relation",
        "columns": {col: str(dtype) for col, dtype in zip(columns, types)},
        "sample": [dict(zip(columns, row)) for row in sample],
    }


class _DataEngine:
    """Internal data engine owned by Session.

    Manages a lazily-created, sandboxed DuckDB connection and registers
    five built-in operations plus an inspector and security policy.
    """

    def __init__(
        self,
        *,
        allowed_dirs: list[str | Path] | None = None,
        allowed_paths: list[str | Path] | None = None,
        max_file_bytes: int = 50_000_000,
        output_format: str = "relation",
        base_dir: Path | None = None,
    ) -> None:
        self._allowed_dirs = allowed_dirs
        self._allowed_paths = allowed_paths
        self._max_file_bytes = max_file_bytes
        self._output_format = output_format
        self._base_dir = base_dir
        self._conn: DuckDBPyConnection | None = None

    @property
    def conn(self) -> DuckDBPyConnection:
        """Return (or lazily create) the sandboxed DuckDB connection."""
        if self._conn is None:
            self._conn = _secure_connect(
                allowed_dirs=self._allowed_dirs,
                allowed_paths=self._allowed_paths,
                base_dir=self._base_dir,
            )
        return self._conn

    def _read_csv(self, path: str) -> DuckDBPyRelation:
        """Read a CSV file into a relation."""
        return self.conn.read_csv(path)

    def _read_parquet(self, path: str) -> DuckDBPyRelation:
        """Read a Parquet file into a relation.

        Supports glob patterns like ``data/*.parquet``.
        """
        return self.conn.read_parquet(path)

    def _read_json(self, path: str) -> DuckDBPyRelation:
        """Read a JSON file into a relation."""
        return self.conn.read_json(path)

    def _query(
        self, sql: str, refs: dict[str, Any] | None = None
    ) -> DuckDBPyRelation:
        """Execute a read-only SQL query, optionally binding ref data as virtual tables.

        Only ``SELECT`` and ``WITH`` (CTE) statements are permitted.

        Args:
            sql: SQL query string (must be a SELECT or WITH query).
            refs: Optional mapping of table names to data (DataFrames,
                  DuckDB relations, etc.).  Each entry is registered as a
                  temporary view before the query runs.
        """
        _validate_sql(sql)
        if refs:
            for view_name, data in refs.items():
                _validate_identifier(view_name)
                self.conn.register(view_name, data)
        return self.conn.query(sql)

    def _export(
        self, relation: Any, format: str = "default"
    ) -> Any:
        """Materialize a relation to a concrete format.

        Args:
            relation: A DuckDB relation to materialize.
            format: Output format — "default" uses the engine's configured
                    output_format; other options are "pandas", "polars",
                    "arrow", "tuples", "dicts".
        """
        fmt = self._output_format if format == "default" else format

        if fmt == "relation":
            return relation
        elif fmt == "pandas":
            try:
                return relation.df()
            except ImportError:
                raise ImportError(
                    "pandas is required for format='pandas'. "
                    "Install with: pip install pandas"
                ) from None
        elif fmt == "polars":
            try:
                return relation.pl()
            except ImportError:
                raise ImportError(
                    "polars is required for format='polars'. "
                    "Install with: pip install polars"
                ) from None
        elif fmt == "arrow":
            return relation.arrow().read_all()
        elif fmt == "tuples":
            return relation.fetchall()
        elif fmt == "dicts":
            columns = relation.columns
            return [dict(zip(columns, row)) for row in relation.fetchall()]
        else:
            raise ValueError(
                f"Unknown format: '{fmt}'. "
                f"Options: relation, pandas, polars, arrow, tuples, dicts"
            )

    def register_into(self, session: Session) -> None:
        """Register operations, inspector, and security policy into *session*."""
        session._operations["read_csv"] = self._read_csv
        session._operations["read_parquet"] = self._read_parquet
        session._operations["read_json"] = self._read_json
        session._operations["query"] = self._query
        session._operations["export"] = self._export
        session._inspectors[DuckDBPyRelation] = _inspect_relation

        policy = _data_policy(
            self._allowed_dirs,
            allowed_paths=self._allowed_paths,
            max_file_bytes=self._max_file_bytes,
            base_dir=self._base_dir,
        )
        if session._auto_merge:
            if session._policy is None:
                session._policy = policy
            else:
                session._policy = session._policy | policy

    def close(self) -> None:
        """Close the DuckDB connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
