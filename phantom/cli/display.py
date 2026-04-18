"""Rich terminal rendering for the Phantom CLI."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from phantom import ChatResponse, ToolResult

    from .config import PhantomConfig

_SQL_KEYS = {"sql", "query"}


class DisplayManager:
    """Handles all Rich terminal output."""

    def __init__(self, console: Console, config: PhantomConfig) -> None:
        self.console = console
        self.config = config
        self._spinner: Status | None = None

    def show_banner(
        self, model: str, provider: str, data_dir: str
    ) -> None:
        import phantom

        self.console.print()
        self.console.print(
            f"[bold]phantom[/bold] [dim]v{phantom.__version__}[/dim]"
        )
        self.console.print(
            f"[dim]{provider}/{model} · {data_dir}[/dim]"
        )
        self.console.print()

    def show_goodbye(self) -> None:
        self.console.print()

    @contextmanager
    def thinking_spinner(self) -> Generator[None, None, None]:
        self.console.print()
        self._spinner = self.console.status(
            "[dim]Thinking...[/dim]", spinner="dots"
        )
        self._spinner.start()
        try:
            yield
        finally:
            self._spinner.stop()
            self._spinner = None

    def _pause_spinner(self) -> None:
        if self._spinner is not None:
            self._spinner.stop()

    def _resume_spinner(self) -> None:
        if self._spinner is not None:
            self._spinner.start()

    @staticmethod
    def _format_path(value: str) -> str:
        """Shorten a file path to basename for display."""
        if os.sep in value or value.startswith("~"):
            return os.path.basename(value)
        return value

    def show_tool_call(self, name: str, arguments: Any) -> None:
        if not self.config.show_tool_calls:
            return
        self._pause_spinner()

        self.console.print()
        self.console.print(
            Text.assemble(("  ▸ ", "dim"), (name, "bold cyan"))
        )

        if isinstance(arguments, dict):
            sql_value: str | None = None
            other_args: dict[str, Any] = {}

            for k, v in arguments.items():
                if k in _SQL_KEYS and isinstance(v, str):
                    sql_value = v
                else:
                    other_args[k] = v

            for k, v in other_args.items():
                display_v = v
                if isinstance(v, str) and len(v) > 80:
                    display_v = v[:77] + "..."
                if isinstance(v, str) and (os.sep in v or v.startswith("~")):
                    display_v = self._format_path(v)
                self.console.print(
                    Text.assemble(
                        ("    ", ""),
                        (f"{k}: ", "dim"),
                        (str(display_v), ""),
                    )
                )

            if sql_value is not None:
                self.console.print(
                    Syntax(
                        sql_value.strip(),
                        "sql",
                        theme="monokai",
                        padding=(0, 4),
                        word_wrap=True,
                    )
                )
        elif arguments is not None:
            self.console.print(f"    [dim]{arguments}[/dim]")

        self._resume_spinner()

    def show_tool_result(
        self,
        result: ToolResult,
        schema: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.show_tool_calls:
            return
        self._pause_spinner()

        if result.kind == "ref":
            ref_id = result.ref.id if result.ref else "?"
            op = result.data.get("op", "")
            label = f"{ref_id}" + (f" ({op})" if op else "")
            self.console.print(
                Text.assemble(
                    ("    → ", "dim"),
                    (label, "cyan"),
                )
            )
            if schema:
                self._show_schema_table(schema)
        elif result.kind == "peek":
            cols = list(result.data.get("columns", {}).keys())
            rows = result.data.get("row_count")
            parts: list[str] = []
            if rows is not None:
                parts.append(f"{rows:,} rows")
            if cols:
                parts.append(f"{len(cols)} cols")
                parts.append(", ".join(cols[:8]))
                if len(cols) > 8:
                    parts.append(f"… +{len(cols) - 8} more")
            summary = " · ".join(parts)
            self.console.print(
                Text.assemble(
                    ("    → ", "dim"),
                    (summary, "green"),
                )
            )
        elif result.kind == "error":
            msg = result.data.get("message", "unknown error")
            self.console.print(
                Text.assemble(
                    ("    → ", "red"),
                    (f"error: {msg}", "red"),
                )
            )

        self.console.print()
        self._resume_spinner()

    def _show_schema_table(self, schema: dict[str, Any]) -> None:
        """Render a transposed schema table (columns shown horizontally)."""
        columns: dict[str, str] = schema.get("columns", {})
        if not columns:
            return

        indent = 4
        available_width = self.console.width - indent
        col_items = list(columns.items())

        col_widths: list[int] = []
        for name, dtype in col_items:
            col_widths.append(max(len(name), len(dtype)) + 3)

        total = 1
        fit_count = 0
        for w in col_widths:
            if total + w > available_width:
                break
            total += w
            fit_count += 1

        fit_count = max(1, fit_count)
        truncated = fit_count < len(col_items)
        display_items = col_items[:fit_count]

        if truncated and fit_count > 1:
            overflow_label = f"+{len(col_items) - fit_count}"
            overflow_width = len(overflow_label) + 3
            while fit_count > 1 and total + overflow_width > available_width:
                fit_count -= 1
                total -= col_widths[fit_count]
            overflow_label = f"+{len(col_items) - fit_count}"
            display_items = col_items[:fit_count]

        table = Table(
            show_header=False,
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
            pad_edge=False,
            width=available_width,
        )

        for _ in display_items:
            table.add_column()
        if truncated:
            table.add_column()

        names: list[Text] = [Text(n, style="cyan") for n, _ in display_items]
        if truncated:
            names.append(Text(f"… {overflow_label}", style="dim"))
        table.add_row(*names)

        types: list[Text] = [Text(t, style="dim") for _, t in display_items]
        if truncated:
            types.append(Text(""))
        table.add_row(*types)

        row_count = schema.get("row_count")
        footer = f"{len(columns)} columns"
        if row_count is not None:
            footer = f"{row_count:,} rows · {footer}"

        self.console.print()
        self.console.print(Padding(table, (0, 0, 0, indent)))
        self.console.print(f"    [dim]{footer}[/dim]")

    def show_response(self, response: ChatResponse) -> None:
        self.console.print()
        self.console.print(Markdown(response.text))

        if self.config.show_usage:
            u = response.usage
            tc = response.tool_calls_made
            meta = (
                f"[dim]{response.turns} turn{'s' if response.turns != 1 else ''}"
                f" · {tc} tool call{'s' if tc != 1 else ''}"
                f" · {u.total_tokens:,} tokens[/dim]"
            )
            self.console.print()
            self.console.print(meta)

        self.console.print()

    def show_error(self, message: str) -> None:
        self.console.print(Text(message, style="red"))
