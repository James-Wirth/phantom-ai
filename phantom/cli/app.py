"""The main CLI application."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console

import phantom
from phantom._providers import _infer_provider

from .commands import CommandContext, CommandRegistry
from .completions import SlashCompleter
from .config import CONFIG_DIR, PhantomConfig
from .display import DisplayManager
from .onboarding import run_onboarding


class PhantomApp:
    """Interactive REPL for data analysis with LLMs."""

    def __init__(
        self,
        data_dir: str | None = None,
        model: str | None = None,
    ) -> None:
        self.console = Console()
        self.config = PhantomConfig.load()
        self.display = DisplayManager(self.console, self.config)
        self.commands = CommandRegistry()

        # CLI args override config
        if data_dir:
            self.config.data_dir = data_dir
        if model:
            self.config.model = model

        self.session: phantom.Session | None = None
        self.chat: phantom.Chat | None = None
        self.total_usage = phantom.Usage()

    def run(self) -> None:
        provider_name = self._resolve_provider_name()
        api_key = self.config.get_api_key(provider_name)

        if not api_key:
            api_key = run_onboarding(
                self.console, self.config, provider_name
            )
            if not api_key:
                self.console.print(
                    "[red]No API key configured. Exiting.[/red]"
                )
                sys.exit(1)

        self._init_session_and_chat(provider_name, api_key)
        data_dir = self.config.data_dir or "."
        self.display.show_banner(
            self.config.model, provider_name, data_dir
        )
        self._repl()

    def _resolve_provider_name(self) -> str:
        if self.config.provider:
            return self.config.provider
        return _infer_provider(self.config.model) or "anthropic"

    def _init_session_and_chat(
        self, provider_name: str, api_key: str
    ) -> None:
        kwargs: dict[str, Any] = {"secure": self.config.secure}
        if self.config.data_dir:
            kwargs["data_dir"] = self.config.data_dir

        self.session = phantom.Session(**kwargs)
        self._wrap_handle_tool_call()

        self.chat = phantom.Chat(
            self.session,
            provider=provider_name,
            api_key=api_key,
            model=self.config.model,
            system=self._build_system_context(),
        )


    def _build_system_context(self) -> str:
        """Build the developer system prompt with data directory info."""
        parts = [
            "You are a data analyst. Help the user explore and "
            "analyze their data using the available tools."
        ]

        if self.config.data_dir:
            resolved = str(Path(self.config.data_dir).expanduser().resolve())
            parts.append(f"\nData directory: {resolved}")

            data_files = self._list_data_files(resolved)
            if data_files:
                parts.append(
                    "Available files (use these full paths with read_csv, "
                    "read_parquet, or read_json):"
                )
                for f in data_files:
                    parts.append(f"  - {f}")

        return "\n".join(parts)

    @staticmethod
    def _list_data_files(directory: str) -> list[str]:
        """List data files in a directory (non-recursive)."""
        exts = {".csv", ".parquet", ".json", ".jsonl", ".tsv"}
        try:
            return sorted(
                str(Path(directory) / entry.name)
                for entry in os.scandir(directory)
                if entry.is_file() and Path(entry.name).suffix in exts
            )
        except OSError:
            return []

    _SCHEMA_OPS = {"read_csv", "read_parquet", "read_json", "query"}

    def _wrap_handle_tool_call(self) -> None:
        """Wrap session.handle_tool_call for tool-call visibility."""
        assert self.session is not None
        original = self.session.handle_tool_call

        def wrapped(
            name: str, arguments: Any, **kw: Any
        ) -> phantom.ToolResult:
            if name != "peek":
                self.display.show_tool_call(name, arguments)
            result = original(name, arguments, **kw)

            schema = None
            if (
                result.kind == "ref"
                and result.ref is not None
                and name in self._SCHEMA_OPS
            ):
                try:
                    peek_data = self.session.peek(result.ref.id)  # type: ignore[union-attr]
                    schema = {
                        "columns": peek_data.get("columns", {}),
                        "row_count": peek_data.get("row_count"),
                    }
                except Exception:
                    pass

            if name != "peek":
                self.display.show_tool_result(result, schema=schema)
            return result

        self.session.handle_tool_call = wrapped  # type: ignore[assignment]

    def recreate_chat(self) -> None:
        """Recreate Session + Chat after config changes."""
        provider_name = self._resolve_provider_name()
        api_key = self.config.get_api_key(provider_name)
        if not api_key:
            self.display.show_error(
                f"No API key for {provider_name}. Use /key to set one."
            )
            return
        self._init_session_and_chat(provider_name, api_key)
        self.console.print(
            f"[dim]{provider_name}/{self.config.model}[/dim]"
        )

    def _repl(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        history_path = CONFIG_DIR / "history"

        prompt_session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            completer=SlashCompleter(self.commands),
            multiline=False,
        )

        while True:
            try:
                user_input = prompt_session.prompt(
                    HTML("<ansiblue>&gt; </ansiblue>")
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self.display.show_goodbye()
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                ctx = CommandContext(app=self)
                self.commands.execute(user_input, ctx)
            else:
                self._ask(user_input)

    def _ask(self, message: str) -> None:
        if self.chat is None:
            self.display.show_error("Chat not initialized.")
            return

        try:
            with self.display.thinking_spinner():
                response = self.chat.ask(message)
        except Exception as exc:
            self.display.show_error(f"Error: {exc}")
            return

        self.total_usage = phantom.Usage(
            input_tokens=(
                self.total_usage.input_tokens
                + response.usage.input_tokens
            ),
            output_tokens=(
                self.total_usage.output_tokens
                + response.usage.output_tokens
            ),
        )

        self.display.show_response(response)
