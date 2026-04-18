"""Slash command auto-completion for prompt_toolkit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from .commands import CommandRegistry


class SlashCompleter(Completer):
    """Auto-complete slash commands."""

    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    def get_completions(
        self, document: Document, complete_event: Any
    ) -> Any:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        partial = text.lstrip("/")
        for name, cmd in self._registry.all_commands.items():
            if name.startswith(partial):
                yield Completion(
                    f"/{name}",
                    start_position=-len(text),
                    display_meta=cmd.help_text,
                )
