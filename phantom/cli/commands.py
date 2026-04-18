"""Slash command registry and built-in commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import PhantomApp


@dataclass
class CommandContext:
    app: PhantomApp


@dataclass
class Command:
    name: str
    handler: Callable[[CommandContext, str], None]
    help_text: str
    usage: str = ""


class CommandRegistry:
    """Registry of slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}
        self._register_builtins()

    def register(
        self,
        name: str,
        help_text: str,
        usage: str = "",
    ) -> Callable[
        [Callable[[CommandContext, str], None]],
        Callable[[CommandContext, str], None],
    ]:
        def decorator(
            fn: Callable[[CommandContext, str], None],
        ) -> Callable[[CommandContext, str], None]:
            self._commands[name] = Command(
                name=name, handler=fn, help_text=help_text, usage=usage
            )
            return fn

        return decorator

    def execute(self, raw_input: str, ctx: CommandContext) -> None:
        parts = raw_input.lstrip("/").split(None, 1)
        name = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        cmd = self._commands.get(name)
        if cmd is None:
            ctx.app.console.print(f"[red]Unknown command: /{name}[/red]")
            ctx.app.console.print("[dim]Type /help for available commands.[/dim]")
            return

        cmd.handler(ctx, args)

    @property
    def all_commands(self) -> dict[str, Command]:
        return dict(self._commands)

    def _register_builtins(self) -> None:
        @self.register("help", "Show available commands")
        def cmd_help(ctx: CommandContext, args: str) -> None:
            from rich.table import Table

            table = Table(show_header=True, border_style="dim")
            table.add_column("Command", style="cyan")
            table.add_column("Description")
            table.add_column("Usage", style="dim")
            for cmd in ctx.app.commands.all_commands.values():
                if cmd.name == "exit":
                    continue
                table.add_row(f"/{cmd.name}", cmd.help_text, cmd.usage or "")
            ctx.app.console.print(table)

        @self.register("model", "Show or switch model", usage="/model <name>")
        def cmd_model(ctx: CommandContext, args: str) -> None:
            if not args:
                ctx.app.console.print(
                    f"Current model: [bold]{ctx.app.config.model}[/bold]"
                )
                return
            ctx.app.config.model = args.strip()
            ctx.app.config.provider = None
            ctx.app.recreate_chat()

        @self.register(
            "provider", "Show or switch provider", usage="/provider <name>"
        )
        def cmd_provider(ctx: CommandContext, args: str) -> None:
            if not args:
                name = ctx.app._resolve_provider_name()
                ctx.app.console.print(
                    f"Current provider: [bold]{name}[/bold]"
                )
                return
            ctx.app.config.provider = args.strip()
            ctx.app.recreate_chat()

        @self.register("key", "Set API key for current provider", usage="/key <key>")
        def cmd_key(ctx: CommandContext, args: str) -> None:
            provider_name = ctx.app._resolve_provider_name()
            if not args:
                has_key = bool(ctx.app.config.get_api_key(provider_name))
                status = "[green]set[/green]" if has_key else "[red]not set[/red]"
                ctx.app.console.print(f"API key for {provider_name}: {status}")
                return
            ctx.app.config.keys[provider_name] = args.strip()
            ctx.app.config.save()
            ctx.app.recreate_chat()
            ctx.app.console.print(
                f"[green]API key saved for {provider_name}.[/green]"
            )

        @self.register("data", "Set data directory", usage="/data <path>")
        def cmd_data(ctx: CommandContext, args: str) -> None:
            if not args:
                ctx.app.console.print(
                    f"Data dir: [bold]{ctx.app.config.data_dir or '(none)'}[/bold]"
                )
                return
            ctx.app.config.data_dir = args.strip()
            ctx.app.recreate_chat()
            ctx.app.console.print(
                f"[green]Data directory: {args.strip()}[/green]"
            )

        @self.register("clear", "Clear conversation history")
        def cmd_clear(ctx: CommandContext, args: str) -> None:
            if ctx.app.chat:
                ctx.app.chat.reset()
            ctx.app.console.print("[dim]Conversation cleared.[/dim]")

        @self.register("refs", "List current refs in session")
        def cmd_refs(ctx: CommandContext, args: str) -> None:
            if not ctx.app.session:
                ctx.app.console.print("[dim]No session.[/dim]")
                return
            refs = ctx.app.session.list_refs()
            if not refs:
                ctx.app.console.print("[dim]No refs yet.[/dim]")
                return
            from rich.table import Table

            table = Table(border_style="dim")
            table.add_column("ID", style="cyan")
            table.add_column("Operation", style="green")
            table.add_column("Parents", style="dim")
            for ref in refs:
                parents = ", ".join(p.id for p in ref.parents) or "-"
                table.add_row(ref.id, ref.op, parents)
            ctx.app.console.print(table)

        @self.register("cost", "Show token usage")
        def cmd_cost(ctx: CommandContext, args: str) -> None:
            u = ctx.app.total_usage
            ctx.app.console.print(
                f"Tokens: [cyan]{u.input_tokens:,}[/cyan] in · "
                f"[cyan]{u.output_tokens:,}[/cyan] out · "
                f"[bold]{u.total_tokens:,}[/bold] total"
            )

        @self.register("save", "Save config to ~/.phantom/config.toml")
        def cmd_save(ctx: CommandContext, args: str) -> None:
            ctx.app.config.save()
            ctx.app.console.print(
                "[green]Config saved to ~/.phantom/config.toml[/green]"
            )

        @self.register("quit", "Exit phantom")
        def cmd_quit(ctx: CommandContext, args: str) -> None:
            ctx.app.display.show_goodbye()
            raise SystemExit(0)

        @self.register("exit", "Exit phantom")
        def cmd_exit(ctx: CommandContext, args: str) -> None:
            cmd_quit(ctx, args)
