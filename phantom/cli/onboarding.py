"""First-run API key setup."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from .config import PhantomConfig

_ENV_HINTS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def run_onboarding(
    console: Console,
    config: PhantomConfig,
    provider_name: str,
) -> str | None:
    """Prompt for an API key. Returns the key or None if skipped."""
    console.print()
    console.print(
        Panel(
            "[bold]Welcome to Phantom![/bold]\n\n"
            "An API key is needed to connect to an LLM provider.\n"
            f"Provider: [cyan]{provider_name}[/cyan]  ·  "
            f"Model: [cyan]{config.model}[/cyan]",
            title="Setup",
            border_style="blue",
        )
    )

    hint = _ENV_HINTS.get(provider_name, "")
    if hint:
        console.print(
            f"\n[dim]You can also set the [bold]{hint}[/bold] "
            f"environment variable instead.[/dim]\n"
        )

    try:
        key = console.input(
            f"Enter your {provider_name} API key (Enter to skip): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not key:
        return None

    config.keys[provider_name] = key
    config.save()
    console.print("[green]API key saved to ~/.phantom/config.toml[/green]")
    return key
