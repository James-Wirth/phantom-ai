"""Configuration management for the Phantom CLI."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".phantom"
CONFIG_PATH = CONFIG_DIR / "config.toml"

_ENV_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


@dataclass
class PhantomConfig:
    """Persistent CLI configuration backed by ~/.phantom/config.toml."""

    model: str = "claude-sonnet-4-6"
    provider: str | None = None
    keys: dict[str, str] = field(default_factory=dict)
    show_tool_calls: bool = True
    show_usage: bool = True
    data_dir: str | None = None
    secure: bool = True

    @classmethod
    def load(cls) -> PhantomConfig:
        if not CONFIG_PATH.exists():
            return cls()
        with open(CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
        return cls(
            model=raw.get("model", {}).get("default", cls.model),
            provider=raw.get("model", {}).get("provider"),
            keys=dict(raw.get("keys", {})),
            show_tool_calls=raw.get("display", {}).get("show_tool_calls", True),
            show_usage=raw.get("display", {}).get("show_usage", True),
            data_dir=raw.get("session", {}).get("data_dir"),
            secure=raw.get("session", {}).get("secure", True),
        )

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text("\n".join(self._to_toml_lines()) + "\n")

    def _to_toml_lines(self) -> list[str]:
        lines = [
            "[model]",
            f'default = "{self.model}"',
        ]
        if self.provider:
            lines.append(f'provider = "{self.provider}"')
        lines += [
            "",
            "[keys]",
        ]
        for name, key in self.keys.items():
            lines.append(f'{name} = "{key}"')
        lines += [
            "",
            "[display]",
            f"show_tool_calls = {'true' if self.show_tool_calls else 'false'}",
            f"show_usage = {'true' if self.show_usage else 'false'}",
            "",
            "[session]",
        ]
        if self.data_dir:
            lines.append(f'data_dir = "{self.data_dir}"')
        lines.append(f"secure = {'true' if self.secure else 'false'}")
        return lines

    def get_api_key(self, provider_name: str) -> str | None:
        key = self.keys.get(provider_name)
        if key:
            return key
        env_var = _ENV_MAP.get(provider_name)
        return os.environ.get(env_var) if env_var else None
