<h1>
<p align="center">
  <img src="https://raw.githubusercontent.com/James-Wirth/phantom-ai/main/assets/logo.png" alt="Phantom" width="80">
  <br>phantom
</h1>
  <p align="center">
    Sandboxed data analysis with LLMs (powered by DuckDB).
    <br><br>
    <a href="https://github.com/James-Wirth/phantom-ai/actions/workflows/ci.yml"><img src="https://github.com/James-Wirth/phantom-ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  </p>
</p>

Phantom is a Python framework for LLM-assisted data analysis. The LLM doesn't need to see the actual data. Phantom reasons with opaque **semantic references** (`@a3f2`), writes SQL, and executes the queries locally in a sandboxed [DuckDB](https://duckdb.org/) engine.

## Quick Start

```bash
pip install phantom-ai
```

```python
import os
import phantom

session = phantom.Session(data_dir="~/data/exoplanets")

chat = phantom.Chat(
    session,
    provider="anthropic",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
    system="You are an astrophysicist.",
)

response = chat.ask(
    "Which habitable-zone exoplanets are within 50 light-years of Earth, "
    "and what kind of stars do they orbit?"
)
```

## How It Works

Given two CSV files and the question *"Which habitable-zone exoplanets are within 50 light-years of Earth, and what kind of stars do they orbit?"*, Phantom produces this tool-call trace:

```
[0] read_csv("exoplanets.csv")            → @6a97
[1] read_csv("stars.csv")                 → @cc35
[2] query({p: @6a97})                     → @b1a0  -- habitable-zone filter
[3] query({s: @cc35})                     → @f4e2  -- nearby stars (< 50 ly)
[4] query({hz: @b1a0, nb: @f4e2})         → @31d7  -- join + rank by distance
[5] export(@31d7)                         → [{name: "Proxima Cen b", ...}]
```

The semantic refs (`@6a97`, `@cc35`, ...) compose into a lazy execution graph:

```
@6a97 → @b1a0 ─┐
                ├→ @31d7
@cc35 → @f4e2 ─┘
```

Shared subgraphs are resolved once and cached. The query engine is [DuckDB](https://duckdb.org/), so JOINs, window functions, CTEs, and aggregations all work natively.

Claude's answer (abridged):

> | Planet | Distance | Star | Spectral type |
> |:-------|:---------|:-----|:--------------|
> | Proxima Cen b | 4.2 ly | Proxima Cen | M-dwarf (3,042 K) |
> | Ross 128 b | 11 ly | Ross 128 | M-dwarf (3,192 K) |
> | Teegarden b | 12 ly | Teegarden | M-dwarf (2,904 K) |
> | GJ 667 Cc | 24 ly | GJ 667 C | M-dwarf (3,350 K) |
> | TRAPPIST-1 e/f/g | 40 ly | TRAPPIST-1 | M-dwarf (2,566 K) |
> | LHS 1140 b | 41 ly | LHS 1140 | M-dwarf (3,216 K) |
> | HD 40307 g | 42 ly | HD 40307 | K-dwarf (4,977 K) |
>
> The nearest habitable-zone candidates overwhelmingly orbit **M-dwarf** stars — small, cool, and the most common type in the galaxy.

## LLM Providers

Built-in support for **Anthropic**, **OpenAI**, and **Google Gemini**:

```bash
pip install "phantom-ai[anthropic]"
pip install "phantom-ai[openai]"
pip install "phantom-ai[google]"
```

```python
chat = phantom.Chat(
    session,
    provider="anthropic",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",
)
chat = phantom.Chat(
    session,
    provider="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o",
)
chat = phantom.Chat(
    session,
    provider="google",
    api_key=os.environ["GOOGLE_API_KEY"],
    model="gemini-2.0-flash",
)
```

Phantom also honours each SDK's native env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`) when `api_key` is omitted — useful for CI.

Any **OpenAI-compatible** API (Groq, Together, Fireworks, Ollama, vLLM, ...) works via `base_url`:

```python
chat = phantom.Chat(
    session,
    provider=phantom.OpenAIProvider(
        api_key="...",
        base_url="https://api.groq.com/openai/v1",
    ),
    model="llama-3.1-70b-versatile",
)
```

## Custom Operations

Register domain-specific tools alongside the built-ins — the LLM can call them like any other operation:

```python
@session.op
def fetch_lightcurve(target: str) -> dict:
    """Fetch a lightcurve from the MAST archive."""
    return mast_api.query(target)
```