"""Phantom CLI -- interactive data analysis in the terminal."""

from __future__ import annotations


def main() -> None:
    import argparse

    from .app import PhantomApp

    parser = argparse.ArgumentParser(
        prog="phantom",
        description="Interactive data analysis with LLMs",
    )
    parser.add_argument("--data", "-d", help="Data directory path")
    parser.add_argument(
        "--model", "-m", help="Model name (e.g. claude-sonnet-4-6)"
    )
    args = parser.parse_args()

    app = PhantomApp(data_dir=args.data, model=args.model)
    app.run()
