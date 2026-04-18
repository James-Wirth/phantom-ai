"""Registry utilities for operation introspection and tool generation."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from ._ref import Ref


def _parse_docstring_args(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from a Google-style Args section."""
    if not docstring:
        return {}

    lines = docstring.split("\n")
    block_lines: list[str] = []
    args_indent: int | None = None
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped == "Args:" or stripped.startswith("Args:\n"):
            args_indent = indent
            continue
        if args_indent is None:
            continue
        if stripped and indent <= args_indent:
            break
        block_lines.append(line)

    if not block_lines:
        return {}

    param_re = r"^\s+(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)"
    params: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for line in block_lines:
        param_match = re.match(param_re, line)
        if param_match:
            if current_name is not None:
                params[current_name] = " ".join(current_lines).strip()
            current_name = param_match.group(1)
            first = param_match.group(2).strip()
            current_lines = [first] if first else []
        elif current_name is not None and line.strip():
            current_lines.append(line.strip())

    if current_name is not None:
        params[current_name] = " ".join(current_lines).strip()

    return params


def _is_ref_type(type_hint: Any) -> bool:
    """Check if a type hint is Ref or Ref[T]."""
    if type_hint is Ref:
        return True
    origin = get_origin(type_hint)
    return origin is Ref


def _extract_ref_inner_type(type_hint: Any) -> type | tuple[type, ...] | None:
    """
    Extract the inner type T from Ref[T].

    Args:
        type_hint: A type hint that is known to be Ref or Ref[T]

    Returns:
        - None if bare Ref (no type parameter)
        - A single type for Ref[SomeType]
        - A tuple of types for Ref[A | B] (union types)
    """
    if type_hint is Ref:
        return None

    args = get_args(type_hint)
    if not args:
        return None

    inner_type = args[0]

    inner_origin = get_origin(inner_type)
    if inner_origin is Union or inner_origin is UnionType:
        union_args = get_args(inner_type)
        types = tuple(t for t in union_args if t is not type(None))
        return types if len(types) > 1 else (types[0] if types else None)

    return inner_type


def _python_type_to_json_schema(type_name: str) -> str:
    """Map Python type names to JSON Schema types."""
    mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "None": "null",
        "NoneType": "null",
    }
    return mapping.get(type_name, "string")


def get_operation_signature_from_func(
    name: str, func: Callable[..., Any]
) -> dict[str, Any]:
    """
    Extract signature info from a function for tool generation.

    Args:
        name: The operation name to use in the signature
        func: The callable to introspect

    Returns:
        Dict with name, doc, params, and return_type
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    arg_docs = _parse_docstring_args(func.__doc__)

    params = {}
    for param_name, param in sig.parameters.items():
        param_info: dict[str, Any] = {}
        if param_name in arg_docs:
            param_info["doc"] = arg_docs[param_name]
        if param_name in hints:
            type_hint = hints[param_name]
            param_info["type_hint"] = type_hint
            if hasattr(type_hint, "__name__"):
                param_info["type"] = type_hint.__name__
            else:
                param_info["type"] = str(type_hint)
            param_info["is_ref"] = _is_ref_type(type_hint)

            if param_info["is_ref"]:
                param_info["ref_inner_type"] = _extract_ref_inner_type(type_hint)

        if param.default is not inspect.Parameter.empty:
            param_info["default"] = param.default
        params[param_name] = param_info

    return_hint = hints.get("return", Any)
    if hasattr(return_hint, "__name__"):
        return_type = return_hint.__name__
    else:
        return_type = str(return_hint)

    return {
        "name": name,
        "doc": func.__doc__,
        "params": params,
        "return_type": return_type,
    }


def get_tools(
    operations: dict[str, Callable[..., Any]],
    format: str = "openai",
    include_peek: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate tool definitions for operations.

    Args:
        operations: Operations dict to generate tools from.
        format: The schema format to use. Options: "openai", "anthropic"
        include_peek: Whether to include the peek tool (default True)

    Returns:
        A list of tool definitions in the specified format.

    Example:
        session = phantom.Session()

        @session.op
        def search(query: str, limit: int = 10) -> list[dict]:
            '''Search for items matching query.'''
            ...

        # OpenAI format (default)
        tools = session.get_tools()
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
        )

        # Anthropic format
        tools = session.get_tools(format="anthropic")
        response = anthropic.messages.create(
            model="claude-sonnet-4-6",
            messages=messages,
            tools=tools,
        )
    """
    if format not in ("openai", "anthropic"):
        raise ValueError(f"Unknown format: {format}. Use 'openai' or 'anthropic'.")

    tools = []

    for op_name, op_func in operations.items():
        sig = get_operation_signature_from_func(op_name, op_func)

        properties = {}
        required = []

        for param_name, param_info in sig["params"].items():
            is_ref = param_info.get("is_ref", False)
            doc = param_info.get("doc", "")
            if is_ref:
                ref_desc = doc or "A ref ID from a prior op"
                prop: dict[str, Any] = {
                    "type": "string",
                    "description": f"{ref_desc} (e.g., '@abc123')",
                    "pattern": "^@[a-f0-9]+$",
                }
            else:
                type_name = param_info.get("type", "str")
                prop = {
                    "type": _python_type_to_json_schema(type_name),
                    "description": doc or f"The {param_name} parameter",
                }
                if "dict" in type_name:
                    prop["type"] = "object"
                    prop["additionalProperties"] = {
                        "type": "string",
                        "pattern": "^@[a-f0-9]+$",
                    }
            properties[param_name] = prop

            if "default" not in param_info:
                required.append(param_name)

        name = sig["name"]
        description = sig["doc"] or f"Execute the {name} operation"
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        if format == "openai":
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
        else:  # anthropic
            tools.append({
                "name": name,
                "description": description,
                "input_schema": parameters,
            })

    if include_peek:
        peek_params = {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The ref ID to inspect (e.g., '@abc123')",
                    "pattern": "^@[a-f0-9]+$",
                }
            },
            "required": ["ref"],
        }
        peek_description = (
            "Inspect a ref to see its type, shape, columns, and sample "
            "data. Use this to understand structure before transforming."
        )

        if format == "openai":
            tools.append({
                "type": "function",
                "function": {
                    "name": "peek",
                    "description": peek_description,
                    "parameters": peek_params,
                },
            })
        else:  # anthropic
            tools.append({
                "name": "peek",
                "description": peek_description,
                "input_schema": peek_params,
            })

    return tools
