"""Chat - Unified LLM interface for Phantom sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._errors import MaxTurnsError
from ._providers import (
    CallOptions,
    LLMProvider,
    Usage,
    _infer_provider,
    get_provider,
)
from ._ref import Ref
from ._session import Session
from ._system_prompt import build_system_prompt


@dataclass
class ChatResponse:
    """Result from ``Chat.ask()``.

    Attributes:
        text: The LLM's final text response.
        refs: Refs created during tool calls in this invocation.
        tool_calls_made: Total number of tool calls executed.
        turns: Number of LLM round-trips.
        model: The model identifier used.
        stop_reason: Why the LLM stopped generating.
        usage: Aggregated token usage across all turns.
    """

    text: str
    refs: list[Ref] = field(default_factory=list)
    tool_calls_made: int = 0
    turns: int = 0
    model: str = ""
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)


class Chat:
    """Unified LLM interface for Phantom sessions.

    Manages a multi-turn conversation with tool use, automatically
    handling the tool-call loop for any registered provider.
    Includes a Phantom system prompt explaining how refs, peek,
    and tool chaining work.

    The *provider* argument accepts a name, a provider instance,
    or ``None`` (auto-detects from *model*).

    Example::

        session = phantom.Session()

        @session.op
        def load_csv(path: str) -> pd.DataFrame:
            return pd.read_csv(path)

        # Pass the API key directly
        chat = phantom.Chat(
            session,
            provider="anthropic",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

        # Auto-detect provider from model name
        chat = phantom.Chat(
            session,
            model="gpt-4o",
            api_key=os.environ["OPENAI_API_KEY"],
        )

        # Pre-configured provider instance (custom base_url, etc.)
        chat = phantom.Chat(
            session,
            provider=phantom.OpenAIProvider(
                api_key="sk-...",
                base_url="https://api.groq.com/openai/v1",
            ),
            model="llama-3.1-70b-versatile",
        )

        response = chat.ask("Top 5 orders by revenue?")
        print(response.text)
        print(response.usage.total_tokens)
    """

    def __init__(
        self,
        session: Session,
        *,
        provider: str | LLMProvider | None = None,
        api_key: str | None = None,
        model: str | None = None,
        system: str = "",
        client: Any | None = None,
        max_tokens: int = 4096,
        max_turns: int = 50,
        catch_errors: bool = True,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        max_retries: int = 2,
        extra: dict[str, Any] | None = None,
    ):
        """Create a Chat bound to a Phantom session.

        Args:
            session: The Phantom session with registered ops.
            provider: LLM provider — a name (``"anthropic"``,
                ``"openai"``, ``"google"``), a provider
                instance, or ``None`` to auto-detect from
                *model* (falls back to ``"anthropic"``).
            api_key: API key forwarded to the provider. Raises
                ``TypeError`` if combined with a pre-built
                ``LLMProvider`` instance — set the key on the
                provider instead. Ignored when *client* is
                also set. If omitted, the underlying SDK falls
                back to its native environment variable
                (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
                ``GOOGLE_API_KEY``).
            model: Model name (defaults per provider).
            system: Developer system prompt appended to
                Phantom's built-in prompt.
            client: Pre-configured SDK client. When set,
                *api_key* is ignored.
            max_tokens: Maximum tokens per LLM response.
            max_turns: Safety limit on round-trips per ask().
            catch_errors: If True, resolution errors are sent
                to the LLM as error tool results.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop_sequences: Stop sequences for generation.
            max_retries: Retries on transient network errors.
            extra: Provider-specific kwargs forwarded to the
                SDK call.
        """
        self._session = session

        if isinstance(provider, LLMProvider):
            if api_key is not None:
                raise TypeError(
                    "Cannot pass api_key together with a "
                    "pre-configured provider instance. Set "
                    "the key on the provider instead: "
                    "AnthropicProvider(api_key=...)."
                )
            self._provider = provider
            self._provider_name = type(provider).__name__
        elif isinstance(provider, str):
            self._provider = get_provider(provider, api_key=api_key)
            self._provider_name = provider
        elif provider is None:
            inferred = (
                _infer_provider(model)
                if model is not None
                else None
            )
            name = inferred or "anthropic"
            self._provider = get_provider(name, api_key=api_key)
            self._provider_name = name
        else:
            raise TypeError(
                f"provider must be a string, LLMProvider "
                f"instance, or None — "
                f"got {type(provider).__name__}"
            )

        self._model = model or self._provider.default_model()
        self._developer_system = system
        self._client = client or self._provider.create_client()
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._catch_errors = catch_errors
        self._temperature = temperature
        self._top_p = top_p
        self._stop_sequences = stop_sequences
        self._max_retries = max_retries
        self._extra = extra or {}

        self._messages: list[dict[str, Any]] = []
        self._refs: list[Ref] = []

    def ask(self, message: str) -> ChatResponse:
        """Send a message and run the full tool-call loop.

        Appends to the internal message history, so subsequent
        calls continue the same conversation.

        Args:
            message: The user message to send.

        Returns:
            ChatResponse with text, refs, usage, and metadata.

        Raises:
            MaxTurnsError: If the loop exceeds max_turns.
        """
        self._messages.append(
            {"role": "user", "content": message}
        )

        system = build_system_prompt(
            self._session.operations,
            self._developer_system,
        )
        tools = self._provider.get_tools(self._session)

        refs_created: list[Ref] = []
        total_tool_calls = 0
        total_usage = Usage()
        turns = 0

        while turns < self._max_turns:
            turns += 1

            options = CallOptions(
                client=self._client,
                model=self._model,
                system=system,
                messages=self._messages,
                tools=tools,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                stop_sequences=self._stop_sequences,
                extra=self._extra,
            )
            response = self._call_with_retry(options)

            total_usage = Usage(
                input_tokens=(
                    total_usage.input_tokens
                    + response.usage.input_tokens
                ),
                output_tokens=(
                    total_usage.output_tokens
                    + response.usage.output_tokens
                ),
            )

            self._messages.append(
                self._provider.format_assistant_message(
                    response
                )
            )

            if not response.tool_calls:
                return ChatResponse(
                    text=response.text or "",
                    refs=refs_created,
                    tool_calls_made=total_tool_calls,
                    turns=turns,
                    model=self._model,
                    stop_reason=response.stop_reason,
                    usage=total_usage,
                )

            result_blocks = []
            for tc in response.tool_calls:
                total_tool_calls += 1
                result = self._session.handle_tool_call(
                    tc.name,
                    tc.arguments,
                    catch_errors=self._catch_errors,
                )

                if (
                    result.ref is not None
                    and result.kind == "ref"
                ):
                    refs_created.append(result.ref)
                    self._refs.append(result.ref)

                result_blocks.append(
                    self._provider.format_tool_result_block(
                        tc.id,
                        result.to_json(),
                        result.is_error,
                    )
                )

            tool_messages = (
                self._provider.format_tool_results(
                    result_blocks
                )
            )
            self._messages.extend(tool_messages)

        raise MaxTurnsError(
            turns=turns, max_turns=self._max_turns
        )

    def _call_with_retry(
        self, options: CallOptions
    ) -> Any:
        """Call the provider with retry on transient errors.

        Retries on network-level exceptions only. API errors
        (auth, rate-limit, bad request) are not retried --
        both the Anthropic and OpenAI SDKs handle rate-limit
        retries internally.
        """
        attempts = max(1, self._max_retries)
        for attempt in range(attempts):
            try:
                return self._provider.call(options)
            except (
                ConnectionError,
                TimeoutError,
                OSError,
            ):
                if attempt + 1 == attempts:
                    raise

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Read-only view of the conversation history."""
        return list(self._messages)

    @property
    def refs(self) -> list[Ref]:
        """All refs created during this chat's tool calls."""
        return list(self._refs)

    def reset(self) -> None:
        """Clear conversation history and ref tracking.

        Does not affect the underlying session's refs or cache.
        """
        self._messages.clear()
        self._refs.clear()
