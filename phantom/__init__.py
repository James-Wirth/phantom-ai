"""
Phantom - Sandboxed data analysis with LLMs (powered by DuckDB).

Sessions come with built-in operations (read_csv, read_parquet, read_json,
query, export) that run in a sandboxed DuckDB engine. You can extend them
with custom operations via ``@session.op``.

Paths support ``~``, ``$ENV_VAR``, and optional ``base_dir`` anchoring::

    session = phantom.Session(data_dir="~/data/sales")
    session = phantom.Session(data_dir="$DATA_DIR")
    session = phantom.Session(data_dir="./data", base_dir=Path(__file__).parent)

Example — query with refs::

    import phantom

    session = phantom.Session(data_dir="~/data")

    # Load data (lazy — nothing executes yet)
    orders = session.ref("read_csv", path="~/data/orders.csv")
    customers = session.ref("read_csv", path="~/data/customers.csv")

    # The refs dict maps alias -> ref; each alias becomes a SQL table name
    result = session.ref(
        "query",
        sql="SELECT c.name, SUM(o.amount) AS total "
            "FROM orders o JOIN customers c ON o.customer_id = c.id "
            "GROUP BY c.name ORDER BY total DESC",
        refs={"orders": orders, "customers": customers},
    )

    # Resolve executes the full graph
    table = session.resolve(result)

    # Or use phantom.Chat for LLM-driven analysis
    chat = phantom.Chat(session, provider="anthropic", model="claude-sonnet-4-6")
    response = chat.ask("Who are the top customers by revenue?")
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phantom-ai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from ._chat import Chat, ChatResponse
from ._errors import CycleError, MaxTurnsError, ResolutionError, TypeValidationError
from ._operation_set import OperationSet
from ._paths import resolve_path
from ._providers import (
    AnthropicProvider,
    CallOptions,
    GoogleProvider,
    LLMProvider,
    OpenAIProvider,
    ProviderResponse,
    ProviderToolCall,
    Usage,
    get_provider,
    register_provider,
)
from ._ref import Ref
from ._result import ToolResult
from ._security import (
    DEFAULT_DENY_PATTERNS,
    FileSizeGuard,
    Guard,
    PathGuard,
    SecurityError,
    SecurityPolicy,
)
from ._session import Session

__all__ = [
    "__version__",
    # Core types
    "Ref",
    "ToolResult",
    "Session",
    "OperationSet",
    # LLM interface
    "Chat",
    "ChatResponse",
    # Provider interface
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "CallOptions",
    "Usage",
    "ProviderResponse",
    "ProviderToolCall",
    "get_provider",
    "register_provider",
    # Paths
    "resolve_path",
    # Security
    "DEFAULT_DENY_PATTERNS",
    "SecurityError",
    "SecurityPolicy",
    "Guard",
    "PathGuard",
    "FileSizeGuard",
    # Errors
    "ResolutionError",
    "TypeValidationError",
    "CycleError",
    "MaxTurnsError",
]
