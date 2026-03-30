"""Async Store factory — backend mirrors the configured checkpointer.

The store and checkpointer share the same ``checkpointer`` section in
*config.yaml* so they always use the same persistence backend:

- ``type: memory``   → :class:`langgraph.store.memory.InMemoryStore`
- ``type: sqlite``   → :class:`langgraph.store.sqlite.aio.AsyncSqliteStore`
- ``type: postgres`` → :class:`langgraph.store.postgres.aio.AsyncPostgresStore`

Usage (e.g. FastAPI lifespan)::

    from deerflow.runtime.store import make_store

    async with make_store() as store:
        app.state.store = store
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
from collections.abc import AsyncIterator

from langgraph.store.base import BaseStore

from deerflow.agents.checkpointer.provider import _resolve_sqlite_conn_str
from deerflow.config.app_config import get_app_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error message constants
# ---------------------------------------------------------------------------

SQLITE_STORE_INSTALL = (
    "langgraph-checkpoint-sqlite is required for the SQLite store. "
    "Install it with: uv add langgraph-checkpoint-sqlite"
)
POSTGRES_STORE_INSTALL = (
    "langgraph-checkpoint-postgres is required for the PostgreSQL store. "
    "Install it with: uv add langgraph-checkpoint-postgres psycopg[binary] psycopg-pool"
)
POSTGRES_CONN_REQUIRED = "checkpointer.connection_string is required for the postgres backend"

# ---------------------------------------------------------------------------
# Internal backend factory
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_store(config) -> AsyncIterator[BaseStore]:
    """Async context manager that constructs and tears down a Store.

    The ``config`` argument is a :class:`deerflow.config.checkpointer_config.CheckpointerConfig`
    instance — the same object used by the checkpointer factory.
    """
    if config.type == "memory":
        from langgraph.store.memory import InMemoryStore

        logger.info("Store: using InMemoryStore (in-process, not persistent)")
        yield InMemoryStore()
        return

    if config.type == "sqlite":
        try:
            from langgraph.store.sqlite.aio import AsyncSqliteStore
        except ImportError as exc:
            raise ImportError(SQLITE_STORE_INSTALL) from exc

        conn_str = _resolve_sqlite_conn_str(config.connection_string or "store.db")
        # Ensure the parent directory exists for real filesystem paths
        if conn_str != ":memory:" and not conn_str.startswith("file:"):
            pathlib.Path(conn_str).parent.mkdir(parents=True, exist_ok=True)

        async with AsyncSqliteStore.from_conn_string(conn_str) as store:
            await store.setup()
            logger.info("Store: using AsyncSqliteStore (%s)", conn_str)
            yield store
        return

    if config.type == "postgres":
        try:
            from langgraph.store.postgres.aio import AsyncPostgresStore  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(POSTGRES_STORE_INSTALL) from exc

        if not config.connection_string:
            raise ValueError(POSTGRES_CONN_REQUIRED)

        async with AsyncPostgresStore.from_conn_string(config.connection_string) as store:
            await store.setup()
            logger.info("Store: using AsyncPostgresStore")
            yield store
        return

    raise ValueError(f"Unknown store backend type: {config.type!r}")


# ---------------------------------------------------------------------------
# Public async context manager
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def make_store() -> AsyncIterator[BaseStore]:
    """Async context manager that yields a Store whose backend matches the
    configured checkpointer.

    Reads from the same ``checkpointer`` section of *config.yaml* used by
    :func:`deerflow.agents.checkpointer.async_provider.make_checkpointer` so
    that both singletons always use the same persistence technology::

        async with make_store() as store:
            app.state.store = store

    Yields an :class:`~langgraph.store.memory.InMemoryStore` when no
    ``checkpointer`` section is configured (emits a WARNING in that case).
    """
    config = get_app_config()

    if config.checkpointer is None:
        from langgraph.store.memory import InMemoryStore

        logger.warning(
            "No 'checkpointer' section in config.yaml — using InMemoryStore for the store. "
            "Thread list will be lost on server restart. "
            "Configure a sqlite or postgres backend for persistence."
        )
        yield InMemoryStore()
        return

    async with _async_store(config.checkpointer) as store:
        yield store
