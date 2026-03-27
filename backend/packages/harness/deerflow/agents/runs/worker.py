"""Background agent execution.

Runs an agent graph inside an ``asyncio.Task``, publishing events to
a :class:`StreamBridge` as they are produced.

Uses ``graph.astream(stream_mode=[...])`` which gives correct full-state
snapshots for ``values`` mode, proper ``{node: writes}`` for ``updates``,
and ``(chunk, metadata)`` tuples for ``messages`` mode.

Note: ``events`` mode is not supported through the gateway — it requires
``graph.astream_events()`` which cannot simultaneously produce ``values``
snapshots.  The JS open-source LangGraph API server works around this via
internal checkpoint callbacks that are not exposed in the Python public API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from deerflow.agents.runs.manager import RunManager, RunRecord
from deerflow.agents.runs.schemas import RunStatus
from deerflow.agents.stream_bridge.base import StreamBridge

logger = logging.getLogger(__name__)

# Valid stream_mode values for LangGraph's graph.astream()
_VALID_LG_MODES = {"values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"}

# Internal LangGraph keys that must NOT appear in values events
_PREGEL_INTERNAL_KEYS = frozenset(
    k
    for k in (
        "__pregel_tasks",
        "__pregel_resuming",
        "__pregel_pull",
        "__pregel_push",
        "__pregel_dedupe",
        "__pregel_store",
    )
)


async def run_agent(
    bridge: StreamBridge,
    run_manager: RunManager,
    record: RunRecord,
    *,
    checkpointer: Any,
    store: Any | None = None,
    agent_factory: Any,
    graph_input: dict,
    config: dict,
    stream_modes: list[str] | None = None,
    stream_subgraphs: bool = False,
    interrupt_before: list[str] | Literal["*"] | None = None,
    interrupt_after: list[str] | Literal["*"] | None = None,
) -> None:
    """Execute an agent in the background, publishing events to *bridge*."""

    run_id = record.run_id
    thread_id = record.thread_id
    requested_modes: set[str] = set(stream_modes or ["values"])

    # Track whether "events" was requested but skipped
    if "events" in requested_modes:
        logger.info(
            "Run %s: 'events' stream_mode not supported in gateway (requires astream_events + checkpoint callbacks). Skipping.",
            run_id,
        )

    try:
        # 1. Mark running
        await run_manager.set_status(run_id, RunStatus.running)

        # Record pre-run checkpoint_id to support rollback (Phase 2).
        pre_run_checkpoint_id = None
        try:
            config_for_check = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            ckpt_tuple = await checkpointer.aget_tuple(config_for_check)
            if ckpt_tuple is not None:
                pre_run_checkpoint_id = getattr(ckpt_tuple, "config", {}).get("configurable", {}).get("checkpoint_id")
        except Exception:
            logger.debug("Could not get pre-run checkpoint_id for run %s", run_id)

        # 2. Publish metadata — useStream needs both run_id AND thread_id
        await bridge.publish(
            run_id,
            "metadata",
            {
                "run_id": run_id,
                "thread_id": thread_id,
            },
        )

        # 3. Build the agent
        from langchain_core.runnables import RunnableConfig
        from langgraph.runtime import Runtime

        # Inject runtime context so middlewares can access thread_id
        # (langgraph-cli does this automatically; we must do it manually)
        runtime = Runtime(context={"thread_id": thread_id}, store=store)
        config.setdefault("configurable", {})["__pregel_runtime"] = runtime

        runnable_config = RunnableConfig(**config)
        agent = agent_factory(config=runnable_config)

        # 4. Attach checkpointer and store
        if checkpointer is not None:
            agent.checkpointer = checkpointer
        if store is not None:
            agent.store = store

        # 5. Set interrupt nodes
        if interrupt_before:
            agent.interrupt_before_nodes = interrupt_before
        if interrupt_after:
            agent.interrupt_after_nodes = interrupt_after

        # 6. Build LangGraph stream_mode list
        #    "events" is NOT a valid astream mode — skip it
        #    "messages-tuple" maps to LangGraph's "messages" mode
        lg_modes: list[str] = []
        for m in requested_modes:
            if m == "messages-tuple":
                lg_modes.append("messages")
            elif m == "events":
                # Skipped — see log above
                continue
            elif m in _VALID_LG_MODES:
                lg_modes.append(m)
        if not lg_modes:
            lg_modes = ["values"]

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for m in lg_modes:
            if m not in seen:
                seen.add(m)
                deduped.append(m)
        lg_modes = deduped

        logger.info("Run %s: streaming with modes %s (requested: %s)", run_id, lg_modes, requested_modes)

        # 7. Stream using graph.astream
        if len(lg_modes) == 1 and not stream_subgraphs:
            # Single mode, no subgraphs: astream yields raw chunks
            single_mode = lg_modes[0]
            async for chunk in agent.astream(graph_input, config=runnable_config, stream_mode=single_mode):
                if record.abort_event.is_set():
                    logger.info("Run %s abort requested — stopping", run_id)
                    break
                sse_event = _lg_mode_to_sse_event(single_mode)
                await bridge.publish(run_id, sse_event, _serialize(chunk, mode=single_mode))
        else:
            # Multiple modes or subgraphs: astream yields tuples
            async for item in agent.astream(
                graph_input,
                config=runnable_config,
                stream_mode=lg_modes,
                subgraphs=stream_subgraphs,
            ):
                if record.abort_event.is_set():
                    logger.info("Run %s abort requested — stopping", run_id)
                    break

                mode, chunk = _unpack_stream_item(item, lg_modes, stream_subgraphs)
                if mode is None:
                    continue

                sse_event = _lg_mode_to_sse_event(mode)
                await bridge.publish(run_id, sse_event, _serialize(chunk, mode=mode))

        # 8. Final status
        if record.abort_event.is_set():
            action = record.abort_action
            if action == "rollback":
                await run_manager.set_status(run_id, RunStatus.error, error="Rolled back by user")
                # TODO(Phase 2): Implement full checkpoint rollback.
                # Use pre_run_checkpoint_id to revert the thread's checkpoint
                # to the state before this run started. Requires a
                # checkpointer.adelete() or equivalent API.
                try:
                    if checkpointer is not None and pre_run_checkpoint_id is not None:
                        # Phase 2: roll back to pre_run_checkpoint_id
                        pass
                    logger.info("Run %s rolled back", run_id)
                except Exception:
                    logger.warning("Failed to rollback checkpoint for run %s", run_id)
            else:
                await run_manager.set_status(run_id, RunStatus.interrupted)
        else:
            await run_manager.set_status(run_id, RunStatus.success)

    except asyncio.CancelledError:
        action = record.abort_action
        if action == "rollback":
            await run_manager.set_status(run_id, RunStatus.error, error="Rolled back by user")
            logger.info("Run %s was cancelled (rollback)", run_id)
        else:
            await run_manager.set_status(run_id, RunStatus.interrupted)
            logger.info("Run %s was cancelled", run_id)

    except Exception as exc:
        error_msg = f"{exc}"
        logger.exception("Run %s failed: %s", run_id, error_msg)
        await run_manager.set_status(run_id, RunStatus.error, error=error_msg)
        await bridge.publish(
            run_id,
            "error",
            {
                "message": error_msg,
                "name": type(exc).__name__,
            },
        )

    finally:
        await bridge.publish_end(run_id)
        asyncio.create_task(bridge.cleanup(run_id, delay=60))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lg_mode_to_sse_event(mode: str) -> str:
    """Map LangGraph internal stream_mode name to SSE event name.

    LangGraph's ``astream(stream_mode="messages")`` produces message
    tuples.  The SSE protocol calls this ``messages-tuple`` when the
    client explicitly requests it, but the default SSE event name used
    by LangGraph Platform is simply ``"messages"``.
    """
    # All LG modes map 1:1 to SSE event names — "messages" stays "messages"
    return mode


def _unpack_stream_item(
    item: Any,
    lg_modes: list[str],
    stream_subgraphs: bool,
) -> tuple[str | None, Any]:
    """Unpack a multi-mode or subgraph stream item into (mode, chunk).

    Returns ``(None, None)`` if the item cannot be parsed.
    """
    if stream_subgraphs:
        if isinstance(item, tuple) and len(item) == 3:
            _ns, mode, chunk = item
            return str(mode), chunk
        if isinstance(item, tuple) and len(item) == 2:
            mode, chunk = item
            return str(mode), chunk
        return None, None

    if isinstance(item, tuple) and len(item) == 2:
        mode, chunk = item
        return str(mode), chunk

    # Fallback: single-element output from first mode
    return lg_modes[0] if lg_modes else None, item


def _serialize(obj: Any, *, mode: str = "") -> Any:
    """Serialize LangChain objects to JSON-serialisable dicts.

    Special handling per mode:

    * ``messages`` — obj is ``(message_chunk, metadata_dict)``; returns
      ``[serialized_chunk, metadata]`` to match LangGraph Platform format.
    * ``values`` — obj is the full state dict; internal ``__pregel_*``
      keys are stripped.
    * everything else — recursive ``model_dump()`` / ``dict()`` fallback.
    """
    if mode == "messages":
        return _serialize_messages_tuple(obj)
    if mode == "values":
        return _serialize_values_state(obj)
    return _serialize_lc_object(obj)


def _serialize_messages_tuple(obj: Any) -> Any:
    """Serialize a messages-mode tuple ``(chunk, metadata)``."""
    if isinstance(obj, tuple) and len(obj) == 2:
        chunk, metadata = obj
        return [_serialize_lc_object(chunk), metadata if isinstance(metadata, dict) else {}]
    # Fallback
    return _serialize_lc_object(obj)


def _serialize_values_state(state: Any) -> Any:
    """Serialize a values-mode full state snapshot.

    Strips internal ``__pregel_*`` keys and ``__interrupt__`` that
    LangGraph injects into the state dict.  These are not part of the
    public API and will confuse the ``useStream`` React hook.
    """
    if not isinstance(state, dict):
        return _serialize_lc_object(state)

    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        # Skip LangGraph internal channels
        if key.startswith("__pregel_") or key == "__interrupt__":
            continue
        cleaned[key] = _serialize_lc_object(value)
    return cleaned


def _serialize_lc_object(obj: Any) -> Any:
    """Recursively serialize a LangChain object to a JSON-serialisable dict."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize_lc_object(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_lc_object(item) for item in obj]
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1 / older objects
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    # Last resort
    try:
        return str(obj)
    except Exception:
        return repr(obj)
