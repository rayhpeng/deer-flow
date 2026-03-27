"""Stateless runs endpoints -- stream and wait without a pre-existing thread.

These endpoints auto-create a temporary thread, execute the run, and
optionally delete the thread on completion.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.gateway.routers.thread_runs import (
    RunCreateRequest,
    _get_bridge,
    _get_checkpointer,
    _get_run_manager,
    _sse_consumer,
    _start_run,
)
from app.gateway.routers.threads import _serialize_channel_values

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("/stream")
async def stateless_stream(body: RunCreateRequest, request: Request) -> StreamingResponse:
    """Create a run on a temporary thread and stream events via SSE."""
    thread_id = str(uuid.uuid4())
    bridge = _get_bridge(request)
    run_mgr = _get_run_manager(request)
    record = await _start_run(body, thread_id, request)

    return StreamingResponse(
        _sse_consumer(bridge, record, request, run_mgr),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/wait", response_model=dict)
async def stateless_wait(body: RunCreateRequest, request: Request) -> dict:
    """Create a run on a temporary thread and block until completion."""
    thread_id = str(uuid.uuid4())
    record = await _start_run(body, thread_id, request)

    if record.task is not None:
        try:
            await record.task
        except asyncio.CancelledError:
            pass

    checkpointer = _get_checkpointer(request)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is not None:
            checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
            channel_values = checkpoint.get("channel_values", {})
            return _serialize_channel_values(channel_values)
    except Exception:
        logger.exception("Failed to fetch final state for run %s", record.run_id)

    return {"status": record.status.value, "error": record.error}
