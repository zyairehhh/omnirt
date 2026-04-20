"""Queued job endpoints."""

from __future__ import annotations

import asyncio
import queue

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from omnirt.server.sse import encode_sse_event

router = APIRouter()


@router.post("/v1/jobs")
async def create_job(payload, request: Request):
    generate_router = request.app.router
    del generate_router
    raise HTTPException(status_code=501, detail="Use POST /v1/generate with async_run=true.")


@router.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    job = request.app.state.engine.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request):
    job = request.app.state.engine.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/v1/jobs/{job_id}/events")
async def stream_job_events(job_id: str, request: Request):
    engine = request.app.state.engine
    job = engine.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    channel = engine.store.subscribe(job_id)

    async def event_stream():
        try:
            for event in job.events:
                yield encode_sse_event(event)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    next_event = await asyncio.to_thread(channel.get, True, 15.0)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if next_event is None:
                    break
                yield encode_sse_event(next_event)
                latest = engine.get_job(job_id)
                if latest is not None and latest.state in {"succeeded", "failed", "cancelled"}:
                    break
        finally:
            engine.store.unsubscribe(job_id, channel)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
