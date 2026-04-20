"""Queued job endpoints."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import queue
import time

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
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


@router.get("/v1/jobs/{job_id}/trace")
async def get_job_trace(job_id: str, request: Request):
    engine = request.app.state.engine
    job = engine.get_job(job_id)
    if job is None or not job.trace_id:
        raise HTTPException(status_code=404, detail="Job not found")
    trace_payload = request.app.state.tracer.get_trace(job.trace_id)
    if trace_payload is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace_payload


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


@router.websocket("/v1/jobs/{job_id}/stream")
async def websocket_job_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    engine = websocket.app.state.engine
    job = engine.get_job(job_id)
    if job is None:
        await websocket.close(code=4404, reason="Job not found")
        return
    channel = engine.store.subscribe(job_id)
    try:
        for event in job.events:
            await websocket.send_json(event.__dict__)
        while True:
            event_task = asyncio.create_task(asyncio.to_thread(channel.get, True, 1.0))
            receive_task = asyncio.create_task(websocket.receive_text())
            done, pending = await asyncio.wait(
                {event_task, receive_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await task

            if receive_task in done:
                try:
                    raw_message = receive_task.result()
                except WebSocketDisconnect:
                    break
                payload = json.loads(raw_message or "{}")
                if payload.get("action") == "cancel":
                    cancelled = engine.cancel(job_id)
                    await websocket.send_json(
                        {
                            "event": "control_ack",
                            "stage": "job",
                            "timestamp_ms": int(time.time() * 1000),
                            "data": {
                                "job_id": job_id,
                                "cancelled": bool(cancelled and cancelled.state == "cancelled"),
                            },
                        }
                    )
            if event_task in done:
                try:
                    next_event = event_task.result()
                except queue.Empty:
                    await websocket.send_json({"event": "keep_alive", "stage": "job", "timestamp_ms": 0, "data": {}})
                    continue
                if next_event is None:
                    break
                await websocket.send_json(next_event.__dict__)
                latest = engine.get_job(job_id)
                if latest is not None and latest.state in {"succeeded", "failed", "cancelled"}:
                    break
    finally:
        engine.store.unsubscribe(job_id, channel)
