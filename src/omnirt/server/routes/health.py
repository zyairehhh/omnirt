"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return {"ok": True}


@router.get("/readyz")
async def readyz(request: Request):
    engine = request.app.state.engine
    return {"ok": bool(engine.is_ready())}
