"""Minimal OpenAI-compatible routes."""

from __future__ import annotations

import tempfile
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from omnirt.api import validate
from omnirt.core.types import GenerateRequest, is_generate_result_like
from omnirt.server.model_aliases import resolve_model_alias
from omnirt.server.request_config import normalize_generate_request

router = APIRouter()


def _resolve_backend(request: Request, backend: str | None) -> str:
    return backend or request.app.state.default_backend


def _image_size_to_config(size: str | None) -> dict:
    if not size:
        return {}
    try:
        width, height = size.lower().split("x", 1)
        return {"width": int(width), "height": int(height)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid size format: {size!r}") from exc


def _result_to_openai_images(result) -> dict:
    return {
        "created": int(time.time()),
        "data": [{"url": artifact.path} for artifact in result.outputs],
    }


@router.post("/v1/images/generations")
async def openai_images_generations(payload: dict, request: Request):
    model = resolve_model_alias(str(payload["model"]), request.app.state.model_aliases)
    req = normalize_generate_request(
        GenerateRequest(
        task="text2image",
        model=model,
        backend=_resolve_backend(request, payload.get("backend")),
        inputs={"prompt": str(payload["prompt"])},
        config={**_image_size_to_config(payload.get("size")), "num_images_per_prompt": int(payload.get("n", 1))},
        ),
        request.app.state,
    )
    validation = validate(req, backend=req.backend)
    if not validation.ok:
        raise HTTPException(status_code=400, detail=validation.format_errors())
    result = request.app.state.engine.run_sync(req, model_spec=validation.model_spec)
    if not is_generate_result_like(result):
        raise HTTPException(status_code=500, detail="Unexpected non-generate result")
    return _result_to_openai_images(result)


@router.post("/v1/images/edits")
async def openai_images_edits(
    request: Request,
    model: str = Form(...),
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile | None = File(default=None),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"-{image.filename or 'image'}") as image_file:
        image_file.write(await image.read())
        image_path = image_file.name
    mask_path = None
    if mask is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"-{mask.filename or 'mask'}") as mask_file:
            mask_file.write(await mask.read())
            mask_path = mask_file.name

    task = "inpaint" if mask_path else "image2image"
    inputs = {"image": image_path, "prompt": prompt}
    if mask_path:
        inputs["mask"] = mask_path
    req = normalize_generate_request(
        GenerateRequest(
            task=task,
            model=resolve_model_alias(model, request.app.state.model_aliases),
            backend=request.app.state.default_backend,
            inputs=inputs,
            config={},
        ),
        request.app.state,
    )
    validation = validate(req, backend=req.backend)
    if not validation.ok:
        raise HTTPException(status_code=400, detail=validation.format_errors())
    result = request.app.state.engine.run_sync(req, model_spec=validation.model_spec)
    if not is_generate_result_like(result):
        raise HTTPException(status_code=500, detail="Unexpected non-generate result")
    return _result_to_openai_images(result)


@router.post("/v1/videos/generations")
async def openai_videos_generations(payload: dict, request: Request):
    model = resolve_model_alias(str(payload["model"]), request.app.state.model_aliases)
    task = "image2video" if payload.get("image") else "text2video"
    inputs = {"prompt": payload.get("prompt")} if payload.get("prompt") else {}
    if payload.get("image"):
        inputs["image"] = payload["image"]
    if payload.get("num_frames") is not None:
        inputs["num_frames"] = int(payload["num_frames"])
    if payload.get("fps") is not None:
        inputs["fps"] = int(payload["fps"])
    req = normalize_generate_request(
        GenerateRequest(
            task=task,
            model=model,
            backend=_resolve_backend(request, payload.get("backend")),
            inputs=inputs,
            config={},
        ),
        request.app.state,
    )
    validation = validate(req, backend=req.backend)
    if not validation.ok:
        raise HTTPException(status_code=400, detail=validation.format_errors())
    result = request.app.state.engine.run_sync(req, model_spec=validation.model_spec)
    if not is_generate_result_like(result):
        raise HTTPException(status_code=500, detail="Unexpected non-generate result")
    return {
        "created": int(time.time()),
        "data": [{"url": artifact.path} for artifact in result.outputs],
    }


@router.post("/v1/audio/speech")
async def openai_audio_speech():
    raise HTTPException(status_code=501, detail="audio/speech compatibility is reserved for a later phase")
