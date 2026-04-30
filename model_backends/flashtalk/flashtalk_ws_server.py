#!/usr/bin/env python3
from __future__ import annotations

"""
FlashTalk WebSocket Server

Wraps the FlashTalk inference pipeline as a WebSocket service for real-time
digital human video generation. Designed to run on 8-card Ascend 910B via torchrun.

Usage:
    torchrun --nproc_per_node=8 model_backends/flashtalk/flashtalk_ws_server.py \
        --ckpt_dir models/SoulX-FlashTalk-14B \
        --wav2vec_dir models/chinese-wav2vec2-base \
        --port 8765

Protocol:
    1. init (JSON)     -> init_ok (JSON)
    2. generate (bin)  -> video frames (bin)
    3. close (JSON)    -> close_ok (JSON)

See README or module docstring for full protocol details.
"""

import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import inspect
import json
import os
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from PIL import Image


def device_type() -> str:
    """Return accelerator name for torch.device strings ('npu' or 'cuda')."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronize() -> None:
    """Best-effort device synchronize for the active backend."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _patch_flash_talk_usp_device() -> None:
    """SoulX ``usp_device.get_device`` hard-codes NCCL + CUDA; Ascend needs HCCL + NPU.

    Must run before ``import flash_talk.inference`` so ``inference`` binds the patched function.
    """
    import flash_talk.src.distributed.usp_device as usp

    def get_device(ulysses_degree: int, ring_degree: int):
        if ulysses_degree > 1 or ring_degree > 1:
            from xfuser.core.distributed import (
                get_world_group,
                init_distributed_environment,
                initialize_model_parallel,
            )

            if hasattr(torch, "npu") and torch.npu.is_available():
                backend = "hccl"
                accel = "npu"
            elif torch.cuda.is_available():
                backend = "nccl"
                accel = "cuda"
            else:
                raise RuntimeError(
                    "FlashTalk parallel path needs torch.distributed with HCCL (torch_npu on Ascend) "
                    "or NCCL (CUDA PyTorch). This interpreter build has neither."
                )

            if not dist.is_initialized():
                dist.init_process_group(backend, timeout=datetime.timedelta(hours=24 * 7))
            init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_degree,
                ulysses_degree=ulysses_degree,
            )

            rank = get_world_group().rank
            device = torch.device(f"{accel}:{rank}")
            if accel == "npu":
                torch.npu.set_device(rank)
            else:
                torch.cuda.set_device(rank)

            logger.info("rank={} device={} (backend={})", rank, device, backend)
        else:
            lr = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
            if hasattr(torch, "npu") and torch.npu.is_available():
                device = torch.device(f"npu:{lr}")
                torch.npu.set_device(lr)
            elif torch.cuda.is_available():
                device = torch.device(f"cuda:{lr}")
                torch.cuda.set_device(lr)
            else:
                device = torch.device("cpu")
        return device

    usp.get_device = get_device


_patch_flash_talk_usp_device()

from flash_talk.inference import (
    get_pipeline,
    get_base_data,
    get_audio_embedding,
    run_pipeline,
    infer_params,
)


def _get_pipeline_compat(**kwargs):
    """Call SoulX ``get_pipeline`` with only kwargs supported by the installed revision."""
    params = inspect.signature(get_pipeline).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return get_pipeline(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        logger.warning(
            "SoulX get_pipeline() does not accept {}; these options are ignored. "
            "Upgrade SoulX-FlashTalk or use a revision that supports quantization kwargs.",
            dropped,
        )
    return get_pipeline(**filtered)

# ---------------------------------------------------------------------------
# Constants from distributed env
# ---------------------------------------------------------------------------
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

# Command codes for inter-rank broadcast
CMD_INIT = 0
CMD_GENERATE = 1
CMD_SHUTDOWN = 2
CMD_NONE = -1

# Binary message magic headers
MAGIC_AUDIO = b"AUDI"
MAGIC_VIDEO = b"VIDX"

# ---------------------------------------------------------------------------
# Inference parameters (read once from infer_params)
# ---------------------------------------------------------------------------
FRAME_NUM = infer_params["frame_num"]
MOTION_FRAMES_NUM = infer_params["motion_frames_num"]
SLICE_LEN = FRAME_NUM - MOTION_FRAMES_NUM
SAMPLE_RATE = infer_params["sample_rate"]
TGT_FPS = infer_params["tgt_fps"]
# TGT_FPS = 20
CACHED_AUDIO_DURATION = infer_params["cached_audio_duration"]
HEIGHT = infer_params["height"]
WIDTH = infer_params["width"]

JPEG_QUALITY = min(100, max(1, int(os.environ.get("FLASHTALK_JPEG_QUALITY", "40"))))
JPEG_WORKERS = max(1, int(os.environ.get("FLASHTALK_JPEG_WORKERS", "1")))
_JPEG_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
PROGRESSIVE_SEND = os.environ.get("FLASHTALK_PROGRESSIVE_SEND", "0") == "1"
WARMUP_ON_STARTUP = os.environ.get("FLASHTALK_WARMUP", "0") == "1"
WARMUP_ON_INIT = os.environ.get("FLASHTALK_WARMUP_ON_INIT", "0") == "1"
WARMUP_REF_IMAGE = os.environ.get("FLASHTALK_WARMUP_REF_IMAGE", "").strip()
WARMUP_PROMPT = os.environ.get(
    "FLASHTALK_WARMUP_PROMPT",
    "A person is talking. Only the foreground characters are moving, the background remains static.",
)
WARMUP_SEED = int(os.environ.get("FLASHTALK_WARMUP_SEED", "9999"))
IDLE_PRELOAD_REFS = [
    item.strip()
    for item in os.environ.get("FLASHTALK_IDLE_PRELOAD_REFS", "").split(",")
    if item.strip()
]
IDLE_CACHE_CHUNKS = max(0, int(os.environ.get("FLASHTALK_IDLE_CACHE_CHUNKS", "4")))
IDLE_CACHE_LEVEL = max(
    0.0,
    float(os.environ.get("FLASHTALK_IDLE_CACHE_LEVEL", "480")) / 32768.0,
)
IDLE_CACHE_CROSSFADE_FRAMES = max(
    0, int(os.environ.get("FLASHTALK_IDLE_CACHE_CROSSFADE_FRAMES", "6"))
)
IDLE_CACHE_PLAYBACK = os.environ.get("FLASHTALK_IDLE_CACHE_PLAYBACK", "pingpong").lower()
IDLE_ENTER_CHUNKS = max(1, int(os.environ.get("FLASHTALK_IDLE_ENTER_CHUNKS", "2")))
IDLE_SILENCE_RMS = max(
    0.0, float(os.environ.get("FLASHTALK_IDLE_SILENCE_RMS", "0.004"))
)
IDLE_REFRESH_INTERVAL = max(
    1, int(os.environ.get("FLASHTALK_IDLE_REFRESH_INTERVAL", "3"))
)
IDLE_HOLD_MIN_CHUNKS = max(
    1, int(os.environ.get("FLASHTALK_IDLE_HOLD_MIN_CHUNKS", "1"))
)
IDLE_HOLD_MAX_CHUNKS = max(
    IDLE_HOLD_MIN_CHUNKS,
    int(os.environ.get("FLASHTALK_IDLE_HOLD_MAX_CHUNKS", "3")),
)
IDLE_MOUTH_LOCK = min(
    1.0, max(0.0, float(os.environ.get("FLASHTALK_IDLE_MOUTH_LOCK", "0.97")))
)
IDLE_MOUTH_TEMPORAL = min(
    1.0, max(0.0, float(os.environ.get("FLASHTALK_IDLE_MOUTH_TEMPORAL", "0.85")))
)
IDLE_EYE_LOCK = min(
    1.0, max(0.0, float(os.environ.get("FLASHTALK_IDLE_EYE_LOCK", "0.65")))
)
IDLE_EYE_TEMPORAL = min(
    1.0, max(0.0, float(os.environ.get("FLASHTALK_IDLE_EYE_TEMPORAL", "0.75")))
)
IDLE_RANDOM_SEED = int(os.environ.get("FLASHTALK_IDLE_RANDOM_SEED", "20260415"))
IDLE_CACHE_VERSION = int(os.environ.get("FLASHTALK_IDLE_CACHE_VERSION", "2"))
IDLE_CACHE_DIR = Path(
    os.environ.get(
        "FLASHTALK_IDLE_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "flashtalk_idle_cache"),
    )
).expanduser()

# Audio chunk size: slice_len frames worth of audio samples
AUDIO_CHUNK_SAMPLES = SLICE_LEN * SAMPLE_RATE // TGT_FPS  # 17920 for default params
AUDIO_CHUNK_BYTES = AUDIO_CHUNK_SAMPLES * 2  # int16 = 2 bytes per sample

# Circular buffer length in samples
CACHED_AUDIO_SAMPLES = SAMPLE_RATE * CACHED_AUDIO_DURATION  # 128000 for 8s @ 16kHz

# Audio embedding window indices (stream mode)
AUDIO_END_IDX = CACHED_AUDIO_DURATION * TGT_FPS  # 200
AUDIO_START_IDX = AUDIO_END_IDX - FRAME_NUM  # 167

# Broadcast device – HCCL (NPU) requires tensors on the NPU device.
# Set after pipeline init via _init_bcast_device().
_BCAST_DEVICE: torch.device = torch.device("cpu")
_PROCESS_START_TIME = time.time()
_CMD_SEQ = 0
_CMD_LAST_SEQ = 0
_AUDIO_EMBEDDING_SHAPE_CACHED: tuple[int, ...] | None = None
_IDLE_CACHE_MEMORY: dict[str, list[np.ndarray]] = {}


def _command_file_path() -> str:
    """Per-torchrun command file used to wake workers without blocking HCCL."""
    cmd_dir = os.environ.get("FLASHTALK_CMD_DIR", tempfile.gettempdir())
    port = os.environ.get("MASTER_PORT", "29500")
    return os.path.join(cmd_dir, f"flashtalk_cmd_{port}.json")


def _write_command(cmd: int) -> int:
    global _CMD_SEQ
    _CMD_SEQ += 1
    path = _command_file_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    payload = {"seq": _CMD_SEQ, "cmd": cmd}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)
    return cmd


def _wait_for_command() -> int:
    """Workers poll for the next command so idle sessions do not hit HCCL timeouts."""
    global _CMD_LAST_SEQ
    path = _command_file_path()
    while True:
        try:
            # Ignore stale command files left by an older torchrun.
            if os.path.getmtime(path) < _PROCESS_START_TIME - 1.0:
                time.sleep(0.05)
                continue
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            seq = int(payload.get("seq", 0))
            cmd = int(payload.get("cmd", CMD_NONE))
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            time.sleep(0.05)
            continue

        if seq > _CMD_LAST_SEQ and cmd != CMD_NONE:
            _CMD_LAST_SEQ = seq
            return cmd
        time.sleep(0.05)


def _encode_jpeg_frame(args: tuple[np.ndarray, list[int]]) -> bytes:
    """Encode one RGB frame to JPEG bytes."""
    frame_rgb, encode_params = args
    import cv2

    # cv2.imencode expects BGR and contiguous memory. The RGB->BGR view has a
    # negative stride, so make the copy explicit before entering OpenCV.
    bgr_frame = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    ok, buf = cv2.imencode(".jpg", bgr_frame, encode_params)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def encode_video_jpegs(video_np: np.ndarray) -> list[bytes]:
    """JPEG-compress frames, optionally in parallel."""
    global _JPEG_EXECUTOR
    import cv2

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    tasks = [(video_np[fi], encode_params) for fi in range(video_np.shape[0])]

    if JPEG_WORKERS <= 1 or len(tasks) <= 1:
        return [_encode_jpeg_frame(task) for task in tasks]

    if _JPEG_EXECUTOR is None:
        _JPEG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=JPEG_WORKERS,
            thread_name_prefix="flashtalk-jpeg",
        )
    return list(_JPEG_EXECUTOR.map(_encode_jpeg_frame, tasks))


def _init_bcast_device():
    """Call after get_pipeline() so device_type() is correct."""
    global _BCAST_DEVICE
    local_rank = int(os.environ.get("LOCAL_RANK", RANK))
    dt = device_type()  # "npu" or "cuda"
    _BCAST_DEVICE = torch.device(f"{dt}:{local_rank}")
    logger.info(f"[Rank {RANK}] broadcast device = {_BCAST_DEVICE}")


# ---------------------------------------------------------------------------
# Helper: broadcast a string from rank 0 to all ranks
# ---------------------------------------------------------------------------
def broadcast_string(s: str | None = None, src: int = 0) -> str:
    """Broadcast a UTF-8 string from src rank to all other ranks."""
    if RANK == src:
        data = s.encode("utf-8")
        length_tensor = torch.tensor([len(data)], dtype=torch.long, device=_BCAST_DEVICE)
    else:
        length_tensor = torch.zeros(1, dtype=torch.long, device=_BCAST_DEVICE)

    dist.broadcast(length_tensor, src=src)
    length = int(length_tensor.item())

    if RANK == src:
        data_tensor = torch.tensor(list(data), dtype=torch.uint8, device=_BCAST_DEVICE)
    else:
        data_tensor = torch.zeros(length, dtype=torch.uint8, device=_BCAST_DEVICE)

    dist.broadcast(data_tensor, src=src)

    if RANK != src:
        s = bytes(data_tensor.cpu().tolist()).decode("utf-8")
    return s


# ---------------------------------------------------------------------------
# Helper: broadcast a command code from rank 0
# ---------------------------------------------------------------------------
def broadcast_cmd(cmd: int) -> int:
    """Publish or receive a command without leaving workers blocked in HCCL."""
    if RANK == 0:
        return _write_command(cmd)
    return _wait_for_command()


# ---------------------------------------------------------------------------
# Helper: broadcast audio embedding tensor from rank 0
# ---------------------------------------------------------------------------
def broadcast_audio_embedding(embedding: torch.Tensor | None = None) -> torch.Tensor:
    """Broadcast audio embedding tensor from rank 0 to all ranks."""
    global _AUDIO_EMBEDDING_SHAPE_CACHED

    if _AUDIO_EMBEDDING_SHAPE_CACHED is None:
        # First broadcast the shape
        if RANK == 0:
            shape_list = list(embedding.shape)
            ndim_tensor = torch.tensor([len(shape_list)], dtype=torch.long, device=_BCAST_DEVICE)
        else:
            ndim_tensor = torch.zeros(1, dtype=torch.long, device=_BCAST_DEVICE)

        dist.broadcast(ndim_tensor, src=0)
        ndim = int(ndim_tensor.item())

        if RANK == 0:
            shape_tensor = torch.tensor(shape_list, dtype=torch.long, device=_BCAST_DEVICE)
        else:
            shape_tensor = torch.zeros(ndim, dtype=torch.long, device=_BCAST_DEVICE)

        dist.broadcast(shape_tensor, src=0)
        _AUDIO_EMBEDDING_SHAPE_CACHED = tuple(int(x) for x in shape_tensor.cpu().tolist())
    elif RANK == 0 and tuple(int(x) for x in embedding.shape) != _AUDIO_EMBEDDING_SHAPE_CACHED:
        raise ValueError(
            f"Audio embedding shape changed within a session: "
            f"{tuple(int(x) for x in embedding.shape)} != {_AUDIO_EMBEDDING_SHAPE_CACHED}"
        )

    shape = _AUDIO_EMBEDDING_SHAPE_CACHED

    # Broadcast the data – keep on _BCAST_DEVICE
    if RANK == 0:
        data_tensor = embedding.contiguous().to(device=_BCAST_DEVICE, dtype=torch.bfloat16)
    else:
        data_tensor = torch.zeros(shape, dtype=torch.bfloat16, device=_BCAST_DEVICE)

    dist.broadcast(data_tensor, src=0)
    return data_tensor.to(dtype=torch.float32)


def _reset_audio_embedding_shape_cache():
    global _AUDIO_EMBEDDING_SHAPE_CACHED
    _AUDIO_EMBEDDING_SHAPE_CACHED = None


def _append_audio_chunk(audio_buffer: np.ndarray, write_pos: int, chunk_audio: np.ndarray) -> int:
    """Append one chunk into the pre-allocated numpy circular buffer (O(chunk_len))."""
    buf_len = audio_buffer.shape[0]
    chunk_len = int(chunk_audio.shape[0])
    if chunk_len >= buf_len:
        audio_buffer[:] = chunk_audio[-buf_len:]
        return 0

    end = write_pos + chunk_len
    if end <= buf_len:
        audio_buffer[write_pos:end] = chunk_audio
    else:
        first = buf_len - write_pos
        audio_buffer[write_pos:] = chunk_audio[:first]
        audio_buffer[:end - buf_len] = chunk_audio[first:]
    return end % buf_len


def _linearize_audio_buffer(audio_buffer: np.ndarray, write_pos: int) -> np.ndarray:
    """Return audio samples ordered from oldest to newest."""
    if write_pos == 0:
        return audio_buffer.copy()
    return np.concatenate([audio_buffer[write_pos:], audio_buffer[:write_pos]])


def _chunk_rms(chunk_audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(chunk_audio, dtype=np.float32), dtype=np.float32)))


def _load_reference_frame(image_path: str) -> np.ndarray:
    """Resize + center-crop the avatar image to the exact video geometry."""
    with Image.open(image_path).convert("RGB") as img:
        scale = max(WIDTH / img.width, HEIGHT / img.height)
        resized_w = max(1, int(np.ceil(img.width * scale)))
        resized_h = max(1, int(np.ceil(img.height * scale)))
        resized = img.resize((resized_w, resized_h), resample=Image.BILINEAR)
        left = max(0, (resized_w - WIDTH) // 2)
        top = max(0, (resized_h - HEIGHT) // 2)
        cropped = resized.crop((left, top, left + WIDTH, top + HEIGHT))
        return np.asarray(cropped, dtype=np.uint8)


def _make_idle_cache_key(reference_frame: np.ndarray) -> str:
    """Build a stable cache key from the cropped avatar and runtime knobs."""
    payload = {
        "version": IDLE_CACHE_VERSION,
        "frame_num": FRAME_NUM,
        "motion_frames_num": MOTION_FRAMES_NUM,
        "height": HEIGHT,
        "width": WIDTH,
        "sample_rate": SAMPLE_RATE,
        "tgt_fps": TGT_FPS,
        "idle_cache_chunks": IDLE_CACHE_CHUNKS,
        "idle_cache_level": IDLE_CACHE_LEVEL,
        "idle_cache_playback": IDLE_CACHE_PLAYBACK,
        "idle_cache_crossfade_frames": IDLE_CACHE_CROSSFADE_FRAMES,
        "idle_enter_chunks": IDLE_ENTER_CHUNKS,
        "idle_silence_rms": IDLE_SILENCE_RMS,
        "idle_refresh_interval": IDLE_REFRESH_INTERVAL,
        "idle_hold_min_chunks": IDLE_HOLD_MIN_CHUNKS,
        "idle_hold_max_chunks": IDLE_HOLD_MAX_CHUNKS,
        "idle_mouth_lock": IDLE_MOUTH_LOCK,
        "idle_mouth_temporal": IDLE_MOUTH_TEMPORAL,
        "idle_eye_lock": IDLE_EYE_LOCK,
        "idle_eye_temporal": IDLE_EYE_TEMPORAL,
        "idle_random_seed": IDLE_RANDOM_SEED,
        "jpeg_quality": JPEG_QUALITY,
    }
    h = hashlib.sha256()
    h.update(reference_frame.tobytes())
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _idle_cache_path(cache_key: str) -> Path:
    return IDLE_CACHE_DIR / f"{cache_key}.npz"


def _load_idle_cache_frames(cache_key: str) -> list[np.ndarray] | None:
    cached = _IDLE_CACHE_MEMORY.get(cache_key)
    if cached is not None:
        return [frames.copy() for frames in cached]

    cache_path = _idle_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            frames = data["frames"]
        if frames.ndim != 5:
            raise ValueError(f"unexpected idle cache shape: {frames.shape}")
        cached_frames = [np.asarray(chunk, dtype=np.uint8) for chunk in frames]
        _IDLE_CACHE_MEMORY[cache_key] = [chunk.copy() for chunk in cached_frames]
        logger.info(
            "[Server] Loaded idle cache from disk: key={} chunks={} path={}",
            cache_key[:12],
            len(cached_frames),
            cache_path,
        )
        return [chunk.copy() for chunk in cached_frames]
    except Exception as exc:
        logger.warning(
            "[Server] Failed to load idle cache {}: {}",
            cache_path,
            exc,
        )
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


def _save_idle_cache_frames(cache_key: str, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    IDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(frames, axis=0).astype(np.uint8, copy=False)
    cache_path = _idle_cache_path(cache_key)
    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, frames=stacked)
    os.replace(tmp_path, cache_path)
    _IDLE_CACHE_MEMORY[cache_key] = [chunk.copy() for chunk in frames]
    logger.info(
        "[Server] Saved idle cache: key={} chunks={} path={}",
        cache_key[:12],
        len(frames),
        cache_path,
    )


def _build_idle_mouth_mask(frame_height: int, frame_width: int) -> np.ndarray:
    """Create a soft ellipse around the mouth/jaw area for idle mouth locking."""
    ys = np.linspace(0.0, 1.0, frame_height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, frame_width, dtype=np.float32)[None, :]

    center_x = 0.50
    center_y = 0.44
    radius_x = 0.14
    radius_y = 0.10
    soft_band = 0.28

    dist = np.sqrt(
        np.square((xs - center_x) / max(radius_x, 1e-6))
        + np.square((ys - center_y) / max(radius_y, 1e-6))
    )
    base_mask = np.clip((1.0 - dist) / soft_band, 0.0, 1.0)

    lower_bias = np.clip((ys - 0.34) / 0.18, 0.0, 1.0)
    mask = np.clip(base_mask * (0.65 + 0.35 * lower_bias), 0.0, 1.0)
    return mask.astype(np.float32)


def _build_idle_eye_mask(frame_height: int, frame_width: int) -> np.ndarray:
    """Create a soft mask around both eyes to calm unnatural idle blinking."""
    ys = np.linspace(0.0, 1.0, frame_height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, frame_width, dtype=np.float32)[None, :]

    def _eye(center_x: float, center_y: float) -> np.ndarray:
        radius_x = 0.095
        radius_y = 0.050
        soft_band = 0.40
        dist = np.sqrt(
            np.square((xs - center_x) / max(radius_x, 1e-6))
            + np.square((ys - center_y) / max(radius_y, 1e-6))
        )
        return np.clip((1.0 - dist) / soft_band, 0.0, 1.0)

    left_eye = _eye(0.39, 0.28)
    right_eye = _eye(0.61, 0.28)
    lid_bias = np.clip(1.0 - np.abs(ys - 0.28) / 0.12, 0.0, 1.0)
    brow_falloff = np.clip((0.36 - ys) / 0.16, 0.0, 1.0)
    mask = np.maximum(left_eye, right_eye)
    mask = np.clip(mask * (0.70 + 0.30 * lid_bias) * (0.80 + 0.20 * brow_falloff), 0.0, 1.0)
    return mask.astype(np.float32)


def _prepare_audio_embedding_for_chunk(pipeline, audio_array: np.ndarray) -> torch.Tensor:
    return get_audio_embedding(pipeline, audio_array, AUDIO_START_IDX, AUDIO_END_IDX)


def _run_pipeline_for_audio_embedding(pipeline, audio_embedding: torch.Tensor) -> torch.Tensor:
    broadcast_cmd(CMD_GENERATE)
    broadcast_audio_embedding(audio_embedding)
    synchronize()
    video = run_pipeline(pipeline, audio_embedding)
    synchronize()
    return video[MOTION_FRAMES_NUM:]


def _render_video_frames_for_audio_embedding(
    pipeline, audio_embedding: torch.Tensor
) -> np.ndarray:
    """Run one chunk and return uint8 RGB frames on CPU."""
    video = _run_pipeline_for_audio_embedding(pipeline, audio_embedding)
    return video.cpu().numpy().astype(np.uint8)


def _render_video_frames_for_audio_embedding_local(
    pipeline, audio_embedding: torch.Tensor
) -> np.ndarray:
    """Rank-local render path used for startup warmup/preload before worker loop starts."""
    video = run_pipeline(pipeline, audio_embedding)
    synchronize()
    return video[MOTION_FRAMES_NUM:].cpu().numpy().astype(np.uint8)


def _build_idle_audio_chunk(chunk_seed: int) -> np.ndarray:
    """Synthesize a low-energy chunk that keeps the avatar subtly alive."""
    rng = np.random.default_rng(chunk_seed)
    t = np.arange(AUDIO_CHUNK_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    base = (
        0.55 * np.sin(2 * np.pi * (0.18 + 0.03 * (chunk_seed % 3)) * t + rng.uniform(0, 2 * np.pi))
        + 0.25 * np.sin(2 * np.pi * (0.41 + 0.05 * (chunk_seed % 5)) * t + rng.uniform(0, 2 * np.pi))
    )
    noise = rng.standard_normal(AUDIO_CHUNK_SAMPLES).astype(np.float32)
    smooth_noise = np.convolve(noise, np.ones(96, dtype=np.float32) / 96.0, mode="same")
    pulse_center = rng.uniform(0.15, 0.85)
    pulse_width = rng.uniform(0.06, 0.12)
    pulse = np.exp(-0.5 * np.square((t / max(t[-1], 1e-6) - pulse_center) / pulse_width))
    envelope = 0.72 + 0.18 * np.sin(2 * np.pi * 0.09 * t + rng.uniform(0, 2 * np.pi))
    chunk = (base + 0.35 * smooth_noise + 0.14 * pulse).astype(np.float32)
    chunk *= envelope.astype(np.float32)
    rms = _chunk_rms(chunk)
    if rms > 1e-6 and IDLE_CACHE_LEVEL > 0:
        chunk *= IDLE_CACHE_LEVEL / rms
    else:
        chunk.fill(0.0)
    return np.clip(chunk, -1.0, 1.0)


def _generate_idle_cache_frames(pipeline) -> list[np.ndarray]:
    """Pre-render a short idle bank so silence does not look like frozen speech."""
    if IDLE_CACHE_CHUNKS <= 0:
        return []

    idle_frames: list[np.ndarray] = []
    idle_audio_buffer = np.zeros(CACHED_AUDIO_SAMPLES, dtype=np.float32)
    idle_write_pos = 0

    logger.info(
        "[Server] Building idle cache: chunks={} level={:.5f} playback={}",
        IDLE_CACHE_CHUNKS,
        IDLE_CACHE_LEVEL,
        IDLE_CACHE_PLAYBACK,
    )

    for idle_idx in range(IDLE_CACHE_CHUNKS):
        idle_chunk_audio = _build_idle_audio_chunk(IDLE_RANDOM_SEED + idle_idx * 17)
        idle_write_pos = _append_audio_chunk(
            idle_audio_buffer, idle_write_pos, idle_chunk_audio
        )
        idle_audio_array = _linearize_audio_buffer(idle_audio_buffer, idle_write_pos)
        idle_embedding = _prepare_audio_embedding_for_chunk(pipeline, idle_audio_array)
        idle_frames.append(_render_video_frames_for_audio_embedding(pipeline, idle_embedding))

    logger.info("[Server] Idle cache ready with {} chunks.", len(idle_frames))
    return idle_frames


def _generate_idle_cache_frames_local(pipeline) -> list[np.ndarray]:
    """Local rank startup path for prebuilding idle cache before the WS server accepts traffic."""
    if IDLE_CACHE_CHUNKS <= 0:
        return []

    idle_frames: list[np.ndarray] = []
    idle_audio_buffer = np.zeros(CACHED_AUDIO_SAMPLES, dtype=np.float32)
    idle_write_pos = 0

    logger.info(
        "[Server] Prebuilding idle cache locally: chunks={} level={:.5f} playback={}",
        IDLE_CACHE_CHUNKS,
        IDLE_CACHE_LEVEL,
        IDLE_CACHE_PLAYBACK,
    )

    for idle_idx in range(IDLE_CACHE_CHUNKS):
        idle_chunk_audio = _build_idle_audio_chunk(IDLE_RANDOM_SEED + idle_idx * 17)
        idle_write_pos = _append_audio_chunk(
            idle_audio_buffer, idle_write_pos, idle_chunk_audio
        )
        idle_audio_array = _linearize_audio_buffer(idle_audio_buffer, idle_write_pos)
        idle_embedding = _prepare_audio_embedding_for_chunk(pipeline, idle_audio_array)
        idle_frames.append(_render_video_frames_for_audio_embedding_local(pipeline, idle_embedding))

    logger.info("[Server] Local idle cache ready with {} chunks.", len(idle_frames))
    return idle_frames


def _build_idle_refresh_audio_array(
    audio_buffer: np.ndarray,
    write_pos: int,
    refresh_seed: int,
) -> np.ndarray:
    """Build a temporary audio window for a live idle refresh without mutating session audio."""
    refresh_buffer = audio_buffer.copy()
    refresh_pos = _append_audio_chunk(
        refresh_buffer,
        write_pos,
        _build_idle_audio_chunk(refresh_seed),
    )
    return _linearize_audio_buffer(refresh_buffer, refresh_pos)


def _crossfade_frames(
    previous_frames: np.ndarray | None,
    next_frames: np.ndarray,
    crossfade_frames: int,
) -> np.ndarray:
    """Blend the beginning of the next chunk with the tail of the previous chunk."""
    if previous_frames is None or crossfade_frames <= 0:
        return next_frames

    blend_frames = min(
        crossfade_frames,
        previous_frames.shape[0],
        next_frames.shape[0],
    )
    if blend_frames <= 0:
        return next_frames

    blended = next_frames.copy()
    alpha = np.linspace(0.0, 1.0, blend_frames + 2, dtype=np.float32)[1:-1]
    alpha = alpha[:, None, None, None]
    prev_tail = previous_frames[-blend_frames:].astype(np.float32)
    next_head = next_frames[:blend_frames].astype(np.float32)
    blended[:blend_frames] = np.clip(
        prev_tail * (1.0 - alpha) + next_head * alpha,
        0.0,
        255.0,
    ).astype(np.uint8)
    return blended


def _apply_idle_region_constraints(
    video_frames: np.ndarray,
    reference_frame: np.ndarray | None,
    region_mask: np.ndarray | None,
    previous_idle_frames: np.ndarray | None,
    lock_strength: float,
    temporal_strength: float,
) -> np.ndarray:
    """Blend a facial region back toward the reference frame and smooth it over time."""
    if (
        reference_frame is None
        or region_mask is None
        or lock_strength <= 0.0
        or video_frames.size == 0
    ):
        return video_frames

    adjusted = video_frames.astype(np.float32)
    ref = reference_frame.astype(np.float32)[None, ...]
    mask = (region_mask * lock_strength)[None, ..., None]
    adjusted = adjusted * (1.0 - mask) + ref * mask

    if previous_idle_frames is not None and temporal_strength > 0.0:
        temporal_mask = (region_mask * temporal_strength)[None, ..., None]
        adjusted = adjusted * (1.0 - temporal_mask) + previous_idle_frames.astype(np.float32) * temporal_mask

    return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)


def _advance_idle_cache_cursor(
    current_index: int,
    current_direction: int,
    cache_size: int,
) -> tuple[int, int]:
    if cache_size <= 1:
        return 0, 1
    if IDLE_CACHE_PLAYBACK == "loop":
        return (current_index + 1) % cache_size, 1
    if IDLE_CACHE_PLAYBACK == "random":
        next_index = (current_index + np.random.randint(1, cache_size)) % cache_size
        return next_index, current_direction

    next_index = current_index + current_direction
    next_direction = current_direction
    if next_index >= cache_size:
        next_direction = -1
        next_index = cache_size - 2
    elif next_index < 0:
        next_direction = 1
        next_index = 1
    return next_index, next_direction


def _sample_idle_hold_chunks(idle_rng: np.random.Generator, cache_size: int) -> int:
    """Keep the same idle state for a short, variable duration."""
    if cache_size <= 1:
        return 1
    return int(idle_rng.integers(IDLE_HOLD_MIN_CHUNKS, IDLE_HOLD_MAX_CHUNKS + 1))


# Whether to use streaming VAE decode + overlapped JPEG encoding
STREAM_DECODE = os.environ.get("FLASHTALK_STREAM_DECODE", "0") == "1"
# Whether to overlap JPEG encoding with motion-frame VAE encode
DEFERRED_MOTION = os.environ.get("FLASHTALK_DEFERRED_MOTION", "1") == "1"


def _run_pipeline_stream_for_audio_embedding(
    pipeline, audio_embedding: torch.Tensor
):
    """Streaming variant: broadcasts, then yields decoded frame batches.

    Each yield is a numpy uint8 array of shape ``[n, H, W, C]`` where *n* is
    1 or 4 (VAE temporal stride).  After the generator is exhausted,
    ``pipeline.latent_motion_frames`` has been updated.
    """
    broadcast_cmd(CMD_GENERATE)
    broadcast_audio_embedding(audio_embedding)
    synchronize()

    frame_idx = 0
    for frame_batch in run_pipeline_stream(pipeline, audio_embedding):
        # frame_batch: [n, H, W, C] float tensor on device
        n = frame_batch.shape[0]
        # Skip motion frames (first MOTION_FRAMES_NUM frames)
        if frame_idx + n <= MOTION_FRAMES_NUM:
            frame_idx += n
            continue
        skip = max(0, MOTION_FRAMES_NUM - frame_idx)
        frame_batch = frame_batch[skip:]
        frame_idx += n
        # Transfer to CPU and convert to uint8
        yield frame_batch.cpu().to(torch.uint8).numpy()

    synchronize()


def _submit_jpeg_futures(video_np: np.ndarray) -> list:
    """Submit JPEG encode jobs without waiting for results."""
    global _JPEG_EXECUTOR
    import cv2

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    tasks = [(video_np[fi], encode_params) for fi in range(video_np.shape[0])]

    if JPEG_WORKERS <= 1 or len(tasks) <= 1:
        # No parallelism: encode immediately, return bytes directly
        return [_encode_jpeg_frame(task) for task in tasks]

    if _JPEG_EXECUTOR is None:
        _JPEG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=JPEG_WORKERS,
            thread_name_prefix="flashtalk-jpeg",
        )
    return [_JPEG_EXECUTOR.submit(_encode_jpeg_frame, task) for task in tasks]


def _collect_jpeg_futures(futures: list) -> list[bytes]:
    """Collect results from _submit_jpeg_futures."""
    if not futures:
        return []
    if isinstance(futures[0], bytes):
        return futures  # already resolved
    return [f.result() for f in futures]


def _run_pipeline_deferred_for_audio_embedding(
    pipeline, audio_embedding: torch.Tensor,
) -> tuple[torch.Tensor, object]:
    """Like _run_pipeline_for_audio_embedding but defers motion-frame encode.

    Returns ``(video_frames, cond_frame)`` — caller must call
    ``pipeline.finalize_motion_frames(cond_frame)`` after JPEG encoding.
    """
    broadcast_cmd(CMD_GENERATE)
    broadcast_audio_embedding(audio_embedding)
    synchronize()
    video, cond_frame = run_pipeline_deferred(pipeline, audio_embedding)
    synchronize()
    return video[MOTION_FRAMES_NUM:], cond_frame


def _run_stream_and_encode(
    pipeline, audio_embedding: torch.Tensor,
) -> list[bytes]:
    """Stream VAE decode and overlap with JPEG encoding.

    Returns the same ``list[bytes]`` as :func:`encode_video_jpegs` so the
    caller's send path is unchanged.
    """
    global _JPEG_EXECUTOR
    import cv2

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    # We'll submit JPEG encode jobs to the thread pool as frames arrive from
    # the streaming VAE decoder, then collect results at the end.
    futures: list[concurrent.futures.Future] = []

    if JPEG_WORKERS > 1 and _JPEG_EXECUTOR is None:
        _JPEG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=JPEG_WORKERS,
            thread_name_prefix="flashtalk-jpeg",
        )

    for frame_batch_np in _run_pipeline_stream_for_audio_embedding(
        pipeline, audio_embedding
    ):
        # frame_batch_np: [n, H, W, C] uint8 numpy
        for fi in range(frame_batch_np.shape[0]):
            frame_rgb = frame_batch_np[fi]
            if JPEG_WORKERS > 1:
                futures.append(
                    _JPEG_EXECUTOR.submit(
                        _encode_jpeg_frame, (frame_rgb, encode_params)
                    )
                )
            else:
                futures.append(
                    _encode_jpeg_frame((frame_rgb, encode_params))
                )

    # Collect results (most should be done already thanks to overlap)
    if JPEG_WORKERS > 1:
        jpeg_parts = [f.result() for f in futures]
    else:
        jpeg_parts = futures  # already bytes
    return jpeg_parts


def _send_video_message(websocket, jpeg_parts: list[bytes]) -> tuple[int, int]:
    """Send encoded JPEG frames using the existing VIDX wire format."""
    n_frames = len(jpeg_parts)

    if PROGRESSIVE_SEND:
        total_bytes = 0
        for jpeg_bytes in jpeg_parts:
            payload = (
                MAGIC_VIDEO
                + struct.pack("<I", 1)
                + struct.pack("<I", len(jpeg_bytes))
                + jpeg_bytes
            )
            websocket.send(payload)
            total_bytes += len(payload)
        return n_frames, total_bytes

    # Protocol: VIDX + uint32(n_frames) + [uint32(jpeg_len) + jpeg_bytes] * n_frames
    header = MAGIC_VIDEO + struct.pack("<I", n_frames)
    parts = [header]
    total_bytes = 8
    for jpeg_bytes in jpeg_parts:
        parts.append(struct.pack("<I", len(jpeg_bytes)))
        parts.append(jpeg_bytes)
        total_bytes += 4 + len(jpeg_bytes)
    websocket.send(b"".join(parts))
    return n_frames, total_bytes


# ---------------------------------------------------------------------------
# Worker loop for non-rank-0 processes
# ---------------------------------------------------------------------------
def worker_loop(pipeline):
    """Non-rank-0 processes: wait for broadcast commands and execute them."""
    logger.info(f"[Rank {RANK}] Worker loop started, waiting for commands...")

    while True:
        cmd = broadcast_cmd(-1)  # value ignored on non-src ranks

        if cmd == CMD_INIT:
            # Receive init parameters
            image_path = broadcast_string()
            prompt = broadcast_string()
            seed_tensor = torch.zeros(1, dtype=torch.long, device=_BCAST_DEVICE)
            dist.broadcast(seed_tensor, src=0)
            seed = int(seed_tensor.item())

            logger.info(f"[Rank {RANK}] Executing get_base_data (seed={seed})")
            get_base_data(
                pipeline,
                input_prompt=prompt,
                cond_image=image_path,
                base_seed=seed,
            )
            _reset_audio_embedding_shape_cache()
            logger.info(f"[Rank {RANK}] get_base_data done")

        elif cmd == CMD_GENERATE:
            # Receive audio embedding from rank 0
            audio_embedding = broadcast_audio_embedding()

            synchronize()
            video = run_pipeline(pipeline, audio_embedding)
            synchronize()
            # Workers discard the video output; only rank 0 sends it back.

        elif cmd == CMD_SHUTDOWN:
            logger.info(f"[Rank {RANK}] Received shutdown command, exiting.")
            break
        else:
            logger.warning(f"[Rank {RANK}] Unknown command: {cmd}")

    # Cleanup
    if WORLD_SIZE > 1:
        dist.barrier()


def _prepare_pipeline_state(
    pipeline,
    image_path: str,
    prompt: str,
    seed: int,
) -> None:
    get_base_data(
        pipeline,
        input_prompt=prompt,
        cond_image=image_path,
        base_seed=seed,
    )
    _reset_audio_embedding_shape_cache()


def _run_startup_warmup(pipeline, image_path: str, prompt: str, seed: int) -> None:
    """Warm the runtime once, outside of request-time init."""
    _prepare_pipeline_state(pipeline, image_path, prompt, seed)
    warmup_buffer = np.zeros(CACHED_AUDIO_SAMPLES, dtype=np.float32)
    _append_audio_chunk(
        warmup_buffer,
        0,
        np.zeros(AUDIO_CHUNK_SAMPLES, dtype=np.float32),
    )
    warmup_embedding = _prepare_audio_embedding_for_chunk(pipeline, warmup_buffer)
    run_pipeline(pipeline, warmup_embedding)
    synchronize()
    _reset_audio_embedding_shape_cache()
    logger.info("[Startup] Warmup chunk complete for {}", image_path)


def _run_session_warmup(pipeline) -> None:
    """Request-time warmup that uses the existing rank-0 broadcast path."""
    warmup_buffer = np.zeros(CACHED_AUDIO_SAMPLES, dtype=np.float32)
    _append_audio_chunk(
        warmup_buffer,
        0,
        np.zeros(AUDIO_CHUNK_SAMPLES, dtype=np.float32),
    )
    warmup_embedding = _prepare_audio_embedding_for_chunk(
        pipeline, warmup_buffer
    )
    _run_pipeline_for_audio_embedding(pipeline, warmup_embedding)
    logger.info("Warmup chunk complete")


def _preload_idle_cache_for_ref(
    pipeline,
    image_path: str,
    prompt: str,
    seed: int,
) -> None:
    reference_frame = _load_reference_frame(image_path)
    cache_key = _make_idle_cache_key(reference_frame)
    cached = _load_idle_cache_frames(cache_key)
    if cached is not None:
        logger.info(
            "[Startup] Idle cache already available for {} (key={})",
            image_path,
            cache_key[:12],
        )
        return

    _prepare_pipeline_state(pipeline, image_path, prompt, seed)
    idle_frames = _generate_idle_cache_frames_local(pipeline)
    if RANK == 0 and idle_frames:
        _save_idle_cache_frames(cache_key, idle_frames)
    _prepare_pipeline_state(pipeline, image_path, prompt, seed)
    logger.info(
        "[Startup] Prebuilt idle cache for {} (key={})",
        image_path,
        cache_key[:12],
    )


# ---------------------------------------------------------------------------
# WebSocket server (rank 0 only) — fully synchronous to avoid HCCL/thread issues
# ---------------------------------------------------------------------------
def run_server(pipeline, host: str, port: int):
    """Rank 0: synchronous WebSocket server — all distributed ops in main thread."""
    from websockets.sync.server import serve as ws_serve

    # Session state
    session_active = False
    audio_buffer = None
    audio_write_pos = 0
    chunk_idx = 0
    temp_image_path = None
    idle_cache_frames: list[np.ndarray] = []
    idle_cache_index = 0
    idle_cache_direction = 1
    idle_hold_remaining = 0
    using_idle_cache = False
    silence_chunk_run = 0
    idle_refresh_counter = 0
    idle_refresh_generation = 0
    idle_reference_frame: np.ndarray | None = None
    idle_mouth_mask: np.ndarray | None = None
    idle_eye_mask: np.ndarray | None = None
    last_idle_locked_frames: np.ndarray | None = None
    last_output_frames: np.ndarray | None = None
    last_chunk_rms = 0.0
    idle_rng = np.random.default_rng(IDLE_RANDOM_SEED)

    def handler(websocket):
        nonlocal session_active, audio_buffer, audio_write_pos, chunk_idx, temp_image_path
        nonlocal idle_cache_frames, idle_cache_index, idle_cache_direction, idle_hold_remaining
        nonlocal using_idle_cache, silence_chunk_run, idle_refresh_counter
        nonlocal idle_refresh_generation, idle_reference_frame, idle_mouth_mask, idle_eye_mask
        nonlocal last_idle_locked_frames, last_output_frames, last_chunk_rms, idle_rng

        remote = websocket.request.headers.get("Host", "unknown")
        logger.info(f"[Server] New connection from {remote}")

        try:
            for message in websocket:
                # --- JSON text messages ---
                if isinstance(message, str):
                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                        continue

                    msg_type = msg.get("type", "")

                    # ---- INIT ----
                    if msg_type == "init":
                        if session_active:
                            logger.warning(
                                "[Server] Replacing existing active session with a new init request."
                            )
                            session_active = False
                            audio_buffer = None
                            audio_write_pos = 0
                            chunk_idx = 0
                            idle_cache_frames = []
                            idle_cache_index = 0
                            idle_cache_direction = 1
                            idle_hold_remaining = 0
                            using_idle_cache = False
                            silence_chunk_run = 0
                            idle_refresh_counter = 0
                            idle_refresh_generation = 0
                            idle_reference_frame = None
                            idle_mouth_mask = None
                            idle_eye_mask = None
                            last_idle_locked_frames = None
                            last_output_frames = None
                            last_chunk_rms = 0.0
                            _reset_audio_embedding_shape_cache()
                            _cleanup_temp_image(temp_image_path)
                            temp_image_path = None

                        ref_image_b64 = msg.get("ref_image", "")
                        prompt = msg.get(
                            "prompt",
                            "A person is talking. Only the foreground characters are moving, the background remains static.",
                        )
                        seed = int(msg.get("seed", 9999))

                        if not ref_image_b64:
                            websocket.send(json.dumps({
                                "type": "error", "message": "Missing 'ref_image' field.",
                            }))
                            continue

                        try:
                            image_data = base64.b64decode(ref_image_b64)
                        except Exception:
                            websocket.send(json.dumps({
                                "type": "error", "message": "Invalid base64 in 'ref_image'.",
                            }))
                            continue

                        fd, temp_image_path = tempfile.mkstemp(suffix=".png")
                        with os.fdopen(fd, "wb") as f:
                            f.write(image_data)

                        logger.info(
                            f"[Server] Init: prompt={prompt!r}, seed={seed}, "
                            f"image={len(image_data)} bytes -> {temp_image_path}"
                        )

                        try:
                            # All distributed calls in main thread — safe for HCCL
                            broadcast_cmd(CMD_INIT)
                            broadcast_string(temp_image_path)
                            broadcast_string(prompt)
                            seed_tensor = torch.tensor([seed], dtype=torch.long, device=_BCAST_DEVICE)
                            dist.broadcast(seed_tensor, src=0)
                            idle_reference_frame = _load_reference_frame(temp_image_path)
                            idle_mouth_mask = _build_idle_mouth_mask(HEIGHT, WIDTH)
                            idle_eye_mask = _build_idle_eye_mask(HEIGHT, WIDTH)
                            idle_cache_key = _make_idle_cache_key(idle_reference_frame)
                            _prepare_pipeline_state(
                                pipeline,
                                temp_image_path,
                                prompt,
                                seed,
                            )
                            idle_cache_frames = _load_idle_cache_frames(idle_cache_key) or []
                            if not idle_cache_frames:
                                # Cache miss: skip generation during init to avoid blocking
                                # the client. The client-side runner builds its own idle
                                # cache asynchronously after the session is ready.
                                logger.info(
                                    '[Server] Idle cache miss for key={}, skipping generation during init',
                                    idle_cache_key[:12],
                                )
                            if WARMUP_ON_INIT:
                                _run_session_warmup(pipeline)
                        except Exception as e:
                            logger.error(f"[Server] Init failed: {e}")
                            _cleanup_temp_image(temp_image_path)
                            temp_image_path = None
                            websocket.send(json.dumps({
                                "type": "error", "message": f"Init failed: {e}",
                            }))
                            continue

                        session_active = True
                        audio_buffer = np.zeros(CACHED_AUDIO_SAMPLES, dtype=np.float32)
                        audio_write_pos = 0
                        chunk_idx = 0
                        idle_rng = np.random.default_rng(IDLE_RANDOM_SEED + seed)
                        idle_cache_index = 0
                        idle_cache_direction = 1
                        idle_hold_remaining = _sample_idle_hold_chunks(
                            idle_rng, len(idle_cache_frames)
                        )
                        using_idle_cache = False
                        silence_chunk_run = 0
                        idle_refresh_counter = 0
                        idle_refresh_generation = 0
                        last_idle_locked_frames = None
                        last_output_frames = None
                        last_chunk_rms = 0.0

                        websocket.send(json.dumps({
                            "type": "init_ok",
                            "frame_num": FRAME_NUM,
                            "motion_frames_num": MOTION_FRAMES_NUM,
                            "slice_len": SLICE_LEN,
                            "fps": TGT_FPS,
                            "height": HEIGHT,
                            "width": WIDTH,
                        }))
                        logger.info("[Server] Init OK, session active.")

                    # ---- CLOSE ----
                    elif msg_type == "close":
                        logger.info("[Server] Client requested close.")
                        session_active = False
                        audio_buffer = None
                        audio_write_pos = 0
                        chunk_idx = 0
                        idle_cache_frames = []
                        idle_cache_index = 0
                        idle_cache_direction = 1
                        idle_hold_remaining = 0
                        using_idle_cache = False
                        silence_chunk_run = 0
                        idle_refresh_counter = 0
                        idle_refresh_generation = 0
                        idle_reference_frame = None
                        idle_mouth_mask = None
                        idle_eye_mask = None
                        last_idle_locked_frames = None
                        last_output_frames = None
                        last_chunk_rms = 0.0
                        _reset_audio_embedding_shape_cache()
                        _cleanup_temp_image(temp_image_path)
                        temp_image_path = None
                        websocket.send(json.dumps({"type": "close_ok"}))

                    else:
                        websocket.send(json.dumps({
                            "type": "error", "message": f"Unknown message type: {msg_type}",
                        }))

                # --- Binary messages (audio chunks) ---
                elif isinstance(message, bytes):
                    if not session_active:
                        websocket.send(json.dumps({
                            "type": "error", "message": "No active session. Send 'init' first.",
                        }))
                        continue

                    if len(message) < 4 or message[:4] != MAGIC_AUDIO:
                        websocket.send(json.dumps({
                            "type": "error", "message": "Binary message must start with 'AUDI' magic.",
                        }))
                        continue

                    pcm_bytes = message[4:]
                    if len(pcm_bytes) != AUDIO_CHUNK_BYTES:
                        websocket.send(json.dumps({
                            "type": "error",
                            "message": (
                                f"Expected {AUDIO_CHUNK_BYTES} bytes of int16 PCM "
                                f"({AUDIO_CHUNK_SAMPLES} samples), got {len(pcm_bytes)}."
                            ),
                        }))
                        continue

                    chunk_audio = (
                        np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    last_chunk_rms = _chunk_rms(chunk_audio)
                    silence_chunk_run = silence_chunk_run + 1 if last_chunk_rms <= IDLE_SILENCE_RMS else 0

                    audio_write_pos = _append_audio_chunk(audio_buffer, audio_write_pos, chunk_audio)
                    audio_array = _linearize_audio_buffer(audio_buffer, audio_write_pos)

                    t_start = time.time()
                    previous_mode = "idle" if using_idle_cache else "live"
                    current_mode = previous_mode
                    selected_idle_slot: int | None = None

                    try:
                        if idle_cache_frames and silence_chunk_run >= IDLE_ENTER_CHUNKS:
                            selected_idle_slot = idle_cache_index
                            should_refresh_idle = (
                                using_idle_cache and idle_refresh_counter >= IDLE_REFRESH_INTERVAL
                            )
                            if should_refresh_idle:
                                current_mode = "idle_refresh"
                                refresh_audio_array = _build_idle_refresh_audio_array(
                                    audio_buffer,
                                    audio_write_pos,
                                    IDLE_RANDOM_SEED + 1000 + idle_refresh_generation * 23,
                                )
                                audio_embedding = _prepare_audio_embedding_for_chunk(
                                    pipeline, refresh_audio_array
                                )
                                video_np = _render_video_frames_for_audio_embedding(
                                    pipeline, audio_embedding
                                )
                                idle_refresh_counter = 0
                                idle_refresh_generation += 1
                            else:
                                current_mode = "idle"
                                video_np = idle_cache_frames[selected_idle_slot].copy()
                                if using_idle_cache:
                                    idle_refresh_counter += 1
                            idle_hold_remaining -= 1
                            if idle_hold_remaining <= 0:
                                idle_cache_index, idle_cache_direction = _advance_idle_cache_cursor(
                                    idle_cache_index,
                                    idle_cache_direction,
                                    len(idle_cache_frames),
                                )
                                idle_hold_remaining = _sample_idle_hold_chunks(
                                    idle_rng, len(idle_cache_frames)
                                )
                        else:
                            current_mode = "live"
                            audio_embedding = _prepare_audio_embedding_for_chunk(
                                pipeline, audio_array
                            )
                            video_np = _render_video_frames_for_audio_embedding(
                                pipeline, audio_embedding
                            )
                            idle_hold_remaining = _sample_idle_hold_chunks(
                                idle_rng, len(idle_cache_frames)
                            )
                            idle_refresh_counter = 0
                    except Exception as e:
                        logger.error(f"[Server] Generate failed at chunk {chunk_idx}: {e}")
                        websocket.send(json.dumps({
                            "type": "error", "message": f"Generate failed: {e}",
                        }))
                        continue

                    t_infer = time.time()

                    if current_mode == "idle" or previous_mode == "idle":
                        video_np = _crossfade_frames(
                            last_output_frames,
                            video_np,
                            IDLE_CACHE_CROSSFADE_FRAMES,
                        )

                    if current_mode != "live":
                        video_np = _apply_idle_region_constraints(
                            video_np,
                            idle_reference_frame,
                            idle_mouth_mask,
                            last_idle_locked_frames,
                            IDLE_MOUTH_LOCK,
                            IDLE_MOUTH_TEMPORAL,
                        )
                        video_np = _apply_idle_region_constraints(
                            video_np,
                            idle_reference_frame,
                            idle_eye_mask,
                            last_idle_locked_frames,
                            IDLE_EYE_LOCK,
                            IDLE_EYE_TEMPORAL,
                        )
                        last_idle_locked_frames = video_np.copy()
                        if selected_idle_slot is not None:
                            idle_cache_frames[selected_idle_slot] = video_np.copy()
                    else:
                        last_idle_locked_frames = None

                    # JPEG encode (in thread pool to leverage multi-core) + send
                    jpeg_parts = encode_video_jpegs(video_np)
                    t_encode = time.time()
                    n_frames, total_bytes = _send_video_message(websocket, jpeg_parts)
                    t_send = time.time()
                    last_output_frames = video_np
                    using_idle_cache = current_mode != "live"
                    logger.info(
                        f"[Server] chunk-{chunk_idx}: {n_frames}f, "
                        f"infer={t_infer - t_start:.2f}s "
                        f"encode={t_encode - t_infer:.2f}s "
                        f"send={t_send - t_encode:.2f}s "
                        f"total={t_send - t_start:.2f}s "
                        f"mode={current_mode} "
                        f"rms={last_chunk_rms:.5f} "
                        f"silent_run={silence_chunk_run} "
                        f"size={total_bytes // 1024}KB "
                        f"jpeg_q={JPEG_QUALITY} jpeg_workers={JPEG_WORKERS}"
                    )
                    chunk_idx += 1

        except Exception as e:
            logger.warning(f"[Server] Connection error: {e}")
        finally:
            if session_active:
                logger.info("[Server] Client disconnected, cleaning up session.")
                session_active = False
                audio_buffer = None
                audio_write_pos = 0
                chunk_idx = 0
                idle_cache_frames = []
                idle_cache_index = 0
                idle_cache_direction = 1
                idle_hold_remaining = 0
                using_idle_cache = False
                silence_chunk_run = 0
                idle_refresh_counter = 0
                idle_refresh_generation = 0
                idle_reference_frame = None
                idle_mouth_mask = None
                idle_eye_mask = None
                last_idle_locked_frames = None
                last_output_frames = None
                last_chunk_rms = 0.0
                _reset_audio_embedding_shape_cache()
                _cleanup_temp_image(temp_image_path)
                temp_image_path = None

    logger.info(f"[Server] Rank 0 WebSocket server starting on {host}:{port}")
    with ws_serve(handler, host, port, max_size=50 * 1024 * 1024) as server:
        server.serve_forever()


def _cleanup_temp_image(path: str | None):
    """Remove temporary image file if it exists."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FlashTalk WebSocket Server")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="models/SoulX-FlashTalk-14B",
        help="FlashTalk model checkpoint directory",
    )
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default="models/chinese-wav2vec2-base",
        help="wav2vec checkpoint directory",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="WebSocket server bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload for low VRAM usage",
    )
    parser.add_argument(
        "--t5_quant",
        type=str,
        default=os.environ.get("FLASHTALK_T5_QUANT", "").strip() or None,
        choices=["int8", "fp8"],
        help="Optional T5 quantization mode. Defaults to FLASHTALK_T5_QUANT when set.",
    )
    parser.add_argument(
        "--t5_quant_dir",
        type=str,
        default=os.environ.get("FLASHTALK_T5_QUANT_DIR", "").strip() or None,
        help="Directory containing t5_<quant>.safetensors and t5_map_<quant>.json. Defaults to ckpt_dir.",
    )
    parser.add_argument(
        "--wan_quant",
        type=str,
        default=os.environ.get("FLASHTALK_WAN_QUANT", "").strip() or None,
        choices=["int8", "fp8"],
        help="Experimental WanModel weight-only quantization mode.",
    )
    parser.add_argument(
        "--wan_quant_include",
        type=str,
        default=os.environ.get("FLASHTALK_WAN_QUANT_INCLUDE", "").strip() or None,
        help="Comma-separated allowlist for WanModel submodule names.",
    )
    parser.add_argument(
        "--wan_quant_exclude",
        type=str,
        default=os.environ.get("FLASHTALK_WAN_QUANT_EXCLUDE", "").strip() or None,
        help="Comma-separated denylist for WanModel submodule names.",
    )
    args = parser.parse_args()
    if args.t5_quant is not None and args.t5_quant_dir is None:
        args.t5_quant_dir = args.ckpt_dir

    logger.info(
        f"[Rank {RANK}/{WORLD_SIZE}] Loading FlashTalk pipeline "
        f"(ckpt={args.ckpt_dir}, wav2vec={args.wav2vec_dir}, "
        f"t5_quant={args.t5_quant}, wan_quant={args.wan_quant})"
    )
    logger.info(
        f"[Rank {RANK}] Params: frame_num={FRAME_NUM}, motion_frames_num={MOTION_FRAMES_NUM}, "
        f"slice_len={SLICE_LEN}, resolution={HEIGHT}x{WIDTH}, fps={TGT_FPS}"
    )

    # All ranks: load model (this initializes torch.distributed internally)
    pipeline = _get_pipeline_compat(
        world_size=WORLD_SIZE,
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        cpu_offload=args.cpu_offload,
        t5_quant=args.t5_quant,
        t5_quant_dir=args.t5_quant_dir,
        wan_quant=args.wan_quant,
        wan_quant_include=args.wan_quant_include,
        wan_quant_exclude=args.wan_quant_exclude,
    )

    logger.info(f"[Rank {RANK}] Pipeline loaded successfully.")
    _init_bcast_device()

    startup_refs = []
    if WARMUP_REF_IMAGE:
        startup_refs.append(WARMUP_REF_IMAGE)
    for ref in IDLE_PRELOAD_REFS:
        if ref not in startup_refs:
            startup_refs.append(ref)

    if startup_refs:
        warmup_target = WARMUP_REF_IMAGE or startup_refs[0]
        for ref in startup_refs:
            ref_path = Path(ref).expanduser()
            if not ref_path.is_absolute():
                ref_path = (Path.cwd() / ref_path).resolve()
            if not ref_path.exists():
                logger.warning("[Startup] Skip missing preload ref: {}", ref_path)
                continue
            _preload_idle_cache_for_ref(
                pipeline,
                str(ref_path),
                WARMUP_PROMPT,
                WARMUP_SEED,
            )
            if WARMUP_ON_STARTUP and ref == warmup_target:
                _run_startup_warmup(
                    pipeline,
                    str(ref_path),
                    WARMUP_PROMPT,
                    WARMUP_SEED,
                )
        if WORLD_SIZE > 1:
            dist.barrier()
    elif WARMUP_ON_STARTUP:
        logger.warning(
            "[Startup] FLASHTALK_WARMUP=1 but no FLASHTALK_WARMUP_REF_IMAGE/FLASHTALK_IDLE_PRELOAD_REFS configured; skipping warmup."
        )

    if RANK == 0:
        # Rank 0: run the synchronous WebSocket server
        try:
            run_server(pipeline, args.host, args.port)
        except KeyboardInterrupt:
            logger.info("[Server] Interrupted, shutting down...")
        finally:
            # Tell all workers to exit
            if WORLD_SIZE > 1:
                broadcast_cmd(CMD_SHUTDOWN)
                dist.barrier()
                dist.destroy_process_group()
    else:
        # Other ranks: enter worker loop
        worker_loop(pipeline)
        if WORLD_SIZE > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
