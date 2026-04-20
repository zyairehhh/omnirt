"""SDXL pipeline implementation backed by Diffusers."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest
from omnirt.models.sdxl.components import DEFAULT_SDXL_MODEL_SOURCE
from omnirt.schedulers import build_scheduler


@register_model(
    id="sdxl-base-1.0",
    task="text2image",
    default_backend="auto",
    resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
)
class SDXLPipeline(BasePipeline):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline = None
        self._pipeline_source = None
        self._pipeline_dtype = None
        self._last_seed = None

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        prompt = req.inputs.get("prompt")
        if not prompt:
            raise ValueError("text2image requires inputs.prompt")
        return {
            "prompt": prompt,
            "negative_prompt": req.inputs.get("negative_prompt"),
            "model_source": req.config.get("model_path", DEFAULT_SDXL_MODEL_SOURCE),
            "scheduler": req.config.get("scheduler", "euler-discrete"),
            "height": int(req.config.get("height", 1024)),
            "width": int(req.config.get("width", 1024)),
            "num_images_per_prompt": int(req.config.get("num_images_per_prompt", 1)),
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Dict[str, Any]:
        steps = int(req.config.get("num_inference_steps", 30))
        seed = req.config.get("seed")
        guidance_scale = float(req.config.get("guidance_scale", 7.5))
        torch_dtype = self._resolve_torch_dtype(req.config.get("dtype"))
        generator = self._build_generator(seed)
        pipeline = self._load_pipeline(
            source=conditions["model_source"],
            torch_dtype=torch_dtype,
            scheduler_name=conditions["scheduler"],
            config=req.config,
        )
        self._last_seed = seed
        return {
            "steps": steps,
            "seed": seed,
            "generator": generator,
            "guidance_scale": guidance_scale,
            "conditions": conditions,
            "pipeline": pipeline,
        }

    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        pipeline = latents["pipeline"]
        result = pipeline(
            prompt=conditions["prompt"],
            negative_prompt=conditions.get("negative_prompt"),
            num_inference_steps=latents["steps"],
            guidance_scale=latents["guidance_scale"],
            generator=latents["generator"],
            height=conditions["height"],
            width=conditions["width"],
            num_images_per_prompt=conditions["num_images_per_prompt"],
            output_type="pil",
        )
        return {
            "images": list(result.images),
            "seed": latents["seed"],
            "generation_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def decode(self, latents: Any) -> Any:
        return latents["images"]

    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        output_dir = self.resolve_output_dir(req)
        artifacts: List[Artifact] = []
        for index, image in enumerate(raw):
            seed_part = self._last_seed if self._last_seed is not None else "random"
            file_path = output_dir / f"{req.model}-{seed_part}-{index}.png"
            image.save(file_path)
            artifacts.append(
                Artifact(
                    kind="image",
                    path=str(file_path),
                    mime="image/png",
                    width=image.width,
                    height=image.height,
                )
            )
        return artifacts

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError("PyTorch is required to run the SDXL pipeline.") from exc
        return torch

    def _resolve_torch_dtype(self, dtype_name: Optional[str]):
        torch = self._torch()
        mapping = {
            None: torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        try:
            return mapping[dtype_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {dtype_name}") from exc

    def _build_generator(self, seed: Optional[int]):
        if seed is None:
            return None
        torch = self._torch()
        for device_name in (self.runtime.device_name, "cpu"):
            try:
                return torch.Generator(device=device_name).manual_seed(int(seed))
            except Exception:
                continue
        return torch.Generator().manual_seed(int(seed))

    def _diffusers_pipeline_cls(self):
        try:
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "diffusers is required for SDXL execution. Install omnirt with runtime dependencies."
            ) from exc
        return StableDiffusionXLPipeline

    def _load_pipeline(
        self,
        *,
        source: str,
        torch_dtype: Any,
        scheduler_name: str,
        config: Dict[str, Any],
    ):
        if self._pipeline is not None and self._pipeline_source == source and self._pipeline_dtype == torch_dtype:
            return self._pipeline

        pipeline_cls = self._diffusers_pipeline_cls()
        pipeline = pipeline_cls.from_pretrained(
            source,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        if scheduler_name != "euler-discrete":
            raise ValueError(f"Unsupported SDXL scheduler: {scheduler_name}")
        scheduler_config = dict(config)
        if getattr(pipeline, "scheduler", None) is not None and hasattr(pipeline.scheduler, "config"):
            scheduler_config["scheduler_config"] = pipeline.scheduler.config
        pipeline.scheduler = build_scheduler(scheduler_config)
        self._wrap_pipeline_modules(pipeline)
        pipeline = self.runtime.to_device(pipeline, dtype=torch_dtype)
        self._apply_adapters(pipeline)
        self._pipeline = pipeline
        self._pipeline_source = source
        self._pipeline_dtype = torch_dtype
        return pipeline

    def _wrap_pipeline_modules(self, pipeline: Any) -> None:
        for tag in ("text_encoder", "text_encoder_2", "unet", "vae"):
            module = getattr(pipeline, tag, None)
            if module is None:
                continue
            wrapped = self.runtime.wrap_module(module, tag=tag)
            setattr(pipeline, tag, wrapped)
            self.components[tag] = wrapped

    def _apply_adapters(self, pipeline: Any) -> None:
        if not self.adapters:
            return
        if not hasattr(pipeline, "load_lora_weights"):
            raise DependencyUnavailableError("Current pipeline does not support LoRA loading.")
        for adapter in self.adapters:
            pipeline.load_lora_weights(adapter.path)
            if hasattr(pipeline, "fuse_lora"):
                pipeline.fuse_lora(lora_scale=adapter.scale)
