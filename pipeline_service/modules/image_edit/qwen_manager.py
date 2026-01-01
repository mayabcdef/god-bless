from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import (
    
    FlowMatchEulerDiscreteScheduler
)
from huggingface_hub import login, whoami
from PIL import Image

from config import Settings
from logger_config import logger


@dataclass(slots=True)
class QwenResult:
    image: Image.Image
    generation_time: float
    seed: int

class QwenManager:
    """Handles Qwen pipeline responsible for generating 2D images."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe = None
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"
        self.dtype = self._resolve_dtype(settings.dtype)
        self.gpu_index = settings.qwen_gpu

    async def startup(self) -> None:
        """Initialize the Qwen pipeline."""
        logger.info("Initializing QwenManager...")
        await self._load_pipeline()
        logger.success("QwenManager ready.")

    async def shutdown(self) -> None:
        """Shutdown the pipeline and free resources."""
        if self.pipe:
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
        self.pipe = None
        logger.info("QwenEditManager closed.")

    def is_ready(self) -> bool:
        """Check if pipeline is loaded and ready."""
        return self.pipe is not None

    async def _load_pipeline(self) -> None:
        """Load the complete pipeline (transformer + scheduler + pipe)."""
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.gpu_index)
            except Exception as err:
                logger.warning(f"Failed to set CUDA device ({self.gpu_index}): {err}")

        t1 = time.time()

        # Load components (to be implemented by subclasses)
        transformer = self._get_model_transformer()
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self._get_scheduler_config())
        self.pipe = self._get_model_pipe(transformer, scheduler)

        self.pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Edit-2511-Lightning",
            weight_name=self.settings.qwen_edit_base_model_path
        )

        # from optimum.quanto import freeze, qint8, quantize
        # quantize(self.pipe.transformer, weights=qint8)
        # freeze(self.pipe.transformer)
        
        self.pipe.fuse_lora()
        self.pipe.transformer.set_attention_backend("_flash_3_hub")

        # Move model pipe to device
        self.pipe = self.pipe.to(self.device)

        load_time = time.time() - t1

        logger.success(f"Qwen pipeline ready (loading: {load_time:.2f}s). Loaded on {self.device} with dtype={self.dtype}.")

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        resolved = mapping.get(dtype.lower(), torch.bfloat16)
        if not torch.cuda.is_available() and resolved in {torch.float16, torch.bfloat16}:
            return torch.float32
        return resolved

    def _derive_seed(self, prompt: str) -> int:
        hash_object = hashlib.md5(prompt.encode("utf-8"))
        # limit to 32 bits
        return int(hash_object.hexdigest()[:8], 16) % (2**32)

     # Abstract methods to be implemented by subclasses
    def _get_model_transformer(self):
        """Load and return the transformer model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_model_transformer()")

    def _get_scheduler_config(self) -> dict:
        """Return scheduler configuration. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_scheduler_config()")

    def _get_model_pipe(self, transformer, scheduler):
        """Create and return the pipeline. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_model_pipe()")

