import math
from os import PathLike
from pathlib import Path
from typing import Optional, Any, Literal
from safetensors import safe_open
import torch
from pydantic import BaseModel, Field
from diffusers import QwenImageEditPlusPipeline
import time
from PIL import Image

import json

from dotenv import load_dotenv

from schemas.custom_types import BFloatTensor, IntTensor
load_dotenv()

from logger_config import logger
import hashlib

from diffusers.models import QwenImageTransformer2DModel
from modules.image_edit.qwen_manager import QwenManager
from config import Settings

class EmbeddedPrompting(BaseModel):
    prompt_embeds: BFloatTensor
    prompt_embeds_mask: Optional[IntTensor] = None

class TextPrompting(BaseModel):
    prompt: str = Field(alias="positive")
    negative_prompt: Optional[str] = Field(default=None, alias="negative")

class QwenEditModule(QwenManager):
    """Qwen module for image editing operations."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._empty_image = Image.new('RGB', (1024, 1024))

        self.base_model_path = settings.qwen_edit_base_model_path
        self.edit_model_path = settings.qwen_edit_model_path
        self.prompt_path = settings.qwen_edit_prompt_path
        self.prompting = self._set_prompting()

        self.pipe_config = {
            "num_inference_steps": settings.num_inference_steps,
            "true_cfg_scale": settings.true_cfg_scale,
            "height": settings.qwen_edit_height,
            "width": settings.qwen_edit_width,

        }

    def _set_text_prompting(self, path: Optional[PathLike] = None) -> TextPrompting:
        path = path or self.prompt_path
        with open(path, "r") as f:
            edit_prompt = TextPrompting.model_validate_json(json.dumps(json.load(f)))
            return edit_prompt


    def _set_embedded_prompting(self, path: Optional[PathLike] = None) -> EmbeddedPrompting:
        path = path or self.prompt_path
        with safe_open(path,framework="pt", device=self.device)as f:
            tensors = {key: f.get_tensor(key) for key in f.keys()}
            embedding = EmbeddedPrompting(**tensors)
        return embedding
    
    def _set_prompting(self, path: Optional[PathLike] = None) -> TextPrompting | EmbeddedPrompting:
        path = Path(path or self.prompt_path)
        if path.suffix == ".safetensors":
            return self._set_embedded_prompting(path)
        else:
            return self._set_text_prompting(path)

    def _get_model_transformer(self):
        """Load the Nunchaku Qwen transformer for image editing."""
        return  QwenImageTransformer2DModel.from_pretrained(
                self.edit_model_path,
                subfolder="transformer",
                torch_dtype=self.dtype
            )

    def _get_model_pipe(self, transformer, scheduler):

        return QwenImageEditPlusPipeline.from_pretrained(
                self.edit_model_path,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=self.dtype
            )

    def _get_scheduler_config(self):
        """Return scheduler configuration for image editing."""
        return  {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }

    def _prepare_input_image(self, image: Image, megapixels: float = 1.0):
        total = int(megapixels * 1024 * 1024)

        scale_by = math.sqrt(total / (image.width * image.height))
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)

        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _run_model_pipe(self, seed: Optional[int] = None, **kwargs):
        if seed:
            kwargs.update(dict(generator=torch.Generator(device=self.device).manual_seed(seed)))
        image = kwargs.pop("image", self._empty_image)
        result = self.pipe(
                image=image,
                **self.pipe_config,
                **kwargs)
        return result
    
    def _run_edit_pipe(self,
                       prompt_image: Image.Image,
                       seed: Optional[int] = None,
                       **kwargs):
        prompt_image = self._prepare_input_image(prompt_image)
        logger.info(f"Prompt image size: {prompt_image.size}")
        logger.info(f"Prompt image: {kwargs}")
        return self._run_model_pipe(seed=seed, image=prompt_image, **kwargs)
    
    
    def edit_image(self, prompt_image: Image.Image, seed: int, prompt: Optional[str] = None):
        """ 
        Edit the image using Qwen Edit.

        Args:
            prompt_image: The prompt image to edit.
            reference_image: The reference image to edit.

        Returns:
            The edited image.
        """
        if self.pipe is None:
            logger.error("Edit Model is not loaded")
            raise RuntimeError("Edit Model is not loaded")
        
        try:
            start_time = time.time()

            prompting = self.prompting.model_dump()
            if prompt:
                prompting["prompt"] = prompt
            
            # Run the edit pipe
            result = self._run_edit_pipe(prompt_image=prompt_image,
                                         **prompting,
                                         seed=seed)
            
            generation_time = time.time() - start_time
            
            image_edited = result.images[0]
            
            logger.success(f"Edited image generated in {generation_time:.2f}s, Size: {image_edited.size}, Seed: {seed}")
            
            return image_edited
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise e