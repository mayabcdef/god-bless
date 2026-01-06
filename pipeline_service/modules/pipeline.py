from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional

from PIL import Image
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import (
    GenerateRequest,
    GenerateResponse,
    TrellisParams,
    TrellisRequest,
    TrellisResult,
)
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.ben2_module import BEN2BackgroundRemovalService
from modules.background_removal.rmbg20_module import RMBG2BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.birenef_pipe import BiRefNetBackgroundRemovalService
from modules.utils import (
    secure_randint,
    set_random_seed,
    decode_image,
    to_png_base64,
    save_files,
)

from compare import compare


class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BiRefNetBackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()

        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()

        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""

        temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_imge_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_imge_bytes, seed=42)

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.

        Args:
            image_bytes: Raw image bytes from uploaded file

        Returns:
            PLY file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create request
        request = GenerateRequest(
            prompt_image=image_base64, prompt_type="image", seed=seed
        )

        # Generate
        response_3_views, response_1_views = await self.generate_gs(request)

        # Return binary PLY
        if not response_3_views.ply_file_base64 or not response_1_views.ply_file_base64:
            raise ValueError("PLY generation failed")
        #check time if > 25s then retun response_3_views
        print(f"Generation time: {response_3_views.generation_time}")
        if response_3_views.generation_time > 25:
            return response_3_views.ply_file_base64
        else:
            print("Generation time < 25s")
            king = await compare(image_bytes, response_3_views.ply_file_base64, response_1_views.ply_file_base64)

            if king == 1:
                return response_3_views.ply_file_base64
            else:
                return response_1_views.ply_file_base64


    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.

        Args:
            request: Generation request with prompt and settings

        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        image_edited = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=request.seed,
            prompt="Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        )

        # 2. Remove background
        image_without_background = self.rmbg.remove_background(image_edited)
        image_without_background.convert("RGB")

        # add another view of the image
        image_edited_2 = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=request.seed,
            prompt="Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        )
        image_without_background_2 = self.rmbg.remove_background(image_edited_2)
        image_without_background_2.convert("RGB")

        # add another view of the image
        image_edited_3 = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=request.seed,
            prompt="Show this object in back view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        )
        image_without_background_3 = self.rmbg.remove_background(image_edited_3)
        image_without_background_3.convert("RGB")

        original_image_without_background = self.rmbg.remove_background(image)
        original_image_without_background.convert("RGB")
        # save to debug
        # image_edited.save("image_edited.png")
        # # image_edited_2.save("image_edited_2.png")
        # image_without_background.save("image_without_background.png")
        # image_without_background_2.save("image_without_background_2.png")
        # image_without_background_3.save("image_without_background_3.png")
        # original_image_without_background.save("original_image_without_background.png")

        trellis_result_3_views: Optional[TrellisResult] = None
        trellis_result_1_views: Optional[TrellisResult] = None

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params
        print(f"Trellis parameters: {trellis_params}")

        # 3. Generate the 3D model
        trellis_result_3_views = self.trellis.generate(
            TrellisRequest(
                images=[image_without_background, image_without_background_2, image_without_background_3],
                seed=request.seed,
                params=trellis_params,
            ),
            threshold=25000,
            mode="multidiffusion",
        )

        trellis_result_1_views = self.trellis.generate(
            TrellisRequest(
                images=[image_without_background, original_image_without_background],
                seed=request.seed,
                params=trellis_params,
            ),
            threshold=10,
            mode="stochastic",
        )

        # Save generated files
        if self.settings.save_generated_files:
            save_files(
                trellis_result_3_views, 
                trellis_result_1_views,
                image, 
                image_edited, 
                image_without_background,
                image_edited_2,
                image_without_background_2,
                image_edited_3,
                image_without_background_3
            )

        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited)
            image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        logger.info(f"Total generation time: {generation_time} seconds")
        # Clean the GPU memory
        self._clean_gpu_memory()

        response_3_views = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result_3_views.ply_file if trellis_result_3_views else None,
            image_edited_file_base64=image_edited_base64
            if self.settings.send_generated_files
            else None,
            image_without_background_file_base64=image_without_background_base64
            if self.settings.send_generated_files
            else None,
        )

        response_1_views = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result_1_views.ply_file if trellis_result_1_views else None,
            image_edited_file_base64=image_edited_base64
            if self.settings.send_generated_files
            else None,
            image_without_background_file_base64=image_without_background_base64
            if self.settings.send_generated_files
            else None,
        )

        print("success")

        return response_3_views, response_1_views
