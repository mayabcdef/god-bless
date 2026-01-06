from __future__ import annotations

import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

from config import Settings
from logger_config import logger


class BiRefNetBackgroundRemovalService:
    """
    BiRefNet background removal with flow:
    resize -> infer mask -> bbox -> crop ORIGINAL -> apply alpha (rgb*alpha -> black bg) -> resize output
    Output: RGB with black background, soft edges.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # settings
        self.input_size = self.settings.input_image_size          # (H,W)
        self.output_size = self.settings.output_image_size        # (H,W)
        self.padding_percentage = self.settings.padding_percentage
        self.limit_padding = self.settings.limit_padding
        self.model_id = self.settings.background_removal_model_id

        # thresholds
        # - bbox_threshold: to find bbox (lower so bbox is larger, won't cut hair/soft edges)
        # - alpha_clean_threshold: to clean background noise (can be higher)
        self.bbox_threshold = getattr(self.settings, "bbox_threshold", 0.2)
        self.alpha_clean_threshold = getattr(self.settings, "alpha_clean_threshold", 0.05)

        # device
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"

        # model
        self.model: AutoModelForImageSegmentation | None = None

        # transforms for inference
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    async def startup(self) -> None:
        logger.info(f"Loading {self.model_id} model...")
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            logger.success(f"{self.model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.model_id} model: {e}")

    async def shutdown(self) -> None:
        self.model = None
        logger.info("BiRefNetBackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        if self.model is None:
            raise RuntimeError(f"{self.model_id} model not initialized.")

    @torch.no_grad()
    def _infer_mask_resized(self, image_rgb: Image.Image) -> np.ndarray:
        """
        Returns float mask in [0,1] at resized scale: shape (H_resized,W_resized)
        """
        inp = self.transform(image_rgb).unsqueeze(0).to(self.device)  # (1,3,H,W)
        preds = self.model(inp)[-1].sigmoid()  # (1,1,H,W)
        pred = preds[0].squeeze().detach().float().cpu().numpy()      # (H,W) float [0,1]
        return pred

    def _mask_to_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        """
        mask: (H,W) float in [0,1]
        returns bbox in resized scale: (left, top, right, bottom)
        """
        H, W = mask.shape
        ys, xs = np.where(mask > self.bbox_threshold)

        if len(xs) == 0 or len(ys) == 0:
            return (0, 0, W, H)

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        width = x_max - x_min
        height = y_max - y_min

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        # square crop + padding
        size = max(width, height)
        size = int(size * (1.0 + self.padding_percentage))

        left = int(cx - size / 2)
        right = int(cx + size / 2)
        top = int(cy - size / 2)
        bottom = int(cy + size / 2)

        if self.limit_padding:
            left = max(0, left)
            top = max(0, top)
            right = min(W, right)
            bottom = min(H, bottom)

        # safety
        if right <= left:
            right = min(W, left + 1)
        if bottom <= top:
            bottom = min(H, top + 1)

        return (left, top, right, bottom)

    @staticmethod
    def _scale_bbox_to_original(
        bbox_resized: tuple[int, int, int, int],
        resized_hw: tuple[int, int],
        original_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = bbox_resized
        Hr, Wr = resized_hw
        Ho, Wo = original_hw

        sx = Wo / Wr
        sy = Ho / Hr

        left_o = int(left * sx)
        right_o = int(right * sx)
        top_o = int(top * sy)
        bottom_o = int(bottom * sy)

        # clamp
        left_o = max(0, min(Wo, left_o))
        right_o = max(0, min(Wo, right_o))
        top_o = max(0, min(Ho, top_o))
        bottom_o = max(0, min(Ho, bottom_o))

        if right_o <= left_o:
            right_o = min(Wo, left_o + 1)
        if bottom_o <= top_o:
            bottom_o = min(Ho, top_o + 1)

        return (left_o, top_o, right_o, bottom_o)

    @staticmethod
    def _resize_mask_to_original(mask_resized: np.ndarray, original_size_wh: tuple[int, int]) -> np.ndarray:
        """
        mask_resized: (H,W) float [0,1]
        original_size_wh: (W_orig,H_orig)
        return mask_orig float [0,1] shape (H_orig,W_orig)
        """
        mask_pil = Image.fromarray((mask_resized * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize(original_size_wh, resample=Image.Resampling.LANCZOS)
        return (np.array(mask_pil).astype(np.float32) / 255.0)

    @staticmethod
    def _rgb_black_bg(image_rgb: Image.Image, alpha_mask: np.ndarray) -> Image.Image:
        """
        image_rgb: PIL RGB
        alpha_mask: float [0,1] same H,W
        Output: PIL RGB where background is black (rgb * alpha), soft edges.
        """
        img_np = np.array(image_rgb.convert("RGB")).astype(np.float32) / 255.0  # (H,W,3)
        alpha = alpha_mask.astype(np.float32)                                   # (H,W)

        out_np = img_np * alpha[..., None]                                      # (H,W,3)
        out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)

        return Image.fromarray(out_np, mode="RGB")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Main function: remove bg with bbox + final resize.
        Returns RGB image in self.output_size with BLACK background and soft edges.
        """
        self.ensure_ready()
        t1 = time.time()

        img_rgb = image.convert("RGB")
        Wo, Ho = img_rgb.size  # (W,H)

        # 1) infer mask at resized scale
        mask_resized = self._infer_mask_resized(img_rgb)  # (H_resized,W_resized)
        Hr, Wr = mask_resized.shape

        # 2) bbox in resized scale (use bbox_threshold)
        bbox_resized = self._mask_to_bbox(mask_resized)

        # 3) scale bbox to original
        bbox_orig = self._scale_bbox_to_original(
            bbox_resized=bbox_resized,
            resized_hw=(Hr, Wr),
            original_hw=(Ho, Wo),
        )
        left, top, right, bottom = bbox_orig

        # 4) resize mask back to original & crop both
        mask_orig = self._resize_mask_to_original(mask_resized, original_size_wh=(Wo, Ho))
        img_crop = img_rgb.crop((left, top, right, bottom))
        mask_crop = mask_orig[top:bottom, left:right]

        # 5) clean alpha a bit (remove noise, KEEP SOFT EDGES)
        if self.alpha_clean_threshold is not None and self.alpha_clean_threshold > 0:
            mask_crop = mask_crop.copy()
            mask_crop[mask_crop < float(self.alpha_clean_threshold)] = 0.0

        # 6) convert -> RGB with black background (rgb * alpha)
        rgb_black = self._rgb_black_bg(img_crop, mask_crop)

        # 7) final resize AFTER crop
        out = rgb_black.resize((self.output_size[1], self.output_size[0]), resample=Image.Resampling.LANCZOS)

        removal_time = time.time() - t1
        logger.success(
            f"BiRefNet BG (RGB black) - Time: {removal_time:.2f}s - OutputSize: {out.size} - InputSize: {image.size} - BBoxOrig: {bbox_orig}"
        )
        return out

