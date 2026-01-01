#!/usr/bin/env python3

# =======================
# Imports
# =======================

import io
import os
import argparse
from functools import lru_cache

import torch
import numpy as np
from PIL import Image

from renderers.gs_renderer.renderer import Renderer
from renderers.ply_loader import PlyLoader


# =======================
# Constants
# =======================

IMG_WIDTH = 518
IMG_HEIGHT = 518
GRID_VIEW_GAP = 5

VIEWS_NUMBER = 16
THETA_ANGLES = np.linspace(0, 360, num=VIEWS_NUMBER)
PHI_ANGLES = np.full_like(THETA_ANGLES, -15.0)
GRID_VIEW_INDICES = [1, 5, 9, 13]

CAM_RAD = 2.5
CAM_FOV_DEG = 49.1
REF_BBOX_SIZE = 1.5


# =======================
# Internal cached objects
# =======================

@lru_cache(maxsize=1)
def _get_ply_loader() -> PlyLoader:
    return PlyLoader()


@lru_cache(maxsize=1)
def _get_renderer() -> Renderer:
    return Renderer()


@lru_cache(maxsize=1)
def _get_view_angles():
    theta = THETA_ANGLES[GRID_VIEW_INDICES].astype("float32")
    phi = PHI_ANGLES[GRID_VIEW_INDICES].astype("float32")
    return theta, phi


# =======================
# PURE PUBLIC API
# =======================

def ply_bytes_to_grid_png_bytes(
    ply_bytes: bytes,
    device: torch.device,
) -> bytes:
    """
    PURE CORE API

    Input:
        ply_bytes : bytes của file .ply
        device    : torch.device

    Output:
        png_bytes : bytes của ảnh PNG grid 2x2

    Không đọc / ghi file
    Có thể import dùng lại ở bất kỳ đâu
    """

    ply_loader = _get_ply_loader()
    renderer = _get_renderer()
    theta_angles, phi_angles = _get_view_angles()

    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)

    with torch.no_grad():
        gs_data = ply_loader.from_buffer(io.BytesIO(ply_bytes))
        gs_data = gs_data.send_to_device(device)

        images = renderer.render_gs(
            gs_data,
            views_number=4,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            theta_angles=theta_angles,
            phi_angles=phi_angles,
            cam_rad=CAM_RAD,
            cam_fov=CAM_FOV_DEG,
            ref_bbox_size=REF_BBOX_SIZE,
            bg_color=bg_color,
        )

    # ---- combine grid 2x2 ----
    row_width = IMG_WIDTH * 2 + GRID_VIEW_GAP
    col_height = IMG_HEIGHT * 2 + GRID_VIEW_GAP

    combined = Image.new("RGB", (row_width, col_height), color="black")
    pil_images = [Image.fromarray(img.cpu().numpy()) for img in images]

    combined.paste(pil_images[0], (0, 0))
    combined.paste(pil_images[1], (IMG_WIDTH + GRID_VIEW_GAP, 0))
    combined.paste(pil_images[2], (0, IMG_HEIGHT + GRID_VIEW_GAP))
    combined.paste(
        pil_images[3],
        (IMG_WIDTH + GRID_VIEW_GAP, IMG_HEIGHT + GRID_VIEW_GAP),
    )

    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return buf.getvalue()


# =======================
# CLI (IO only)
# =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    ply_files = sorted(
        f for f in os.listdir(args.input_dir) if f.lower().endswith(".ply")
    )

    for fname in ply_files:
        ply_path = os.path.join(args.input_dir, fname)
        png_path = os.path.join(
            args.output_dir, os.path.splitext(fname)[0] + ".png"
        )

        try:
            with open(ply_path, "rb") as f:
                ply_bytes = f.read()

            png_bytes = ply_bytes_to_grid_png_bytes(ply_bytes, device)

            with open(png_path, "wb") as f:
                f.write(png_bytes)

            print(f"✅ {fname}")

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")


if __name__ == "__main__":
    main()
