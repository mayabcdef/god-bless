from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

config_dir = Path(__file__).parent

class Settings(BaseSettings):
    api_title: str = "3D Generation pipeline Service"

    # API settings
    host: str = "0.0.0.0"
    port: int = 10006

    # GPU settings
    qwen_gpu: int = Field(default=0, env="QWEN_GPU")
    trellis_gpu: int = Field(default=0, env="TRELLIS_GPU")
    dtype: str = Field(default="bf16", env="QWEN_DTYPE")

    # Hugging Face settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    # Generated files settings
    save_generated_files: bool = Field(default=False, env="SAVE_GENERATED_FILES")
    send_generated_files: bool = Field(default=False, env="SEND_GENERATED_FILES")
    output_dir: Path = Field(default=Path("generated_outputs"), env="OUTPUT_DIR")

    # Trellis settings
    trellis_model_id: str = Field(default="jetx/trellis-image-large", env="TRELLIS_MODEL_ID")
    trellis_sparse_structure_steps: int = Field(default=8, env="TRELLIS_SPARSE_STRUCTURE_STEPS")
    trellis_sparse_structure_cfg_strength: float = Field(default=5.75, env="TRELLIS_SPARSE_STRUCTURE_CFG_STRENGTH")
    trellis_slat_steps: int = Field(default=20, env="TRELLIS_SLAT_STEPS")
    trellis_slat_cfg_strength: float = Field(default=2.4, env="TRELLIS_SLAT_CFG_STRENGTH")
    trellis_num_oversamples: int = Field(default=3, env="TRELLIS_NUM_OVERSAMPLES")
    compression: bool = Field(default=False, env="COMPRESSION")

    # Qwen Edit settings
    qwen_edit_base_model_path: str = Field(default="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",env="QWEN_EDIT_BASE_MODEL_PATH")
    qwen_edit_model_path: str = Field(default="Qwen/Qwen-Image-Edit-2511",env="QWEN_EDIT_MODEL_PATH")
    qwen_edit_height: int = Field(default=1024, env="QWEN_EDIT_HEIGHT")
    qwen_edit_width: int = Field(default=1024, env="QWEN_EDIT_WIDTH")
    num_inference_steps: int = Field(default=4, env="NUM_INFERENCE_STEPS")
    true_cfg_scale: float = Field(default=1.0, env="TRUE_CFG_SCALE")
    qwen_edit_prompt_path: Path = Field(default=config_dir.joinpath("qwen_edit_prompt.json"), env="QWEN_EDIT_PROMPT_PATH")

    # Backgorund removal settings
    background_removal_model_id: str = Field(default="tuandao-zenai/rm_bg", env="BACKGROUND_REMOVAL_MODEL_ID")
    input_image_size: tuple[int, int] = Field(default=(1024, 1024), env="INPUT_IMAGE_SIZE") # (height, width)
    output_image_size: tuple[int, int] = Field(default=(518, 518), env="OUTPUT_IMAGE_SIZE") # (height, width)
    padding_percentage: float = Field(default=0.2, env="PADDING_PERCENTAGE")
    limit_padding: bool = Field(default=True, env="LIMIT_PADDING")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

__all__ = ["Settings", "settings"]

