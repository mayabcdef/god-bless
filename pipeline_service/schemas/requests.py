from typing import Optional, Literal

from pydantic import BaseModel
from fastapi import File, UploadFile

from schemas.trellis_schemas import TrellisParamsOverrides


class GenerateRequest(BaseModel):
    # Prompt data
    prompt_type: Literal["text", "image"] = "image"
    prompt_image: str 
    seed: int = -1

    # Trellis parameters
    trellis_params: Optional[TrellisParamsOverrides] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt_type": "text",
                "prompt_image": "file_name.jpg",
                "seed": 42,
                "trellis_params": {
                    "sparse_structure_steps": 8,
                    "sparse_structure_cfg_strength": 5.75,
                    "slat_steps": 20,
                    "slat_cfg_strength": 2.4,
                }
            }
        }

