from typing import Optional

from pydantic import BaseModel


class GenerateResponse(BaseModel):
    generation_time: float 
    ply_file_base64: Optional[str | bytes] = None
    image_edited_file_base64: Optional[str] = None
    image_without_background_file_base64: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "generation_time": 7.2,
                "ply_file_base64": "base64_encoded_ply_file",
                "image_edited_file_base64": "base64_encoded_image_edited_file",
                "image_without_background_file_base64": "base64_encoded_image_without_background_file",
            }
        }

