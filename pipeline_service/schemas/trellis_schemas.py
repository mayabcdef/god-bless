from dataclasses import dataclass
from typing import Optional, TypeAlias
from PIL import Image

from schemas.overridable import OverridableModel


class TrellisParams(OverridableModel):
    """Trellis parameters with automatic fallback to settings."""
    sparse_structure_steps: int
    sparse_structure_cfg_strength: float
    slat_steps: int
    slat_cfg_strength: float
    num_oversamples: int = 1
    
    @classmethod
    def from_settings(cls, settings) -> "TrellisParams":
        return cls(
            sparse_structure_steps = settings.trellis_sparse_structure_steps,
            sparse_structure_cfg_strength = settings.trellis_sparse_structure_cfg_strength,
            slat_steps = settings.trellis_slat_steps,
            slat_cfg_strength = settings.trellis_slat_cfg_strength,
            num_oversamples = settings.trellis_num_oversamples,
        )

TrellisParamsOverrides = TrellisParams.Overrides


@dataclass
class TrellisRequest:
    """Request for Trellis 3D generation (internal use only)."""
    images: list[Image.Image]
    seed: int
    params: Optional[TrellisParamsOverrides] = None


@dataclass(slots=True)
class TrellisResult:
    """Result from Trellis 3D generation."""
    ply_file: bytes | None = None


