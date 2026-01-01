import torch
from pydantic_tensor import Tensor
from typing import Any, Literal, TypeAlias

IntTensor: TypeAlias = Tensor[torch.Tensor, Any, Literal["int64"]]
HalfTensor: TypeAlias = Tensor[torch.Tensor, Any, Literal["float16"]]
BFloatTensor: TypeAlias = Tensor[torch.Tensor, Any, Literal["bfloat16"]]
FloatTensor: TypeAlias = Tensor[torch.Tensor, Any, Literal["float32"]]
DoubleTensor: TypeAlias = Tensor[torch.Tensor, Any, Literal["float64"]]