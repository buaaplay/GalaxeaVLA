from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json
import numpy as np
import torch
from PIL import Image


def _load_image_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() == ".npy":
            tensor = torch.from_numpy(np.load(path))
        else:
            image = Image.open(path).convert("RGB")
            tensor = torch.from_numpy(np.asarray(image))
    else:
        raise TypeError(f"Unsupported image input type: {type(value)}")

    if tensor.ndim == 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.ndim == 4 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims [T,C,H,W], got {tuple(tensor.shape)}")

    return tensor.contiguous()


def _load_state_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().to(torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 2:
        raise ValueError(f"Expected state tensor with 2 dims [T,D], got {tuple(tensor.shape)}")

    return tensor.contiguous()


@dataclass
class PolicyObservation:
    instruction: str
    images: Dict[str, torch.Tensor]
    state: Dict[str, torch.Tensor]
    coarse_task: str = ""
    idx: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyObservation":
        if "instruction" not in data:
            raise ValueError("Observation must include an 'instruction' field.")
        if "images" not in data or "state" not in data:
            raise ValueError("Observation must include both 'images' and 'state'.")

        images = {key: _load_image_tensor(value) for key, value in data["images"].items()}
        state = {key: _load_state_tensor(value) for key, value in data["state"].items()}
        return cls(
            instruction=data["instruction"],
            coarse_task=data.get("coarse_task", ""),
            images=images,
            state=state,
            idx=int(data.get("idx", 0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PolicyObservation":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


@dataclass
class InferenceResult:
    action: Dict[str, np.ndarray]
    action_flat: np.ndarray
    first_action_flat: np.ndarray

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "action": {key: value.tolist() for key, value in self.action.items()},
            "action_flat": self.action_flat.tolist(),
            "first_action_flat": self.first_action_flat.tolist(),
        }
