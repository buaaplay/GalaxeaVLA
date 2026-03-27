from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import json
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch


def _load_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim != 3:
            raise ValueError(f"Expected HWC image array, got shape {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    torch_mod = None
    try:
        import torch as torch_mod  # type: ignore
    except Exception:
        torch_mod = None
    if torch_mod is not None and isinstance(value, torch_mod.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(value)}")


def _load_pose(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32)
    if pose.shape != (7,):
        raise ValueError(f"Expected ee_pose shape (7,), got {pose.shape}")
    return pose


def _load_gripper(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float32).reshape(()))


@dataclass
class ArmState:
    ee_pose: np.ndarray
    gripper: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArmState":
        return cls(
            ee_pose=_load_pose(data["ee_pose"]),
            gripper=_load_gripper(data["gripper"]),
        )

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "ee_pose": self.ee_pose.tolist(),
            "gripper": self.gripper,
        }


@dataclass
class Last0RobotState:
    left: ArmState
    right: ArmState

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Last0RobotState":
        return cls(
            left=ArmState.from_dict(data["left"]),
            right=ArmState.from_dict(data["right"]),
        )

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "left": self.left.to_serializable(),
            "right": self.right.to_serializable(),
        }


@dataclass
class Last0Observation:
    instruction: str
    head_rgb: Image.Image
    left_wrist_rgb: Image.Image
    right_wrist_rgb: Image.Image
    robot_state: Last0RobotState | None = None
    idx: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Last0Observation":
        if "instruction" not in data:
            raise ValueError("Observation must include an 'instruction' field.")
        return cls(
            instruction=data["instruction"],
            head_rgb=_load_pil_image(data["head_rgb"]),
            left_wrist_rgb=_load_pil_image(data["left_wrist_rgb"]),
            right_wrist_rgb=_load_pil_image(data["right_wrist_rgb"]),
            robot_state=Last0RobotState.from_dict(data["robot_state"]) if data.get("robot_state") else None,
            idx=int(data.get("idx", 0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Last0Observation":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def slow_images(self) -> list[Image.Image]:
        return [self.head_rgb]

    def fast_images(self) -> list[Image.Image]:
        return [self.left_wrist_rgb, self.right_wrist_rgb]


@dataclass
class PoseCommand:
    left_target_pose: np.ndarray
    right_target_pose: np.ndarray
    left_gripper: float
    right_gripper: float
    left_delta_action: np.ndarray
    right_delta_action: np.ndarray

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "left_target_pose": self.left_target_pose.tolist(),
            "right_target_pose": self.right_target_pose.tolist(),
            "left_gripper": self.left_gripper,
            "right_gripper": self.right_gripper,
            "left_delta_action": self.left_delta_action.tolist(),
            "right_delta_action": self.right_delta_action.tolist(),
        }


@dataclass
class Last0InferenceResult:
    action_chunk: np.ndarray
    first_action: np.ndarray
    pose_command: PoseCommand | None = None

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "action_chunk": self.action_chunk.tolist(),
            "first_action": self.first_action.tolist(),
            "pose_command": None if self.pose_command is None else self.pose_command.to_serializable(),
        }
