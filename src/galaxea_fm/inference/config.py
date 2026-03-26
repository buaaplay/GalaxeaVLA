from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class InferenceRuntimeConfig:
    model_config_path: str = "configs/model/vla/g0plus.yaml"
    data_config_path: str = "configs/data/r1lite/arm_torso_velocity_chassis.yaml"
    checkpoint_path: str = "checkpoints/G0Plus_3B_base/model_state_dict.pt"
    dataset_stats_path: str = "checkpoints/G0Plus_3B_base/dataset_stats.json"
    paligemma_path: str = "data/google/paligemma-3b-pt-224"
    device: str = "cuda"
    strict_checkpoint_loading: bool = False
    seed: int = 42

    @classmethod
    def from_file(cls, path: str | Path) -> "InferenceRuntimeConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw_cfg: Dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(**raw_cfg)
