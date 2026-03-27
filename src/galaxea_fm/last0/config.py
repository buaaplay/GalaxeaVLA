from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Last0InferenceConfig:
    last0_root: str = "../last0"
    model_root: str = ""
    stats_path: str | None = None
    device: str = "cuda"
    cuda_id: int = 0
    action_chunk: int = 16
    latent_size: int = 8
    use_latent: bool = True
    use_proprio: bool = False
    seed: int = 42

    @classmethod
    def from_file(cls, path: str | Path) -> "Last0InferenceConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw_cfg: Dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(**raw_cfg)
