from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import importlib
import json
import sys

import numpy as np
import torch

from .action_adapter import Last0ActionAdapter
from .config import Last0InferenceConfig
from .schemas import Last0InferenceResult, Last0Observation


class Last0InferenceRuntime:
    def __init__(self, config: Last0InferenceConfig, initialize_model: bool = True):
        self.config = config
        self.project_root = Path(__file__).resolve().parents[3]
        self.last0_root = self._resolve_path(config.last0_root)
        self.device = self._resolve_device(config.device)
        self.adapter = Last0ActionAdapter()

        self._deps_loaded = False
        self._robot_utils = None
        self._processor_cls = None
        self._auto_model_cls = None
        self._action_tokenizer_cls = None

        self.model_dir = self._resolve_model_dir(self._resolve_path(config.model_root))
        self.stats_path = self._resolve_stats_path()
        self.statistic, self.unnorm_key = self._load_stats(self.stats_path)

        self.cfg = SimpleNamespace(
            cuda=str(config.cuda_id),
            num_open_loop_steps=config.action_chunk,
            latent_size=config.latent_size,
            use_latent=bool(config.use_latent),
            use_proprio=bool(config.use_proprio),
            seed=config.seed,
        )

        self.processor = None
        self.model = None
        self.action_tokenizer = None

        if initialize_model:
            self._load_dependencies()
            self.processor = self._processor_cls.from_pretrained(str(self.model_dir))
            tokenizer = self.processor.tokenizer
            self.model = self._auto_model_cls.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                use_latent=bool(config.use_latent),
                flow=True,
                action_dim=self.statistic["action_dim"],
                action_chunk=config.action_chunk,
                fast_and_slow=True,
                fast_image_num=2,
            )
            self.action_tokenizer = self._action_tokenizer_cls(tokenizer)

    def _resolve_path(self, path: str) -> Path:
        if not path:
            raise ValueError("Path must not be empty.")
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.project_root / path_obj).resolve()

    def _resolve_device(self, device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    def _load_dependencies(self) -> None:
        if self._deps_loaded:
            return

        last0_root_str = str(self.last0_root)
        if last0_root_str not in sys.path:
            sys.path.insert(0, last0_root_str)

        robot_utils = importlib.import_module("experiments.robot.robot_utils")
        janus_models = importlib.import_module("janus.models")
        transformers_mod = importlib.import_module("transformers")

        self._robot_utils = robot_utils
        self._processor_cls = janus_models.VLChatProcessor
        self._action_tokenizer_cls = janus_models.ActionTokenizer
        self._auto_model_cls = transformers_mod.AutoModelForCausalLM
        self._deps_loaded = True

    @staticmethod
    def _resolve_model_dir(model_root: Path) -> Path:
        tfmr = model_root / "tfmr"
        return tfmr if tfmr.exists() else model_root

    def _resolve_stats_path(self) -> Path:
        if self.config.stats_path:
            return self._resolve_path(self.config.stats_path)

        root = self._resolve_path(self.config.model_root)
        candidates = [
            root / "stats_data.json",
            root.parent / "stats_data.json",
            root.parent / f"{root.parent.name}_train_statistics.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find stats file near {root}")

    @staticmethod
    def _load_stats(stats_path: Path) -> tuple[Dict[str, Any], str]:
        with open(stats_path, "r", encoding="utf-8") as handle:
            stats_data = json.load(handle)
        if len(stats_data) != 1:
            raise ValueError(f"Expected one top-level stats key, got: {list(stats_data.keys())}")
        key = next(iter(stats_data))
        statistic: Dict[str, Any] = {
            "action_mask": np.array(stats_data[key]["action"]["mask"]),
            "action_min": np.array(stats_data[key]["action"]["q01"], dtype=np.float32),
            "action_max": np.array(stats_data[key]["action"]["q99"], dtype=np.float32),
            "action_dim": int(len(stats_data[key]["action"]["q01"])),
        }
        if "state" in stats_data[key]:
            statistic["state_mask"] = np.array(stats_data[key]["state"]["mask"])
            statistic["state_min"] = np.array(stats_data[key]["state"]["q01"], dtype=np.float32)
            statistic["state_max"] = np.array(stats_data[key]["state"]["q99"], dtype=np.float32)
        return statistic, key

    def predict(self, observation: Last0Observation, clip_action: bool = False) -> Last0InferenceResult:
        self._load_dependencies()
        if self.processor is None or self.model is None or self.action_tokenizer is None:
            raise RuntimeError("Runtime was created without model initialization.")

        if self.device == "cpu":
            raise RuntimeError("LaST0 inference currently expects CUDA execution.")

        actions = self._robot_utils.get_action(
            self.cfg,
            self.statistic,
            self.action_tokenizer,
            self.processor,
            observation.instruction,
            self.model,
            observation.fast_images(),
            observation.slow_images(),
            state=None,
        )
        action_chunk = np.asarray(actions, dtype=np.float32)
        first_action = action_chunk[0]
        if clip_action:
            first_action = self.adapter.clip_action(first_action)
            action_chunk = action_chunk.copy()
            action_chunk[0] = first_action

        pose_command = None
        if observation.robot_state is not None:
            pose_command = self.adapter.adapt(first_action, observation.robot_state, clip=clip_action)

        return Last0InferenceResult(
            action_chunk=action_chunk,
            first_action=first_action,
            pose_command=pose_command,
        )
