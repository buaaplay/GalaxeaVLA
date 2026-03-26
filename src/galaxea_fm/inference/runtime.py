from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from galaxea_fm.inference.config import InferenceRuntimeConfig
from galaxea_fm.inference.schemas import InferenceResult, PolicyObservation
from galaxea_fm.utils.config_resolvers import register_default_resolvers
from galaxea_fm.utils.normalizer import load_dataset_stats_from_json
from galaxea_fm.utils.pytorch_utils import dict_apply, dict_to_array, set_global_seed


class GalaxeaInferenceRuntime:
    def __init__(self, config: InferenceRuntimeConfig, initialize_model: bool = True):
        register_default_resolvers()
        self.config = config
        self.project_root = Path(__file__).resolve().parents[3]
        self.device = self._resolve_device(config.device)
        self.cfg = self._build_runtime_cfg()
        self.processor = None
        self.model = None

        if initialize_model:
            self.processor = self._build_processor().eval()
            self.model = self._build_model()

    def _resolve_path(self, path: str) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return self.project_root / path_obj

    def _resolve_device(self, device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    def _build_runtime_cfg(self):
        model_cfg = OmegaConf.load(self._resolve_path(self.config.model_config_path))
        data_cfg = OmegaConf.load(self._resolve_path(self.config.data_config_path))

        runtime_cfg = OmegaConf.create({"model": model_cfg, "data": data_cfg})
        runtime_cfg.model.model_arch.pretrained_model_path = str(self._resolve_path(self.config.paligemma_path))
        runtime_cfg.model.processor.tokenizer_params.pretrained_model_name_or_path = str(
            self._resolve_path(self.config.paligemma_path)
        )
        return runtime_cfg

    def _build_processor(self):
        processor = instantiate(self.cfg.model.processor)
        stats = load_dataset_stats_from_json(str(self._resolve_path(self.config.dataset_stats_path)))
        processor.set_normalizer_from_stats(stats)
        return processor

    def _load_state_dict(self) -> Dict[str, Any]:
        checkpoint = torch.load(
            self._resolve_path(self.config.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        return checkpoint

    def _build_model(self):
        set_global_seed(self.config.seed)
        model = instantiate(self.cfg.model.model_arch)
        state_dict = self._load_state_dict()
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=self.config.strict_checkpoint_loading,
        )
        if self.config.strict_checkpoint_loading and (missing or unexpected):
            raise RuntimeError(
                f"Strict checkpoint loading failed. Missing={len(missing)}, unexpected={len(unexpected)}"
            )
        return model.to(self.device).eval()

    def _validate_shapes(self, observation: PolicyObservation) -> None:
        obs_steps = self.cfg.data.obs_size

        for meta in self.cfg.data.shape_meta.images:
            key = meta.key
            if key not in observation.images:
                raise KeyError(f"Missing image key '{key}' in observation.")
            image = observation.images[key]
            if image.shape[0] != obs_steps:
                raise ValueError(f"Image key '{key}' expects {obs_steps} obs steps, got {image.shape[0]}.")

        for meta in self.cfg.data.shape_meta.state:
            key = meta.key
            if key not in observation.state:
                raise KeyError(f"Missing state key '{key}' in observation.")
            state = observation.state[key]
            if state.shape[0] != obs_steps:
                raise ValueError(f"State key '{key}' expects {obs_steps} obs steps, got {state.shape[0]}.")
            if state.shape[-1] != meta.raw_shape:
                raise ValueError(
                    f"State key '{key}' expects dim {meta.raw_shape}, got {state.shape[-1]}."
                )

    def _build_raw_sample(self, observation: PolicyObservation) -> Dict[str, Any]:
        self._validate_shapes(observation)
        obs_steps = self.cfg.data.obs_size

        return {
            "task": observation.instruction,
            "coarse_task": observation.coarse_task,
            "images": observation.images,
            "state": observation.state,
            "state_is_pad": torch.zeros(obs_steps, dtype=torch.bool),
            "image_is_pad": torch.zeros(obs_steps, dtype=torch.bool),
            "idx": observation.idx,
        }

    @staticmethod
    def _batchify(sample: Dict[str, Any]) -> Dict[str, Any]:
        def add_batch_dim(value: Any):
            if isinstance(value, torch.Tensor):
                return value.unsqueeze(0)
            return value

        return dict_apply(sample, add_batch_dim)

    def predict(self, observation: PolicyObservation) -> InferenceResult:
        if self.processor is None or self.model is None:
            raise RuntimeError("Runtime was created without model initialization.")

        raw_sample = self._build_raw_sample(observation)
        processed = self.processor.preprocess(raw_sample)
        batch = self._batchify(processed)
        batch = dict_apply(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)

        with torch.no_grad():
            batch = self.model.predict_action(batch)

        batch = dict_apply(batch, lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x)
        batch = self.processor.postprocess(batch)

        action = {key: value[0].numpy() for key, value in batch["action"].items()}
        action_flat = dict_to_array(action)
        return InferenceResult(
            action=action,
            action_flat=action_flat,
            first_action_flat=action_flat[0],
        )
