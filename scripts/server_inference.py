#!/usr/bin/env python3
"""
Reusable inference entrypoint for GalaxeaVLA.

This script supports two modes:
1. Dummy mode for smoke testing the inference stack wiring.
2. JSON observation mode for later integration with robot middleware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxea_fm.inference import GalaxeaInferenceRuntime, InferenceRuntimeConfig, PolicyObservation


def _make_dummy_observation(runtime: GalaxeaInferenceRuntime) -> PolicyObservation:
    obs_steps = runtime.cfg.data.obs_size

    images = {}
    for meta in runtime.cfg.data.shape_meta.images:
        c, h, w = meta.raw_shape
        images[meta.key] = torch.randint(
            low=0,
            high=255,
            size=(obs_steps, c, h, w),
            dtype=torch.uint8,
        )

    state = {}
    for meta in runtime.cfg.data.shape_meta.state:
        state[meta.key] = torch.zeros((obs_steps, meta.raw_shape), dtype=torch.float32)

    return PolicyObservation(
        instruction="Pick up the red cup and place it on the table",
        coarse_task="Pick and place",
        images=images,
        state=state,
        idx=0,
    )


def _dump_example_request(path: Path, runtime: GalaxeaInferenceRuntime) -> None:
    obs_steps = runtime.cfg.data.obs_size
    payload = {
        "instruction": "Pick up the red cup and place it on the table",
        "coarse_task": "Pick and place",
        "idx": 0,
        "images": {
            meta.key: f"/absolute/path/to/{meta.key}.png"
            for meta in runtime.cfg.data.shape_meta.images
        },
        "state": {
            meta.key: [[0.0] * int(meta.raw_shape) for _ in range(obs_steps)]
            for meta in runtime.cfg.data.shape_meta.state
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_result(result) -> None:
    print("\nResults")
    print("=" * 70)
    print(f"Action horizon x dim: {tuple(result.action_flat.shape)}")
    print(f"First action step (flat): {result.first_action_flat}")
    print("\nAction by field:")
    for key, value in result.action.items():
        print(f"  {key}: shape={tuple(value.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="GalaxeaVLA inference entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference/r1lite_g0plus_pytorch.yaml",
        help="Inference runtime config YAML.",
    )
    parser.add_argument(
        "--observation_json",
        type=str,
        default=None,
        help="Path to observation JSON. If omitted, a dummy observation is used.",
    )
    parser.add_argument(
        "--dump_example_json",
        type=str,
        default=None,
        help="Write an example observation JSON and exit.",
    )
    parser.add_argument(
        "--save_result_json",
        type=str,
        default=None,
        help="Optional path to save inference output JSON.",
    )
    args = parser.parse_args()

    initialize_model = args.dump_example_json is None
    runtime = GalaxeaInferenceRuntime(
        InferenceRuntimeConfig.from_file(args.config),
        initialize_model=initialize_model,
    )

    print("GalaxeaVLA Inference")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {runtime.device}")
    print(f"Checkpoint: {runtime.config.checkpoint_path}")
    print(f"Dataset stats: {runtime.config.dataset_stats_path}")
    print(f"PaliGemma: {runtime.config.paligemma_path}")

    if args.dump_example_json is not None:
        path = Path(args.dump_example_json)
        _dump_example_request(path, runtime)
        print(f"\nExample observation JSON written to: {path}")
        return 0

    if args.observation_json is not None:
        observation = PolicyObservation.from_json(args.observation_json)
        print(f"\nLoaded observation from: {args.observation_json}")
    else:
        observation = _make_dummy_observation(runtime)
        print("\nUsing dummy observation.")

    result = runtime.predict(observation)
    _print_result(result)

    if args.save_result_json is not None:
        out_path = Path(args.save_result_json)
        out_path.write_text(
            json.dumps(result.to_serializable(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved result JSON to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
