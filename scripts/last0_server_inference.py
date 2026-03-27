#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from galaxea_fm.last0 import Last0InferenceConfig, Last0Observation


def _dump_example_request(path: Path) -> None:
    payload = {
        "instruction": "Pick up the object and place it into the container",
        "idx": 0,
        "head_rgb": "/absolute/path/to/head_rgb.png",
        "left_wrist_rgb": "/absolute/path/to/left_wrist_rgb.png",
        "right_wrist_rgb": "/absolute/path/to/right_wrist_rgb.png",
        "robot_state": {
            "left": {
                "ee_pose": [0.45, 0.25, 0.30, 0.0, 0.0, 0.0, 1.0],
                "gripper": 0.0,
            },
            "right": {
                "ee_pose": [0.45, -0.25, 0.30, 0.0, 0.0, 0.0, 1.0],
                "gripper": 0.0,
            },
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_result(result) -> None:
    print("\nResults")
    print("=" * 70)
    print(f"Action horizon x dim: {tuple(result.action_chunk.shape)}")
    print(f"First action step: {result.first_action}")
    if result.pose_command is not None:
        print("\nPose command")
        print(f"Left target pose:  {result.pose_command.left_target_pose}")
        print(f"Right target pose: {result.pose_command.right_target_pose}")
        print(f"Left gripper: {result.pose_command.left_gripper}")
        print(f"Right gripper: {result.pose_command.right_gripper}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LaST0 inference entrypoint for Galaxea")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference/last0_galaxea.yaml",
        help="LaST0 inference config YAML.",
    )
    parser.add_argument(
        "--observation_json",
        type=str,
        default=None,
        help="Path to observation JSON.",
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
    parser.add_argument(
        "--clip_action",
        action="store_true",
        help="Clip the first action step before generating pose targets.",
    )
    args = parser.parse_args()

    if args.dump_example_json is not None:
        path = Path(args.dump_example_json)
        _dump_example_request(path)
        print(f"Example observation JSON written to: {path}")
        return 0

    if args.observation_json is None:
        raise SystemExit("--observation_json is required unless --dump_example_json is used.")

    from galaxea_fm.last0 import Last0InferenceRuntime

    runtime = Last0InferenceRuntime(Last0InferenceConfig.from_file(args.config))
    observation = Last0Observation.from_json(args.observation_json)

    print("Galaxea LaST0 Inference")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {runtime.device}")
    print(f"Model dir: {runtime.model_dir}")
    print(f"Stats path: {runtime.stats_path}")
    print(f"Loaded observation: {args.observation_json}")

    result = runtime.predict(observation, clip_action=args.clip_action)
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
