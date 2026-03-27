#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _normalize_quaternion(quat_xyzw: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat_xyzw / norm).astype(np.float32)


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float32)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = rotvec / theta
    half = theta * 0.5
    sin_half = np.sin(half)
    cos_half = np.cos(half)
    return _normalize_quaternion(
        np.array(
            [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half],
            dtype=np.float32,
        )
    )


def _load_items(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        items = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise TypeError(f"Unsupported input payload type: {type(data)}")


def _state_to_robot_state(state: list[float]) -> dict:
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape == (16,):
        left_pos = arr[0:3]
        left_rotvec = arr[3:6]
        left_gripper = float(arr[6])
        right_pos = arr[8:11]
        right_rotvec = arr[11:14]
        right_gripper = float(arr[14])

        return {
            "left": {
                "ee_pose": np.concatenate([left_pos, _rotvec_to_quaternion(left_rotvec)]).astype(np.float32).tolist(),
                "gripper": left_gripper,
            },
            "right": {
                "ee_pose": np.concatenate([right_pos, _rotvec_to_quaternion(right_rotvec)]).astype(np.float32).tolist(),
                "gripper": right_gripper,
            },
        }

    if arr.shape == (7,):
        right_pos = arr[0:3]
        right_rotvec = arr[3:6]
        right_gripper = float(arr[6])
        default_left = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return {
            "left": {
                "ee_pose": default_left.tolist(),
                "gripper": 0.0,
            },
            "right": {
                "ee_pose": np.concatenate([right_pos, _rotvec_to_quaternion(right_rotvec)]).astype(np.float32).tolist(),
                "gripper": right_gripper,
            },
        }

    raise ValueError(f"Expected 7D or 16D state, got shape {arr.shape}")


def _build_observation(item: dict) -> dict:
    slow = item.get("input_image_slow", [])
    fast = item.get("input_image_fast", [])
    if len(slow) < 1 or len(fast) < 1:
        raise ValueError("Sample must contain at least one slow image and one fast image.")

    left_wrist = fast[0]
    right_wrist = fast[1] if len(fast) > 1 else fast[0]

    obs = {
        "instruction": item["input_prompt"],
        "idx": 0,
        "head_rgb": slow[0],
        "left_wrist_rgb": left_wrist,
        "right_wrist_rgb": right_wrist,
    }
    if "state" in item:
        obs["robot_state"] = _state_to_robot_state(item["state"])
    return obs


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a LaST0 training sample into an observation JSON.")
    parser.add_argument("--input", required=True, help="Path to train.json, train.jsonl, or a single sample json.")
    parser.add_argument("--index", type=int, default=0, help="Sample index to extract when input is a list/jsonl.")
    parser.add_argument("--output", required=True, help="Path to write observation JSON.")
    args = parser.parse_args()

    items = _load_items(Path(args.input))
    if not items:
        raise SystemExit("No items found in input file.")
    if args.index < 0 or args.index >= len(items):
        raise SystemExit(f"--index {args.index} out of range for {len(items)} items.")

    observation = _build_observation(items[args.index])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(observation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote observation JSON to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
