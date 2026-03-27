#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState


@dataclass(frozen=True)
class Topic:
    name: str
    msg_type: type


class ObservationCollector:
    def __init__(self, hardware: str = "R1_LITE"):
        self.hardware = hardware
        self.dof_of_arm = 6 if hardware == "R1_LITE" else 7
        self.callback_group = ReentrantCallbackGroup()
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.node = rclpy.create_node("galaxea_observation_test")
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)

        self.image_topics = {
            "head_rgb": Topic("/hdas/camera_head/left_raw/image_raw_color/compressed", CompressedImage),
            "left_wrist_rgb": Topic("/hdas/camera_wrist_left/color/image_raw/compressed", CompressedImage),
            "right_wrist_rgb": Topic("/hdas/camera_wrist_right/color/image_raw/compressed", CompressedImage),
        }
        self.state_topics = {
            "left_ee_pose": Topic("/motion_control/pose_ee_arm_left", PoseStamped),
            "right_ee_pose": Topic("/motion_control/pose_ee_arm_right", PoseStamped),
            "left_gripper": Topic("/hdas/feedback_gripper_left", JointState),
            "right_gripper": Topic("/hdas/feedback_gripper_right", JointState),
            "left_arm": Topic("/hdas/feedback_arm_left", JointState),
            "right_arm": Topic("/hdas/feedback_arm_right", JointState),
            "torso": Topic("/hdas/feedback_torso", JointState),
            "chassis": Topic("/hdas/feedback_chassis", JointState),
        }

        self.buffers: dict[str, deque[dict[str, Any]]] = {}
        self.subscribers = []
        self._init_topics()

    def _init_topics(self) -> None:
        for key, topic in self.image_topics.items():
            self.buffers[key] = deque(maxlen=3)
            self.subscribers.append(
                self.node.create_subscription(
                    topic.msg_type,
                    topic.name,
                    lambda msg, obs_key=key: self._image_callback(obs_key, msg),
                    self.qos,
                    callback_group=self.callback_group,
                )
            )

        for key, topic in self.state_topics.items():
            self.buffers[key] = deque(maxlen=80)
            self.subscribers.append(
                self.node.create_subscription(
                    topic.msg_type,
                    topic.name,
                    lambda msg, obs_key=key: self._state_callback(obs_key, msg),
                    self.qos,
                    callback_group=self.callback_group,
                )
            )

    @staticmethod
    def _stamp_to_time(stamp: RosTime) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def _pose_to_array(msg: PoseStamped) -> np.ndarray:
        pose = msg.pose
        return np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _compressed_image_to_chw_rgb(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode compressed image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    def _append(self, key: str, timestamp: float, data: np.ndarray) -> None:
        self.buffers[key].append(
            {
                "message_time": timestamp,
                "receive_time": time.time(),
                "data": data,
            }
        )

    def _image_callback(self, key: str, msg: CompressedImage) -> None:
        self._append(key, self._stamp_to_time(msg.header.stamp), self._compressed_image_to_chw_rgb(msg.data))

    def _state_callback(self, key: str, msg: JointState | PoseStamped) -> None:
        timestamp = self._stamp_to_time(msg.header.stamp)
        if key in {"left_ee_pose", "right_ee_pose"}:
            data = self._pose_to_array(msg)
        elif key in {"left_gripper", "right_gripper"}:
            data = np.array(msg.position, dtype=np.float32)
        elif key in {"left_arm", "right_arm"}:
            data = np.array(msg.position[: self.dof_of_arm], dtype=np.float32)
        elif key == "torso":
            data = np.array(msg.position, dtype=np.float32)
        elif key == "chassis":
            data = np.array(msg.velocity[:3], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported state key: {key}")
        self._append(key, timestamp, data)

    def spin_once(self, timeout_sec: float = 0.1) -> None:
        self.executor.spin_once(timeout_sec=timeout_sec)

    @staticmethod
    def _find_nearest(buffer: deque[dict[str, Any]], target_time: float) -> dict[str, Any] | None:
        best_msg = None
        best_diff = float("inf")
        for item in reversed(buffer):
            diff = abs(item["message_time"] - target_time)
            if diff < best_diff:
                best_msg = item
                best_diff = diff
                if diff < 0.001:
                    break
        return best_msg

    def gather_observation(self) -> tuple[float, dict[str, Any]] | tuple[None, None]:
        if len(self.buffers["head_rgb"]) == 0:
            return None, None

        head_msg = self.buffers["head_rgb"][-1]
        reference_time = head_msg["message_time"]
        obs = {"images": {}, "state": {}}

        for key, buffer in self.buffers.items():
            if len(buffer) == 0:
                return None, None
            item = head_msg if key == "head_rgb" else self._find_nearest(buffer, reference_time)
            if item is None:
                return None, None
            if key in self.image_topics:
                obs["images"][key] = item["data"]
            else:
                obs["state"][key] = item["data"]
        return reference_time, obs

    def close(self) -> None:
        self.executor.shutdown()
        for sub in self.subscribers:
            self.node.destroy_subscription(sub)
        self.node.destroy_node()


def chw_rgb_to_hwc_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB image, got shape {image.shape}")
    return np.ascontiguousarray(image.transpose(1, 2, 0))


def save_observation_snapshot(
    out_dir: Path,
    obs_time: float,
    observation: dict[str, Any],
    instruction: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = {}
    for key, image in observation["images"].items():
        hwc_rgb = chw_rgb_to_hwc_rgb(image)
        path = out_dir / f"{key}.png"
        cv2.imwrite(str(path), cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR))
        image_paths[key] = str(path)

    payload = {
        "instruction": instruction,
        "idx": 0,
        "head_rgb": image_paths["head_rgb"],
        "left_wrist_rgb": image_paths["left_wrist_rgb"],
        "right_wrist_rgb": image_paths["right_wrist_rgb"],
        "robot_state": {
            "left": {
                "ee_pose": observation["state"]["left_ee_pose"].tolist(),
                "gripper": float(np.asarray(observation["state"]["left_gripper"]).reshape(-1)[0]),
            },
            "right": {
                "ee_pose": observation["state"]["right_ee_pose"].tolist(),
                "gripper": float(np.asarray(observation["state"]["right_gripper"]).reshape(-1)[0]),
            },
        },
        "raw_state": {key: np.asarray(value).tolist() for key, value in observation["state"].items()},
        "meta": {
            "obs_time": obs_time,
            "image_keys": list(observation["images"].keys()),
            "state_keys": list(observation["state"].keys()),
        },
    }

    json_path = out_dir / "observation_snapshot.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect one synchronized Galaxea observation from ROS2.")
    parser.add_argument("--output-dir", required=True, help="Directory to save PNGs and observation JSON.")
    parser.add_argument("--instruction", default="pick up the object and place it into the container")
    parser.add_argument("--hardware", default="R1_LITE", choices=["R1_LITE", "R1_PRO"])
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument("--spin-timeout-sec", type=float, default=0.05)
    args = parser.parse_args()

    rclpy.init()
    collector = ObservationCollector(hardware=args.hardware)

    start = time.time()
    try:
        while time.time() - start < args.timeout_sec:
            collector.spin_once(timeout_sec=args.spin_timeout_sec)
            obs_time, observation = collector.gather_observation()
            if observation is None:
                continue
            out_dir = Path(args.output_dir)
            json_path = save_observation_snapshot(out_dir, obs_time, observation, args.instruction)
            summary = {
                "output_dir": str(out_dir),
                "observation_json": str(json_path),
                "obs_time": obs_time,
                "image_shapes": {k: list(v.shape) for k, v in observation["images"].items()},
                "state_shapes": {k: list(np.asarray(v).shape) for k, v in observation["state"].items()},
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        print(f"Timed out after {args.timeout_sec}s without a synchronized observation.")
        return 1
    finally:
        collector.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
