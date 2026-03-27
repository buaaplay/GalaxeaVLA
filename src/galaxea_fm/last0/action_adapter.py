from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import Last0RobotState, PoseCommand


def _normalize_quaternion(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
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


def _quaternion_multiply(lhs_xyzw: np.ndarray, rhs_xyzw: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = _normalize_quaternion(lhs_xyzw)
    x2, y2, z2, w2 = _normalize_quaternion(rhs_xyzw)
    return _normalize_quaternion(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float32,
        )
    )


@dataclass
class Last0ActionAdapter:
    max_position_delta_m: float = 0.03
    max_rotation_delta_rad: float = 0.15
    binarize_gripper: bool = True

    def split_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,):
            raise ValueError(f"Expected 14D action, got shape {action.shape}")
        return action[:7].copy(), action[7:].copy()

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        clipped = np.asarray(action, dtype=np.float32).copy()
        clipped[:3] = np.clip(clipped[:3], -self.max_position_delta_m, self.max_position_delta_m)
        clipped[3:6] = np.clip(clipped[3:6], -self.max_rotation_delta_rad, self.max_rotation_delta_rad)
        clipped[7:10] = np.clip(clipped[7:10], -self.max_position_delta_m, self.max_position_delta_m)
        clipped[10:13] = np.clip(clipped[10:13], -self.max_rotation_delta_rad, self.max_rotation_delta_rad)
        if self.binarize_gripper:
            clipped[6] = 1.0 if clipped[6] >= 0.5 else 0.0
            clipped[13] = 1.0 if clipped[13] >= 0.5 else 0.0
        return clipped

    def _compose_target_pose(self, ee_pose: np.ndarray, delta_action: np.ndarray) -> np.ndarray:
        curr_pos = np.asarray(ee_pose[:3], dtype=np.float32)
        curr_quat = _normalize_quaternion(np.asarray(ee_pose[3:], dtype=np.float32))
        dpos = np.asarray(delta_action[:3], dtype=np.float32)
        delta_quat = _rotvec_to_quaternion(np.asarray(delta_action[3:6], dtype=np.float32))

        target_pos = curr_pos + dpos
        target_quat = _quaternion_multiply(delta_quat, curr_quat)
        return np.concatenate([target_pos, target_quat], axis=0).astype(np.float32)

    def adapt(self, action: np.ndarray, robot_state: Last0RobotState, clip: bool = True) -> PoseCommand:
        processed = self.clip_action(action) if clip else np.asarray(action, dtype=np.float32)
        left_delta, right_delta = self.split_action(processed)

        return PoseCommand(
            left_target_pose=self._compose_target_pose(robot_state.left.ee_pose, left_delta),
            right_target_pose=self._compose_target_pose(robot_state.right.ee_pose, right_delta),
            left_gripper=float(left_delta[6]),
            right_gripper=float(right_delta[6]),
            left_delta_action=left_delta,
            right_delta_action=right_delta,
        )
