#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OBS_JSON="${1:-${OBS_JSON:-}}"
if [[ -z "${OBS_JSON}" ]]; then
  echo "Usage: bash $(basename "$0") /absolute/path/to/observation.json"
  echo "Or set OBS_JSON=/absolute/path/to/observation.json"
  exit 1
fi

RUN_STAMP="$(date '+%m%d_%H%M%S')"
LAST0_ROOT="${LAST0_ROOT:-/home/robot/zhy/last0}"
MODEL_ROOT="${MODEL_ROOT:-/home/robot/weights/MyData_finetune_last0_checkpoint-13-171570/tfmr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/last0}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_STAMP}__galaxea_real__last0_ckpt13"
CONFIG_PATH="${RUN_DIR}/last0_runtime.yaml"
RESULT_JSON="${RUN_DIR}/last0_result.json"
POSE_JSON="${RUN_DIR}/last0_pose_command.json"
CONSOLE_LOG="${RUN_DIR}/console.log"
CLIP_ACTION="${CLIP_ACTION:-1}"
RUN_BRIDGE="${RUN_BRIDGE:-1}"

mkdir -p "${RUN_DIR}"

cat > "${CONFIG_PATH}" <<EOF
last0_root: ${LAST0_ROOT}
model_root: ${MODEL_ROOT}
stats_path: null
device: cuda
cuda_id: 0
action_chunk: 16
latent_size: 8
use_latent: true
use_proprio: false
seed: 42
EOF

echo "Project root: ${PROJECT_ROOT}" | tee "${CONSOLE_LOG}"
echo "Run dir: ${RUN_DIR}" | tee -a "${CONSOLE_LOG}"
echo "Observation JSON: ${OBS_JSON}" | tee -a "${CONSOLE_LOG}"
echo "LaST0 root: ${LAST0_ROOT}" | tee -a "${CONSOLE_LOG}"
echo "Model root: ${MODEL_ROOT}" | tee -a "${CONSOLE_LOG}"
echo "Config path: ${CONFIG_PATH}" | tee -a "${CONSOLE_LOG}"

CLIP_FLAG=()
if [[ "${CLIP_ACTION}" == "1" ]]; then
  CLIP_FLAG+=(--clip_action)
fi

cd "${PROJECT_ROOT}"

PYTHONPATH=src python scripts/last0_server_inference.py \
  --config "${CONFIG_PATH}" \
  --observation_json "${OBS_JSON}" \
  --save_result_json "${RESULT_JSON}" \
  "${CLIP_FLAG[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

if [[ "${RUN_BRIDGE}" == "1" ]]; then
  PYTHONPATH=src python scripts/last0_robot_bridge.py \
    --observation_json "${OBS_JSON}" \
    --action_json "${RESULT_JSON}" \
    --save_result_json "${POSE_JSON}" 2>&1 | tee -a "${CONSOLE_LOG}"
fi

echo "Inference result saved to: ${RESULT_JSON}" | tee -a "${CONSOLE_LOG}"
if [[ "${RUN_BRIDGE}" == "1" ]]; then
  echo "Pose command saved to: ${POSE_JSON}" | tee -a "${CONSOLE_LOG}"
fi
