#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SAMPLE_PATH="${1:-${SAMPLE_PATH:-}}"
if [[ -z "${SAMPLE_PATH}" ]]; then
  echo "Usage: bash $(basename "$0") /absolute/path/to/last0_train.jsonl"
  echo "Or set SAMPLE_PATH=/absolute/path/to/last0_train.jsonl"
  exit 1
fi

RUN_STAMP="$(date '+%m%d_%H%M%S')"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
LAST0_ROOT="${LAST0_ROOT:-/home/robot/zhy/last0}"
MODEL_ROOT="${MODEL_ROOT:-/home/robot/weights/MyData_finetune_last0_checkpoint-13-171570/tfmr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/last0_offline}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_STAMP}__last0_offline_replay"
OBS_JSON="${RUN_DIR}/observation_from_sample.json"
RESULT_JSON="${RUN_DIR}/last0_result.json"
POSE_JSON="${RUN_DIR}/last0_pose_command.json"
CONFIG_PATH="${RUN_DIR}/last0_runtime.yaml"
CONSOLE_LOG="${RUN_DIR}/console.log"
CLIP_ACTION="${CLIP_ACTION:-1}"

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

if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
  set -u
fi

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Project root: ${PROJECT_ROOT}" | tee "${CONSOLE_LOG}"
echo "Sample path: ${SAMPLE_PATH}" | tee -a "${CONSOLE_LOG}"
echo "Sample index: ${SAMPLE_INDEX}" | tee -a "${CONSOLE_LOG}"
echo "Model root: ${MODEL_ROOT}" | tee -a "${CONSOLE_LOG}"
echo "Run dir: ${RUN_DIR}" | tee -a "${CONSOLE_LOG}"

PYTHONPATH=src "${PYTHON_BIN}" scripts/last0_prepare_observation.py \
  --input "${SAMPLE_PATH}" \
  --index "${SAMPLE_INDEX}" \
  --output "${OBS_JSON}" 2>&1 | tee -a "${CONSOLE_LOG}"

CLIP_FLAG=()
if [[ "${CLIP_ACTION}" == "1" ]]; then
  CLIP_FLAG+=(--clip_action)
fi

PYTHONPATH=src "${PYTHON_BIN}" scripts/last0_server_inference.py \
  --config "${CONFIG_PATH}" \
  --observation_json "${OBS_JSON}" \
  --save_result_json "${RESULT_JSON}" \
  "${CLIP_FLAG[@]}" 2>&1 | tee -a "${CONSOLE_LOG}"

PYTHONPATH=src "${PYTHON_BIN}" scripts/last0_robot_bridge.py \
  --observation_json "${OBS_JSON}" \
  --action_json "${RESULT_JSON}" \
  --save_result_json "${POSE_JSON}" 2>&1 | tee -a "${CONSOLE_LOG}"

echo "Prepared observation: ${OBS_JSON}" | tee -a "${CONSOLE_LOG}"
echo "Inference result: ${RESULT_JSON}" | tee -a "${CONSOLE_LOG}"
echo "Pose command: ${POSE_JSON}" | tee -a "${CONSOLE_LOG}"
