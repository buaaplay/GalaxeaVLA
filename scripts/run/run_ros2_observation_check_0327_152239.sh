#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_STAMP="$(date '+%m%d_%H%M%S')"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/ros2_observation}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_STAMP}__ros2_obs_check"
INSTRUCTION="${INSTRUCTION:-pick up the object and place it into the container}"
HARDWARE="${HARDWARE:-R1_LITE}"
TIMEOUT_SEC="${TIMEOUT_SEC:-15}"

mkdir -p "${RUN_DIR}"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

cd "${PROJECT_ROOT}"

PYTHONPATH=src python scripts/test_ros2_observation.py \
  --output-dir "${RUN_DIR}" \
  --instruction "${INSTRUCTION}" \
  --hardware "${HARDWARE}" \
  --timeout-sec "${TIMEOUT_SEC}"
