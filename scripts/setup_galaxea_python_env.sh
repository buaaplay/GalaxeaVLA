#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-galaxea}"
WITH_LAST0=0

for arg in "$@"; do
  case "$arg" in
    --with-last0)
      WITH_LAST0=1
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: bash scripts/setup_galaxea_python_env.sh [--with-last0]"
      exit 1
      ;;
  esac
done

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found under ~/miniconda3 or ~/anaconda3"
  exit 1
fi

conda activate "${ENV_NAME}"

echo "[1/3] Python version"
python --version

echo "[2/3] Installing observation pipeline dependencies"
python -m pip install --upgrade pip
python -m pip install \
  numpy \
  opencv-python \
  pillow \
  pyyaml \
  toml \
  omegaconf \
  hydra-core \
  loguru

if [[ "${WITH_LAST0}" == "1" ]]; then
  echo "[3/3] Installing LaST0 inference dependencies"
  python -m pip install \
    torch \
    torchvision \
    torchaudio \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    scipy \
    einops \
    tqdm
else
  echo "[3/3] Skipping LaST0 inference dependencies. Re-run with --with-last0 if needed."
fi

echo
echo "Environment ready in conda env: ${ENV_NAME}"
echo "Observation pipeline test can now use:"
echo "  source /opt/ros/humble/setup.bash"
echo "  conda activate ${ENV_NAME}"
