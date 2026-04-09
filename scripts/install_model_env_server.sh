#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_NAME="${ENV_NAME:-model}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.4.0}"
BUILD_ROSA_CPU_EXTENSION="${BUILD_ROSA_CPU_EXTENSION:-1}"
FORCE_ENV_UPDATE="${FORCE_ENV_UPDATE:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ROSA] conda not found. Please install conda or mamba first." >&2
  exit 1
fi

echo "[ROSA] repo: ${ROOT_DIR}"
echo "[ROSA] env name: ${ENV_NAME}"
echo "[ROSA] pytorch index: ${PYTORCH_INDEX_URL}"
echo "[ROSA] torch/vision/audio: ${TORCH_VERSION} / ${TORCHVISION_VERSION} / ${TORCHAUDIO_VERSION}"
echo "[ROSA] transformers: ${TRANSFORMERS_VERSION}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${FORCE_ENV_UPDATE}" == "1" ]]; then
    echo "[ROSA] updating existing environment ${ENV_NAME}..."
    conda env update -n "${ENV_NAME}" -f "${ROOT_DIR}/env/model_server_environment.yml" --prune
  else
    echo "[ROSA] environment ${ENV_NAME} already exists, skip conda env create."
  fi
else
  echo "[ROSA] creating conda environment ${ENV_NAME}..."
  conda env create -n "${ENV_NAME}" -f "${ROOT_DIR}/env/model_server_environment.yml"
fi

echo "[ROSA] upgrading pip tooling..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

echo "[ROSA] installing torch stack..."
conda run -n "${ENV_NAME}" python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

echo "[ROSA] installing transformers..."
conda run -n "${ENV_NAME}" python -m pip install "transformers==${TRANSFORMERS_VERSION}"

echo "[ROSA] installing remaining pip packages..."
conda run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/env/model_server_pip_requirements.txt"

if [[ "${BUILD_ROSA_CPU_EXTENSION}" == "1" ]]; then
  echo "[ROSA] building compiled CPU address extension..."
  conda run -n "${ENV_NAME}" python -c "from rosa_cpp_extension import build_rosa_sam_cpu_extension; build_rosa_sam_cpu_extension(verbose=True, force_rebuild=False)"
fi

echo "[ROSA] verifying environment..."
conda run -n "${ENV_NAME}" python -c "import sys, torch, transformers; print('python', sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__)"

echo "[ROSA] done."
