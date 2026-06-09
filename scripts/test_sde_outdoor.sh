#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/media/hzho0442/Project/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/media/hzho0442/Project/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda was not found. Please activate the light environment manually." >&2
  exit 1
fi

conda activate light

WEIGHTS="${1:-experiments/RetinexFormer_SDE_outdoor/models/net_g_latest.pth}"
GPU_IDS="${2:-0}"
OUTPUT_DIR="${3:-results/SDE_outdoor/enhanced}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python3 Enhancement/test_sde.py \
  --opt Options/RetinexFormer_SDE_outdoor.yml \
  --weights "${WEIGHTS}" \
  --output_dir "${OUTPUT_DIR}" \
  --gpus "${GPU_IDS}" \
  "$@"
