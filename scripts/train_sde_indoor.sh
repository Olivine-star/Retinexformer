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

GPU_IDS="${1:-0}"
if [ "$#" -gt 0 ]; then shift; fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python3 basicsr/train.py \
  --opt Options/RetinexFormer_SDE_indoor.yml \
  "$@"
