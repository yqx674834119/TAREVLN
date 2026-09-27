#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
torch_root="$(dirname "$project")/envs/torch-cu128"
if [[ ! -f "$torch_root/torch/__init__.py" ]]; then
  echo "The isolated CUDA 12.8 PyTorch install is missing or incomplete" >&2
  exit 2
fi
export PYTHONPATH="$torch_root${PYTHONPATH:+:$PYTHONPATH}"
exec bash "$project/scripts/with_gpu_compat.sh" "$@"
