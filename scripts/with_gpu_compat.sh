#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libdir="$root/toolchains/nvidia-580.173.02/lib"
if [[ ! -e "$libdir/libnvidia-ml.so.1" ]]; then
  echo "run scripts/setup_gpu_compat.sh first" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$root/toolchains/nvidia-580.173.02/nvidia_icd.json"
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
exec "$@"

