#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vln="$(dirname "$project")"
export UV_CACHE_DIR="$vln/cache/uv"
export PIP_CACHE_DIR="$vln/cache/pip"
export TMPDIR="$vln/tmp"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" "$vln/logs"
uv="$vln/toolchains/uv/local/bin/uv"
if [[ ! -x "$vln/envs/odometry/bin/python" ]]; then
  "$uv" venv --python "$vln/envs/ros/bin/python" --seed "$vln/envs/odometry"
fi
"$uv" pip install --python "$vln/envs/odometry/bin/python" -r "$project/config/odometry-lock.txt"
"$vln/envs/odometry/bin/python" -m pip freeze > "$vln/logs/odometry.freeze"
