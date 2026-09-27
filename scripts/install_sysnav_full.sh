#!/usr/bin/env bash
# Isolated Python overlay; never modifies the ROS or Isaac environments.
set -euo pipefail
VLN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SYSNAV_ENV="$VLN_ROOT/envs/sysnav-full"
export PIP_CACHE_DIR="$VLN_ROOT/cache/pip"
export HF_HOME="$VLN_ROOT/cache/huggingface"
export TORCH_HOME="$VLN_ROOT/cache/torch"
mkdir -p "$VLN_ROOT/logs"
"$VLN_ROOT/envs/ros/bin/python" -m venv --system-site-packages "$SYSNAV_ENV"
"$SYSNAV_ENV/bin/python" -m pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128
"$SYSNAV_ENV/bin/python" -m pip install 'numpy<2' 'opencv-python<4.12' open3d scikit-learn ultralytics openai supervision shapely lap line_profiler 'rerun-sdk<0.24' hydra-core iopath timm ftfy regex
SAM2_BUILD_CUDA=0 "$SYSNAV_ENV/bin/python" -m pip install --no-build-isolation --no-deps -e "$VLN_ROOT/deps/sam2"
"$SYSNAV_ENV/bin/python" -m pip freeze > "$VLN_ROOT/logs/sysnav-full-python.lock"
