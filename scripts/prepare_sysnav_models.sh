#!/usr/bin/env bash
set -euo pipefail
VLN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PIP_CACHE_DIR="$VLN_ROOT/cache/pip"
export HF_HOME="$VLN_ROOT/cache/huggingface"
export YOLO_CONFIG_DIR="$VLN_ROOT/cache/ultralytics"
export TORCH_HOME="$VLN_ROOT/cache/torch"
mkdir -p "$VLN_ROOT/assets/sysnav" "$VLN_ROOT/assets/sam"
if [[ ! -d "$VLN_ROOT/deps/ultralytics-clip/.git" ]]; then
  git -c http.proxy= -c https.proxy= clone --depth 1 https://github.com/ultralytics/CLIP.git "$VLN_ROOT/deps/ultralytics-clip"
fi
"$VLN_ROOT/envs/sysnav-full/bin/python" -m pip install --no-deps "$VLN_ROOT/deps/ultralytics-clip"
for spec in \
  'sysnav/yoloe-26x-seg.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt' \
  'sam/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt'; do
  read -r relative url <<< "$spec"
  target="$VLN_ROOT/assets/$relative"
  if [[ ! -f "$target" ]]; then
    curl -fL --retry 3 "$url" -o "$target.part"
    mv "$target.part" "$target"
  fi
done
CUDA_VISIBLE_DEVICES=2 bash "$VLN_ROOT/cross-building-floor-nav/scripts/with_gpu_compat.sh" \
  "$VLN_ROOT/envs/sysnav-full/bin/python" -u "$VLN_ROOT/cross-building-floor-nav/scripts/prepare_sysnav_models.py"
