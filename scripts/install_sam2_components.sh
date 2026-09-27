#!/usr/bin/env bash
# Project-local setup; Isaac Python and the CUDA 12.8 Torch overlay already exist.
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vln="$(dirname "$project")"
checkpoint_source="${1:-/media/yqx/Win10Pro X64/SAM/sam2.1_hiera_large.pt}"
source_root="$vln/deps/sam2"
deps="$vln/envs/sam2-deps"
revision=2b90b9f5ceec907a1c18123530e92e794ad901a4
export UV_CACHE_DIR="$vln/cache/uv" TMPDIR="$vln/tmp"
mkdir -p "$vln/assets/sam" "$UV_CACHE_DIR" "$TMPDIR" "$vln/logs"
if [[ ! -e "$source_root" ]]; then
  git clone https://github.com/facebookresearch/sam2.git "$source_root"
  git -C "$source_root" checkout --detach "$revision"
fi
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$revision" ]] || { echo 'SAM source revision differs' >&2; exit 2; }
git -C "$source_root" diff --quiet
git -C "$source_root" diff --cached --quiet
if [[ ! -e "$vln/assets/sam/sam2.1_hiera_large.pt" ]]; then
  [[ -f "$checkpoint_source" ]] || { echo 'Supply the existing SAM2.1 large checkpoint path' >&2; exit 2; }
  ln -s "$checkpoint_source" "$vln/assets/sam/sam2.1_hiera_large.pt"
fi
if [[ ! -d "$deps/iopath" || ! -d "$deps/portalocker" || ! -d "$deps/tqdm" ]]; then
  "$vln/toolchains/uv/local/bin/uv" pip install --python "$vln/envs/isaac51/bin/python" \
    --target "$deps" --no-deps iopath==0.1.10 portalocker==2.10.1 tqdm==4.70.1
fi
PYTHONPATH="$source_root:$deps" bash "$project/scripts/with_training_env.sh" "$vln/envs/isaac51/bin/python" - "$project" <<'PY'
import hashlib,importlib.metadata,json,sys
from pathlib import Path
project=Path(sys.argv[1]);vln=project.parent
expected=json.loads((project/'config/sam2-provenance.json').read_text())
actual={name:importlib.metadata.version(name) for name in expected['runtime'] if name!='python'}
for name,version in actual.items():
    if version!=expected['runtime'][name]:raise RuntimeError(f'{name} version differs: {version}')
weight=vln/expected['checkpoint_relative_to_VLN']
if hashlib.sha256(weight.read_bytes()).hexdigest()!=expected['checkpoint_sha256']:
    raise RuntimeError('SAM checkpoint SHA differs')
from sam2.sam2_image_predictor import SAM2ImagePredictor
report={'versions':actual,'checkpoint_sha256':expected['checkpoint_sha256'],
        'scope':'dependencies and weights verified; no simulation or physical robot tested'}
(vln/'logs/sam2-components.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
PY
