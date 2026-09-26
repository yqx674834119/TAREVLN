#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
uv_bin="$root/toolchains/uv/local/bin/uv"
mkdir -p "$root"/{cache,envs,logs,tmp}
export UV_CACHE_DIR="$root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$root/toolchains/python"
export MENAGERIE_CACHE_DIR="$root/cache/mujoco_menagerie"
export TMPDIR="$root/tmp"
printf 'status: creating MuJoCo environment\n' > "$root/logs/mujoco-go2.status"
"$uv_bin" venv --python 3.11 "$root/envs/mujoco-go2"
printf 'status: downloading MuJoCo and pinned Go2 package\n' > "$root/logs/mujoco-go2.status"
"$uv_bin" pip install --python "$root/envs/mujoco-go2/bin/python" \
  'mujoco==3.7.0' 'mujoco-menagerie==2026.9.2' 'pillow>=11,<13'
printf 'status: downloading and compiling Go2 model\n' > "$root/logs/mujoco-go2.status"
"$root/envs/mujoco-go2/bin/python" - <<'PY'
import mujoco_menagerie as mm
mm.prefetch(['unitree_go2'])
robot = mm.get('unitree_go2')
model = mm.load('unitree_go2')
print(f'Go2 model: license={robot.license} oid={robot.oid} nq={model.nq} nv={model.nv}')
PY
printf 'status: installed; scene and locomotion validation pending\n' > "$root/logs/mujoco-go2.status"
date -Is > "$root/logs/mujoco-go2.installed"
