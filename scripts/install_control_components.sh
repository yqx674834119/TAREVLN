#!/usr/bin/env bash
# Run with nohup; all downloads/cache/envs stay in ~/VLN. No system packages.
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vln="$(dirname "$project")"
python="$vln/envs/isaac51/bin/python"
lab="$vln/deps/IsaacLab"
pin=3c6e67bb5c7ada942a6d1884ab69338f57596f77
export PIP_CACHE_DIR="$vln/cache/pip"
export TMPDIR="$vln/cache/pip-tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$vln/logs"
if [[ ! -x "$python" ]]; then echo "Install Isaac 5.1 first" >&2; exit 2; fi
if [[ ! -d "$lab/.git" ]]; then
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    git -c http.proxy= -c https.proxy= clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git "$lab"
fi
if [[ "$(git -C "$lab" rev-parse HEAD)" != "$pin" ]]; then
  echo "Isaac Lab checkout differs from pinned revision; preserving it" >&2; exit 2
fi
"$python" -m pip freeze > "$vln/logs/isaac-before-control-install.freeze"
"$python" -m pip install --no-deps setuptools==80.9.0
"$python" -m pip install --no-deps --no-build-isolation -r "$project/config/control-runtime-requirements.txt"
for component in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  "$python" -m pip install --no-deps --no-build-isolation -e "$lab/source/$component"
done
"$python" -m pip install --upgrade --target "$vln/envs/torch-cu128" \
  --index-url https://download.pytorch.org/whl/cu128 -r "$project/config/torch-cu128-requirements.txt"
PYTHONPATH="$vln/envs/torch-cu128" "$python" -m pip freeze > "$vln/logs/control-runtime.freeze"
echo "Control runtime installation finished; verify GPU arithmetic before training."
