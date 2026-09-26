#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$root"/{cache,envs,logs,tmp,toolchains}
export PIP_CACHE_DIR="$root/cache/pip"
export UV_CACHE_DIR="$root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$root/toolchains/python"
export TMPDIR="$root/tmp"
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
printf 'started: %s\n' "$(date -Is)"
printf 'status: installing uv\n' > "$root/logs/isaac51.status"
python3 -m pip install --prefix "$root/toolchains/uv" uv
uv_bin="$root/toolchains/uv/local/bin/uv"
"$uv_bin" --version
printf 'status: installing Python 3.11\n' > "$root/logs/isaac51.status"
"$uv_bin" python install 3.11
printf 'status: creating environment\n' > "$root/logs/isaac51.status"
"$uv_bin" venv --python 3.11 --seed "$root/envs/isaac51"
printf 'status: downloading and installing Isaac Sim 5.1.0\n' > "$root/logs/isaac51.status"
"$root/envs/isaac51/bin/python" -m pip install --timeout 120 --retries 5 --progress-bar off \
  'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
"$root/envs/isaac51/bin/python" -m pip show isaacsim
printf 'status: installed; runtime/EULA/GPU scene validation pending\n' > "$root/logs/isaac51.status"
date -Is > "$root/logs/isaac51.installed"
printf 'completed: %s\n' "$(date -Is)"
