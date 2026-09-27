#!/usr/bin/env bash
# Run this in the background; all runtimes and caches are project-local.
set -eo pipefail
VLN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$VLN_ROOT/toolchains/micromamba" "$VLN_ROOT/cache" "$VLN_ROOT/logs"
MAMBA_BIN="$VLN_ROOT/toolchains/micromamba/bin/micromamba"
if ! test -x "$MAMBA_BIN"; then
  curl -fLsS https://micro.mamba.pm/api/micromamba/linux-64/2.9.0 | tar -xj -C "$VLN_ROOT/toolchains/micromamba" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$VLN_ROOT/cache/mamba"
export MAMBA_USE_SHARDED_REPODATA=false
"$MAMBA_BIN" create -y -p "$VLN_ROOT/envs/ros" -f "$VLN_ROOT/cross-building-floor-nav/config/ros-humble-linux-64.lock"
"$VLN_ROOT/envs/usdtools/bin/python" "$VLN_ROOT/cross-building-floor-nav/scripts/fetch_usd_scene.py"
