#!/usr/bin/env bash
# Diagnostic build only; run build_sysnav.sh to return to Release.
set -eo pipefail
VLN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export MAMBA_ROOT_PREFIX="$VLN_ROOT/cache/mamba"
export CMAKE_BUILD_PARALLEL_LEVEL=6
cd "$VLN_ROOT/deps/sysnav"
"$VLN_ROOT/toolchains/micromamba/bin/micromamba" run -p "$VLN_ROOT/envs/ros" \
 colcon build --symlink-install --packages-select tare_planner --cmake-args \
 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
 "-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer -include $VLN_ROOT/cross-building-floor-nav/config/asan-ortools-abi.h -DEIGEN_MALLOC_ALREADY_ALIGNED=1 -DEIGEN_MAX_ALIGN_BYTES=16" \
 "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O1 -g1 -DNDEBUG"
