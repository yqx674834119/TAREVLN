#!/usr/bin/env bash
set -euo pipefail
VLN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export MAMBA_ROOT_PREFIX="$VLN_ROOT/cache/mamba"
eval "$("$VLN_ROOT/toolchains/micromamba/bin/micromamba" shell hook -s bash)"
set +u  # RoboStack activation scripts read optional conda variables.
micromamba activate "$VLN_ROOT/envs/ros"
set -u
export CMAKE_BUILD_PARALLEL_LEVEL=6
cd "$VLN_ROOT/deps/sysnav"
colcon build --symlink-install --packages-select serial terrain_analysis terrain_analysis_ext local_planner tare_planner \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  "-DCMAKE_CXX_FLAGS=-DEIGEN_MALLOC_ALREADY_ALIGNED=1 -DEIGEN_MAX_ALIGN_BYTES=16"
