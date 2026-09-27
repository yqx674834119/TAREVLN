#!/usr/bin/env bash
set -euo pipefail
VLN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$VLN_ROOT/deps/basis_universal"
REVISION=433888a9736a8802fe9d17ae7f489b4cf7a45556
if [ ! -d "$SOURCE" ]; then
  git clone --depth 1 --filter=blob:none --sparse --branch v2_0_2 https://github.com/BinomialLLC/basis_universal.git "$SOURCE"
  git -C "$SOURCE" sparse-checkout set encoder transcoder zstd
fi
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$REVISION"
cmake -S "$SOURCE" -B "$SOURCE/build" -DBASISU_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$SOURCE/build" --target basisu -j8
