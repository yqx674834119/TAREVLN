#!/usr/bin/env bash
set -euo pipefail
vln_root="${VLN_ROOT:-$HOME/VLN}"
uv_bin="$vln_root/toolchains/uv/local/bin/uv"
isaac_python="$vln_root/envs/isaac51/bin/python"
if ! command -v bwrap >/dev/null; then
  echo "Missing bubblewrap: the isolated P1 policy requires bwrap; no unsandboxed fallback." >&2
  exit 1
fi
test -x "$uv_bin"
test -x "$isaac_python"
if [[ ! -x "$vln_root/envs/usdtools/bin/python" ]]; then
  "$uv_bin" venv --python "$isaac_python" "$vln_root/envs/usdtools"
fi
"$uv_bin" pip install --python "$vln_root/envs/usdtools/bin/python" \
  usd-core==25.11 pillow==12.3.0 qrcode==8.2
# Keep Kit's bundled USD intact: never install usd-core into the Isaac env.
"$uv_bin" pip install --python "$isaac_python" zxing-cpp==3.1.1 imageio imageio-ffmpeg==0.6.0
python3 "$(dirname "$0")/fetch_p1_assets.py" --root "$vln_root/assets"
