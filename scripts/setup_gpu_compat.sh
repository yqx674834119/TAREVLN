#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libdir="$root/toolchains/nvidia-580.173.02/lib"
source_dir="${NVIDIA_COMPAT_SOURCE:-$HOME/.cache/nvidia-driver-compat/580.173.02/extracted}"
kernel_version="$(sed -n 's/^NVRM version:.*  \([0-9][0-9.]*\)  Release.*/\1/p' /proc/driver/nvidia/version)"
if [[ "$kernel_version" != "580.173.02" ]]; then
  echo "loaded NVIDIA module is $kernel_version, expected 580.173.02; refusing mismatched userspace" >&2
  exit 2
fi
if [[ ! -f "$source_dir/libnvidia-ml.so.580.173.02" ]]; then
  echo "matching driver userspace files are unavailable: $source_dir" >&2
  exit 2
fi
mkdir -p "$libdir"
for source_file in "$source_dir"/*.so.580.173.02; do
  target_file="$libdir/${source_file##*/}"
  if [[ -L "$target_file" ]]; then
    rm "$target_file"
  fi
  cp -a "$source_file" "$target_file"
  soname="$(readelf -d "$source_file" | sed -n 's/.*(SONAME).*\[\([^]]*\)\].*/\1/p' | head -1)"
  if [[ -n "$soname" && "$soname" != "${source_file##*/}" ]]; then
    ln -sfn "${source_file##*/}" "$libdir/$soname"
  fi
done
# Kit's CUDA interop plugins also dlopen the unversioned development names.
ln -sfn libcuda.so.1 "$libdir/libcuda.so"
ln -sfn libnvidia-ml.so.1 "$libdir/libnvidia-ml.so"
cp -a "$source_dir/nvidia-smi" "$root/toolchains/nvidia-580.173.02/nvidia-smi"
cat > "$root/toolchains/nvidia-580.173.02/nvidia_icd.json" <<EOF
{"file_format_version":"1.0.1","ICD":{"library_path":"$libdir/libGLX_nvidia.so.0","api_version":"1.3.0"}}
EOF
"$root/cross-building-floor-nav/scripts/with_gpu_compat.sh" "$root/toolchains/nvidia-580.173.02/nvidia-smi" --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$root/cross-building-floor-nav/scripts/with_gpu_compat.sh" vulkaninfo --summary 2>&1 | grep -E 'deviceName|driverInfo' | head -12
