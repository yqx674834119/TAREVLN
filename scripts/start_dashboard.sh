#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Use the existing project user's xrdp desktop; never start/change a display server.
if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X10 && -f "$HOME/.Xauthority" ]]; then
  if DISPLAY=:10 XAUTHORITY="$HOME/.Xauthority" xdpyinfo >/dev/null 2>&1; then
    export DISPLAY=:10 XAUTHORITY="$HOME/.Xauthority"
  fi
fi
exec python3 -u scripts/dashboard/server.py "$@"
