#!/usr/bin/env bash
# Launch the ADAS instrument cluster on a Jetson (or any CUDA Linux host).
#
#   ./scripts/run_cluster.sh --video drive.mp4
#   ./scripts/run_cluster.sh --camera 0 --speed obd --fullscreen
#   ./scripts/run_cluster.sh --sim
#
# All arguments go straight to scripts/run_pyside6_app.py (see --help).
# Set ADAS_MAX_PERF=1 to lock the Jetson to its max power mode and clocks
# first (needs sudo; the setting lasts until reboot).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# No cd: relative --video paths stay relative to where you ran this.

if [[ -f "$ROOT/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

if [[ "${ADAS_MAX_PERF:-0}" == "1" ]] && command -v nvpmodel >/dev/null 2>&1; then
  sudo nvpmodel -m 0 || true
  sudo jetson_clocks || true
fi

# Qt Quick 3D on Jetson: OpenGL RHI, X11 (xcb) unless a platform is forced.
export QSG_RHI_BACKEND="${QSG_RHI_BACKEND:-opengl}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export DISPLAY="${DISPLAY:-:0}"
export PYTHONUNBUFFERED=1

exec python "$ROOT/scripts/run_pyside6_app.py" "$@"
