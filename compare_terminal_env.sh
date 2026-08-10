#!/usr/bin/env bash
set -Eeuo pipefail

OUT_DIR="${1:-/tmp/wrs_env_compare}"
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="$OUT_DIR/env_${STAMP}_$$.txt"

{
  echo "=== BASIC INFO ==="
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "user=${USER:-}"
  echo "pwd=$(pwd)"
  echo "shell=${SHELL:-}"
  echo "bash_version=${BASH_VERSION:-}"
  echo "tty=$(tty 2>/dev/null || true)"
  echo

  echo "=== IMPORTANT VARIABLES ==="
  for key in \
    PATH \
    PYTHONPATH \
    LD_LIBRARY_PATH \
    DISPLAY \
    HOME \
    ROS_DISTRO \
    ROS_VERSION \
    ROS_DOMAIN_ID \
    RMW_IMPLEMENTATION \
    AMENT_PREFIX_PATH \
    COLCON_PREFIX_PATH \
    CMAKE_PREFIX_PATH \
    PYTHONNOUSERSITE \
    VIRTUAL_ENV
  do
    printf '%s=%s\n' "$key" "${!key-}"
  done
  echo

  echo "=== PYTHON ==="
  command -v python || true
  command -v python3 || true
  python --version 2>&1 || true
  python3 --version 2>&1 || true
  if command -v python >/dev/null 2>&1; then
    python - <<'PY'
import sys, os
print('sys.executable=', sys.executable)
print('sys.version=', sys.version.replace('\n', ' '))
print('sys.path[0:10]=', sys.path[:10])
print('cwd=', os.getcwd())
PY
  fi
  echo

  echo "=== ROS SETUP CHECK ==="
  for f in \
    /opt/ros/humble/setup.bash \
    /home/nvidia/wrs_main/wrs-virtual/install/setup.bash
  do
    if [[ -f "$f" ]]; then
      echo "FOUND $f"
    else
      echo "MISSING $f"
    fi
  done
  echo

  echo "=== PROJECT CHECK ==="
  for f in \
    /home/nvidia/wrs_main/wrs-virtual/.venv_linux/bin/python \
    /home/nvidia/wrs_main/wrs-virtual/wrs/Fruit_picker_project/FLASK/run_server.py \
    /home/nvidia/wrs_main/wrs-virtual/wrs/Fruit_picker_project/main.py \
    /home/nvidia/wrs_main/wrs-virtual/wrs/Fruit_picker_project/Weight/best.pt \
    /home/nvidia/wrs_main/wrs-virtual/wrs/Pepper/best.pt \
    /home/nvidia/wrs_main/wrs-virtual/install/setup.bash
  do
    if [[ -e "$f" ]]; then
      echo "FOUND $f"
    else
      echo "MISSING $f"
    fi
  done
  echo

  echo "=== NETWORK CHECK ==="
  if command -v curl >/dev/null 2>&1; then
    curl -m 3 -sS -o /dev/null -w 'curl_health_%{http_code}\n' http://127.0.0.1:5000/health || true
    curl -m 3 -sS -o /dev/null -w 'curl_lift_%{http_code}\n' http://169.254.128.2:5480/woosh/ros/CallAction || true
  else
    echo "curl not found"
  fi
  echo

  echo "=== PYCHARM HINTS ==="
  env | sort | grep -E '^(PYCHARM|IDEA|JETBRAINS|TERM_SESSION|COLORTERM|DBUS|XDG_SESSION|SSH_|VIRTUAL_ENV|ROS_|AMENT_|CMAKE_PREFIX_PATH|COLCON_PREFIX_PATH|PYTHONPATH|LD_LIBRARY_PATH|DISPLAY)=' || true
} > "$OUT_FILE"

echo "$OUT_FILE"
