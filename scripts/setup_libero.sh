#!/usr/bin/env bash
# LIBERO evaluation client. The venv is LIBERO-evaluation/.venv-libero.
# The policy server stays in the repo-root .venv from setup_train_server.sh.
#
# Python 3.8.13 and torch 1.11.0+cu113 are the versions LIBERO pins.
# That torch build has no kernel for compute capability 8.9 (4090).
# This client only runs the simulator and sends observations to the server.
#
#   bash scripts/setup_libero.sh
#   source LIBERO-evaluation/.venv-libero/bin/activate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT="$ROOT/LIBERO-evaluation"
VENV="$CLIENT/.venv-libero"
LIBERO="$CLIENT/LIBERO"
PY_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
WHEEL_DIR="${WHEEL_DIR:-$HOME/.cache/evo-depth/torch-cu113}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export DEBIAN_FRONTEND=noninteractive

log() { printf '\n==> %s\n' "$*"; }
domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}
uvpip() {
  domestic uv pip install --python "$VENV/bin/python" "$@"
}
install_torch() {
  log "Install torch 1.11.0+cu113"
  mkdir -p "$WHEEL_DIR"
  local file url
  # curl keeps the cluster proxy. uv's own fetch of these wheels is much slower.
  while read -r file url; do
    if [[ ! -s "$WHEEL_DIR/$file" ]]; then
      log "Download ${file}"
      curl -fL --retry 3 --retry-delay 2 -o "$WHEEL_DIR/$file.partial" "$url"
      mv "$WHEEL_DIR/$file.partial" "$WHEEL_DIR/$file"
    fi
  done <<'EOF'
torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl
torchvision-0.12.0+cu113-cp38-cp38-linux_x86_64.whl https://download.pytorch.org/whl/cu113/torchvision-0.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl
torchaudio-0.11.0+cu113-cp38-cp38-linux_x86_64.whl https://download.pytorch.org/whl/cu113/torchaudio-0.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl
EOF
  uvpip --index-url "$PY_INDEX" \
    "$WHEEL_DIR/torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl" \
    "$WHEEL_DIR/torchvision-0.12.0+cu113-cp38-cp38-linux_x86_64.whl" \
    "$WHEEL_DIR/torchaudio-0.11.0+cu113-cp38-cp38-linux_x86_64.whl"
}

if ! command -v uv >/dev/null 2>&1; then
  log "Install uv"
  for py in python3 python3.12 python3.11 python3.10 \
            /usr/local/miniconda3/envs/py312/bin/python; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -m pip --version >/dev/null 2>&1; then
      domestic "$py" -m pip install -i "$PY_INDEX" uv
      break
    fi
  done
  command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
fi

if [[ ! -d "$LIBERO/.git" ]]; then
  log "Clone LIBERO"
  rm -rf "$LIBERO"
  domestic git clone "https://gh-proxy.com/https://github.com/Lifelong-Robot-Learning/LIBERO.git" "$LIBERO" \
    || git clone "https://github.com/Lifelong-Robot-Learning/LIBERO.git" "$LIBERO"
fi

if command -v apt-get >/dev/null 2>&1; then
  log "Install system libraries used by opencv and mujoco"
  apt-get update -qq
  apt-get install -y -qq build-essential cmake \
    libgl1 libglib2.0-0 libegl1 libosmesa6 libglew2.2 libglfw3 patchelf
fi

if ! domestic uv python find 3.8.13 >/dev/null 2>&1; then
  domestic uv python install 3.8.13 \
    --mirror "https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download" \
    || uv python install 3.8.13
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  log "Create Python 3.8.13 venv at ${VENV}"
  domestic uv venv "$VENV" --python 3.8.13
else
  log "venv already exists"
fi

install_torch

log "Install LIBERO requirements"
uvpip --index-url "$PY_INDEX" -r "$LIBERO/requirements.txt"

log "Reinstall the pinned torch in case requirements replaced it"
install_torch

log "Install LIBERO, websockets, and imageio"
# compat mode puts the repo root on sys.path. The package has no top-level __init__.py,
# so uv's default editable finder installs an empty mapping and import libero fails.
uvpip --index-url "$PY_INDEX" --config-settings editable_mode=compat \
  -e "$LIBERO" websockets imageio

if [[ ! -f "$HOME/.libero/config.yaml" ]]; then
  log "Write ~/.libero/config.yaml"
  mkdir -p "$HOME/.libero"
  "$VENV/bin/python" - "$LIBERO" <<'PY'
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1]) / "libero" / "libero"
config = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
Path.home().joinpath(".libero").mkdir(parents=True, exist_ok=True)
Path.home().joinpath(".libero", "config.yaml").write_text(yaml.safe_dump(config))
print(config["benchmark_root"])
PY
fi

log "Check the environment"
"$VENV/bin/python" - <<'PY'
import imageio
import numpy
import torch
import websockets
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__)
print("libero suites", benchmark.get_benchmark_dict().keys())
print("websockets", websockets.__version__)
print("imageio", imageio.__version__)
print("OffScreenRenderEnv", OffScreenRenderEnv.__name__)
print("libero path helper", get_libero_path.__name__)
PY
log "Done. Activate with: source ${VENV}/bin/activate"
