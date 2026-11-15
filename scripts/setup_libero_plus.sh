#!/usr/bin/env bash
# LIBERO-PLUS evaluation client. The venv is LIBERO-PLUS-evaluation/.venv-libero-plus.
# This does not touch LIBERO-evaluation/.venv-libero or ~/.libero/config.yaml.
# PLUS reads ~/.libero-plus through LIBERO_CONFIG_PATH, which test_libero_plus.sh exports.
#
# The checkout stays on disk. The client opens
# LIBERO-plus/libero/libero/benchmark/task_classification.json from that directory.
#
#   bash scripts/setup_libero_plus.sh
#   source LIBERO-PLUS-evaluation/.venv-libero-plus/bin/activate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT="$ROOT/LIBERO-PLUS-evaluation"
VENV="$CLIENT/.venv-libero-plus"
LIBERO="$CLIENT/LIBERO-plus"
PY_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
WHEEL_DIR="${WHEEL_DIR:-$HOME/.cache/evo-depth/torch-cu113}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export DEBIAN_FRONTEND=noninteractive
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero-plus}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
ASSET_ZIP_DIR="${ASSET_ZIP_DIR:-$HOME/.cache/evo-depth/libero-plus-assets}"

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
  log "Clone LIBERO-plus"
  rm -rf "$LIBERO"
  domestic git clone "https://gh-proxy.com/https://github.com/sylvestf/LIBERO-plus.git" "$LIBERO" \
    || git clone "https://github.com/sylvestf/LIBERO-plus.git" "$LIBERO"
fi

if command -v apt-get >/dev/null 2>&1; then
  log "Install system libraries used by opencv, mujoco, and wand"
  apt-get update -qq || log "apt update failed, using packages already installed"
  apt-get install -y -qq build-essential cmake \
    libgl1 libglib2.0-0 libegl1 libosmesa6 libglew2.2 libglfw3 patchelf \
    libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev unzip
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

log "Install LIBERO-plus requirements"
uvpip --index-url "$PY_INDEX" -r "$LIBERO/requirements.txt"

log "Reinstall the pinned torch in case requirements replaced it"
install_torch

log "Install LIBERO-plus, websockets, and imageio"
uvpip --index-url "$PY_INDEX" --config-settings editable_mode=compat \
  -e "$LIBERO" websockets imageio
uvpip --index-url "$PY_INDEX" -r "$LIBERO/extra_requirements.txt"

if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
  log "Write ${LIBERO_CONFIG_PATH}/config.yaml"
  mkdir -p "$LIBERO_CONFIG_PATH"
  "$VENV/bin/python" - "$LIBERO" "$LIBERO_CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1]) / "libero" / "libero"
dest = Path(sys.argv[2])
config = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
dest.mkdir(parents=True, exist_ok=True)
(dest / "config.yaml").write_text(yaml.safe_dump(config))
print(config["benchmark_root"])
PY
fi

ASSETS="$LIBERO/libero/libero/assets"
if [[ ! -d "$ASSETS/textures" || ! -d "$ASSETS/scenes" || ! -d "$ASSETS/stable_hope_objects" ]]; then
  log "Download assets.zip from ${HF_ENDPOINT}"
  mkdir -p "$ASSET_ZIP_DIR"
  uvpip --index-url "$PY_INDEX" "huggingface_hub==0.36.2"
  domestic "$VENV/bin/python" - "$ASSET_ZIP_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id="Sylvest/LIBERO-plus",
    repo_type="dataset",
    filename="assets.zip",
    local_dir=sys.argv[1],
))
PY
  log "Unzip assets into ${LIBERO}/libero/libero"
  unzip -q -o "$ASSET_ZIP_DIR/assets.zip" -d "$LIBERO/libero/libero"
  src="$(find "$LIBERO/libero/libero" -type d -path '*/assets/textures' -printf '%h\n' | head -1)"
  if [[ -z "$src" ]]; then
    echo "assets.zip did not contain assets/textures" >&2
    exit 1
  fi
  # This archive stores assets under a machine-specific prefix, not at the zip root.
  if [[ "$src" != "$ASSETS" ]]; then
    rm -rf "$ASSETS"
    mv "$src" "$ASSETS"
    rm -rf "$LIBERO/libero/libero/inspire"
  fi
fi

log "Check the environment"
"$VENV/bin/python" - "$LIBERO" <<'PY'
import sys
from pathlib import Path
import imageio
import numpy
import torch
import websockets
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
classification = Path(sys.argv[1]) / "libero/libero/benchmark/task_classification.json"
if not classification.is_file():
    raise SystemExit(f"missing {classification}")
print("python ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__)
print("libero suites", list(benchmark.get_benchmark_dict()))
print("websockets", websockets.__version__)
print("imageio", imageio.__version__)
print("OffScreenRenderEnv", OffScreenRenderEnv.__name__)
print("config", get_libero_path("benchmark_root"))
assets = Path(get_libero_path("assets"))
for name in ("textures", "scenes", "stable_hope_objects", "new_objects"):
    path = assets / name
    if not path.is_dir():
        raise SystemExit(f"missing {path}")
    print("assets", name, "ok")
print("task classification", classification)
PY
log "Done. Activate with: source ${VENV}/bin/activate"
