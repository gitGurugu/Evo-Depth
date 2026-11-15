#!/usr/bin/env bash
# Training and server venv for Evo-Depth. The venv is created at the repo root.
#
# Compute capability >= 10 (5090): stop. torch 2.5.1 has no kernel for it.
# Compute capability < 8 (2080): same torch, skip flash-attn.
# gsplat v1.5.2 is compiled with GLM 1.0.1 and the CUDA 12.4 nvcc.
#
#   bash scripts/setup_train_server.sh
#   source .venv/bin/activate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PY_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_INDEX="https://mirror.sjtu.edu.cn/pytorch-wheels/cu124"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export DEBIAN_FRONTEND=noninteractive

log() { printf '\n==> %s\n' "$*"; }
domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}
# Prefer the GitHub proxy, then the original URL.
fetch() {
  local dest="$1" url="$2"
  curl -fL --retry 3 --retry-delay 2 -o "$dest" "https://gh-proxy.com/${url}" \
    || curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"
}
uvpip() {
  domestic uv pip install --python "$VENV/bin/python" "$@"
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

command -v nvidia-smi >/dev/null
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
log "GPU: ${GPU_NAME} (compute capability ${GPU_CAP})"
if awk -v cap="$GPU_CAP" 'BEGIN { exit !(cap + 0 >= 10) }'; then
  echo "5090 / Blackwell is not handled. torch 2.5.1+cu124 cannot run on it." >&2
  exit 1
fi
INSTALL_FLASH=1
if awk -v cap="$GPU_CAP" 'BEGIN { exit !(cap + 0 < 8) }'; then
  INSTALL_FLASH=0
fi

if ! domestic uv python find 3.10 >/dev/null 2>&1; then
  domestic uv python install 3.10
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  domestic uv venv "$VENV" --python 3.10
fi

log "Install torch 2.5.1+cu124"
uvpip --index-url "$TORCH_INDEX" --extra-index-url "$PY_INDEX" --index-strategy unsafe-best-match \
  "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" "xformers==0.0.28.post3"

if [[ "$INSTALL_FLASH" -eq 1 ]]; then
  log "Install flash-attn"
  wheel="flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  fetch "/tmp/${wheel}" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${wheel}"
  uv pip install --python "$VENV/bin/python" "/tmp/${wheel}"
fi

log "Install gsplat v1.5.2 with GLM 1.0.1"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq cuda-nvcc-12-4 cuda-cudart-dev-12-4 python3.10-dev \
    libgl1 libglib2.0-0 libegl1 libgomp1 libusb-1.0-0 libtbb12
fi
if [[ ! -x /usr/local/cuda-12.4/bin/nvcc ]]; then
  echo "CUDA 12.4 nvcc is required to build gsplat against torch 2.5.1+cu124." >&2
  exit 1
fi
rm -rf /tmp/gsplat-src /tmp/glm-src
mkdir -p /tmp/gsplat-src /tmp/glm-src
fetch /tmp/gsplat.tar.gz "https://github.com/nerfstudio-project/gsplat/archive/refs/tags/v1.5.2.tar.gz"
fetch /tmp/glm.tar.gz "https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz"
tar -xzf /tmp/gsplat.tar.gz -C /tmp/gsplat-src
tar -xzf /tmp/glm.tar.gz -C /tmp/glm-src
gsplat_root="$(find /tmp/gsplat-src -mindepth 1 -maxdepth 1 -type d | head -1)"
glm_root="$(find /tmp/glm-src -mindepth 1 -maxdepth 1 -type d | head -1)"
mkdir -p "$gsplat_root/gsplat/cuda/csrc/third_party"
rm -rf "$gsplat_root/gsplat/cuda/csrc/third_party/glm"
mv "$glm_root" "$gsplat_root/gsplat/cuda/csrc/third_party/glm"
PATH="/usr/local/cuda-12.4/bin:$PATH" CUDA_HOME=/usr/local/cuda-12.4 \
  TORCH_CUDA_ARCH_LIST="$GPU_CAP" MAX_JOBS="${MAX_JOBS:-4}" \
  uvpip --no-build-isolation --index-url "$PY_INDEX" "$gsplat_root"

log "Install the remaining Python packages"
filtered="$(mktemp)"
grep -vE '^(torch|torchvision|xformers)([=<>[:space:]]|$)' "$ROOT/Evo_depth/requirements.txt" > "$filtered"
uvpip --index-url "$PY_INDEX" --index-strategy unsafe-best-match \
  -r "$filtered" "numpy==1.26.4" "addict==2.4.0"
rm -f "$filtered"

log "Check the environment"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, torch.cuda.get_device_name(0))
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except ModuleNotFoundError:
    print("flash_attn skipped")
import gsplat
print("gsplat", gsplat.__version__)
PY
log "Done. Activate with: source ${VENV}/bin/activate"
