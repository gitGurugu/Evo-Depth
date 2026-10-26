#!/usr/bin/env bash
# 四个难度组一起跑。server 要先在 9000–9003 上起来。
# Usage: bash run_mt50_clients.sh [preview] [viewer]
#   preview: on（默认，8090–8093）| off
#   viewer:  off（默认）| on   on 时四块桌面各一个网页，四路分别用 DISPLAY :1 到 :4
set -euo pipefail

PREVIEW=${1:-on}
VIEWER=${2:-off}
if [[ "$PREVIEW" != on && "$PREVIEW" != off ]]; then
  echo "Usage: $0 [on|off] [on|off]" >&2
  exit 1
fi
if [[ "$VIEWER" != on && "$VIEWER" != off ]]; then
  echo "Usage: $0 [on|off] [on|off]" >&2
  exit 1
fi

ROOT=/home/kjl/Evo-Depth/Metaworld-evaluation
PIDFILE=/home/kjl/evo_mt50_client_pids
if [[ "$VIEWER" == on ]]; then
  bash "$ROOT/start_mujoco_desktop.sh"
fi
source "$ROOT/.venv-metaworld/bin/activate"
cd "$ROOT"
export MUJOCO_GL=egl
: > "$PIDFILE"

launch() {
  local port=$1 level=$2 name=$3 preview=$4 display=$5
  local cmd=(bash ./test_metaworld.sh ./logs "ws://127.0.0.1:${port}" --target_level "$level" --run_name "$name")
  [[ "$PREVIEW" == on ]] && cmd+=(--preview_port "$preview")
  [[ "$VIEWER" == on ]] && cmd+=(--viewer)
  if [[ "$VIEWER" == on ]]; then
    DISPLAY=":$display" "${cmd[@]}" &
  else
    "${cmd[@]}" &
  fi
  echo $! >> "$PIDFILE"
}

launch 9000 easy mt50_easy 8090 1
launch 9001 medium mt50_medium 8091 2
launch 9002 hard mt50_hard 8092 3
launch 9003 very_hard mt50_very_hard 8093 4
if [[ "$VIEWER" == on ]]; then
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    got=0
    for n in 1 2 3 4; do
      DISPLAY=:$n xwininfo -root -tree 2>/dev/null | grep -q "MuJoCo : MuJoCo Model" && got=$((got + 1))
    done
    [[ "$got" -eq 4 ]] && break
    sleep 0.25
  done
  bash "$ROOT/start_mujoco_desktop.sh"
fi
wait
