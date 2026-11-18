#!/usr/bin/env bash
# Start LiteGauss websocket server, then LIBERO client (same protocol as Evo).
#
#   bash scripts/run_litegauss_libero.sh
#   bash scripts/run_litegauss_libero.sh libero_spatial
#   LITEGAUSS_CKPT=checkpoints/litegauss/run-bs224-400 bash scripts/run_litegauss_libero.sh
#
# Does not touch Evo_depth_server.py. Kill only the pid recorded in the log dir.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:-libero_spatial}"
PORT="${EVO_DEPTH_SERVER_PORT:-9000}"
CKPT="${LITEGAUSS_CKPT:-${LITEGAUSS_CKPT_DIR:-$ROOT/checkpoints/litegauss/run-bs224-400}}"
LOG_DIR="$ROOT/LIBERO-evaluation/logs/litegauss_${SUITE}"
SERVER_LOG="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"
SERVER_PY="$ROOT/.venv/bin/python"
CLIENT_PY="$ROOT/LIBERO-evaluation/.venv-libero/bin/python"

log() { printf '\n==> %s\n' "$*"; }
domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}

case "$SUITE" in
  libero_spatial|libero_goal|libero_object|libero_10) ;;
  *) echo "suite must be libero_spatial, libero_goal, libero_object, or libero_10" >&2; exit 1 ;;
esac
[[ -x "$SERVER_PY" ]] || { echo "Run scripts/setup_train_server.sh first." >&2; exit 1; }
[[ -x "$CLIENT_PY" ]] || { echo "Run scripts/setup_libero.sh first." >&2; exit 1; }
for rel in config.json norm_stats.json model.pt; do
  [[ -f "$CKPT/$rel" ]] || { echo "missing $CKPT/$rel" >&2; exit 1; }
done
if "$SERVER_PY" -c 'import socket,sys; s=socket.socket(); r=s.connect_ex(("127.0.0.1", int(sys.argv[1]))); s.close(); raise SystemExit(0 if r==0 else 1)' "$PORT"; then
  echo "port ${PORT} is already in use" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    pkill -P "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}
trap cleanup EXIT

port_open() {
  "$SERVER_PY" -c 'import socket,sys; s=socket.socket(); r=s.connect_ex(("127.0.0.1", int(sys.argv[1]))); s.close(); raise SystemExit(0 if r==0 else 1)' "$1"
}

log "Start LiteGauss server ckpt=${CKPT}"
domestic \
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  HF_HUB_DISABLE_XET=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/Evo_depth" \
  LITEGAUSS_CKPT_DIR="$CKPT" \
  EVO_DEPTH_SERVER_PORT="$PORT" \
  "$SERVER_PY" -u Evo_depth/scripts/litegauss_server.py \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!
echo "$server_pid" >"$PID_FILE"

ready=0
for ((i=0; i<360; i++)); do
  if grep -qE "LiteGauss server running|Evo_depth server running" "$SERVER_LOG" 2>/dev/null || port_open "$PORT"; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited. Last lines of ${SERVER_LOG}:" >&2
    tail -n 40 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
[[ "$ready" -eq 1 ]] || { echo "server did not start within 30 minutes" >&2; exit 1; }

log "Run LIBERO client ${SUITE}"
cd "$ROOT/LIBERO-evaluation"
PYTHONPATH="${PYTHONPATH:-}:$ROOT/LIBERO-evaluation/LIBERO" \
  MUJOCO_GL=osmesa \
  "$CLIENT_PY" libero_client.py \
  --task_suites "$SUITE" \
  --server_url "ws://127.0.0.1:${PORT}" \
  --log_file "$LOG_DIR/log.txt" \
  --video_log_dir "$LOG_DIR/videos"
log "Client finished. Log: ${LOG_DIR}/log.txt"
