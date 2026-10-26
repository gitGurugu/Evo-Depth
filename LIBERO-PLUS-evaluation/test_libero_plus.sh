#!/bin/bash
# LIBERO-PLUS evaluation - pass log path, task, and filter category.
# Usage: ./test_libero_plus.sh <log_path> [task] [filter_category] [server_url]
#   log_path: dir to save log.txt and videos (required)
#   task: libero_spatial | libero_goal | libero_object | libero_10  (default: libero_spatial)
#   filter_category: perturbation category to evaluate, comma-separated; empty string means all tasks (default: background)
#   server_url: default ws://127.0.0.1:9000

LOG_PATH=${1:?Usage: $0 <log_path> [task] [filter_category] [server_url]}
TASK=${2:-libero_spatial}
FILTER_CATEGORY=${3:-background}
SERVER_URL=${4:-ws://127.0.0.1:9000}
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero-plus}"
if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
    echo "Missing $LIBERO_CONFIG_PATH/config.yaml"
    exit 1
fi
export PYTHONPATH=$PYTHONPATH:./LIBERO-plus/
echo "=============================="
echo "Log path: $LOG_PATH | Task: $TASK | Filter: $FILTER_CATEGORY | Server: $SERVER_URL"
echo "=============================="

python libero_plus_client.py \
    --task_suites "$TASK" \
    --filter_category "$FILTER_CATEGORY" \
    --server_url "$SERVER_URL" \
    --log_file "${LOG_PATH}/log.txt" \
    --video_log_dir "${LOG_PATH}/videos"
