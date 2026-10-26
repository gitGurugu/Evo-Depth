#!/bin/bash
# 7 个扰动一起跑。server 要先在 9000–9006 上起来。
# Usage: bash run_plus_clients.sh [libero_spatial|libero_goal|libero_object|libero_10]
SUITE=${1:-libero_spatial}
ROOT=/home/kjl/Evo-Depth/LIBERO-PLUS-evaluation
source "$ROOT/.venv-libero-plus/bin/activate"
cd "$ROOT"

case "$SUITE" in
  libero_spatial) tag=spatial ;;
  libero_goal) tag=goal ;;
  libero_object) tag=object ;;
  libero_10) tag=10 ;;
  *) echo "Usage: $0 libero_spatial|libero_goal|libero_object|libero_10" >&2; exit 1 ;;
esac

i=0
for cat in camera robot language light background noise layout; do
  port=$((9000 + i))
  logdir="./logs/plus_${tag}_${cat}"
  mkdir -p "$logdir"
  if [[ $i -eq 0 ]]; then
    first_cat=$cat
    first_port=$port
    first_logdir=$logdir
  else
    bash ./test_libero_plus.sh "$logdir" "$SUITE" "$cat" "ws://127.0.0.1:${port}" >> "$logdir/log.txt" 2>&1 &
  fi
  i=$((i + 1))
done
bash ./test_libero_plus.sh "$first_logdir" "$SUITE" "$first_cat" "ws://127.0.0.1:${first_port}"
echo "camera finished, waiting for the other 6"
wait
