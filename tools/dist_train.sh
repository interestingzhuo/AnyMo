#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?Usage: dist_train.sh CONFIG [GPUS] [extra args]}
GPUS=${2:-1}
shift "$(( $# < 2 ? $# : 2 ))"
python -m torch.distributed.run --nproc_per_node="$GPUS" --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${PORT:-29500}" "$(dirname "$0")/train.py" "$CONFIG" --launcher pytorch "$@"
