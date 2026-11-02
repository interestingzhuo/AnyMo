#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?Usage: dist_test.sh CONFIG CHECKPOINT [GPUS] [extra args]}
CHECKPOINT=${2:?Missing checkpoint}
GPUS=${3:-1}
shift "$(( $# < 3 ? $# : 3 ))"
python -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="${PORT:-29500}" "$(dirname "$0")/test.py" "$CONFIG" "$CHECKPOINT" --launcher pytorch "$@"
