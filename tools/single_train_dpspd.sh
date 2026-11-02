#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?Usage: single_train_dpspd.sh CONFIG [extra args]}
shift
python "$(dirname "$0")/train_deepspeed.py" "$CONFIG" "$@"
