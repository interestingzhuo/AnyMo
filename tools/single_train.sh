#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?Usage: single_train.sh CONFIG [extra args]}
shift
python "$(dirname "$0")/train.py" "$CONFIG" "$@"
