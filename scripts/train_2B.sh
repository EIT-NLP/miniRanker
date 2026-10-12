#!/usr/bin/env bash
set -euo pipefail
export MODEL_SIZE=2B
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" "$@"
