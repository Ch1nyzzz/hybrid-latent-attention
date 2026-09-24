#!/usr/bin/env bash
# Caller uses set -euo pipefail: preserve the command's failure and complete log.
run_logged() {
    local log_path=$1
    shift
    "$@" 2>&1 | tee "$log_path"
}
