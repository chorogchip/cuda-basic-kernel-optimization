#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_FILE="${SCRIPT_DIR}/target_params.txt"

block_dims=(64 128 256 512 1024)
read_per_thread=(1 2 4 8 16 32 64)

{
    for block in "${block_dims[@]}"; do
        for rpt in "${read_per_thread[@]}"; do
            printf "%s,%s\n" "${block}" "${rpt}"
        done
    done
} > "${OUT_FILE}"
