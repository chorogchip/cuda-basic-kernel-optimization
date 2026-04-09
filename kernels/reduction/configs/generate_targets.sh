#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_FILE="${SCRIPT_DIR}/targets.mk"

versions=(1 2)
block_dims=(64 128 256 512 1024)
read_per_thread=(1 2 4 8 16 32 64)

{
    for version in "${versions[@]}"; do
        for block in "${block_dims[@]}"; do
            for rpt in "${read_per_thread[@]}"; do
                printf "TARGET_NAMES += reduction_%s_%s_%s\n" "${version}" "${block}" "${rpt}"
                printf "TARGET_FLAGS_reduction_%s_%s_%s := -DREDUCTION_VERSION=%s -DMY_BLOCKDIM=%s -DMY_READPERTHREAD=%s\n" \
                    "${version}" "${block}" "${rpt}" "${version}" "${block}" "${rpt}"
            done
        done
    done
} > "${OUT_FILE}"
