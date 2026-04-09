#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <binary_path> [n]" >&2
  exit 1
fi

binary_path=$1
problem_size=${2:-1048576}

if [[ ! -f "$binary_path" ]]; then
  echo "Binary not found: $binary_path" >&2
  exit 1
fi

if [[ ! -x "$binary_path" ]]; then
  echo "Binary is not executable: $binary_path" >&2
  exit 1
fi

ncu_path=${NCU_PATH:-}

if [[ -z "$ncu_path" && -x /usr/local/cuda-13.1/bin/ncu ]]; then
  ncu_path=/usr/local/cuda-13.1/bin/ncu
fi

if [[ -z "$ncu_path" ]]; then
  ncu_path=$(command -v ncu || true)
fi

if [[ -z "$ncu_path" ]]; then
  echo "ncu not found in PATH. Set NCU_PATH or install Nsight Compute." >&2
  exit 1
fi

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  exec sudo "$ncu_path" --set basic -c 1 "$binary_path" "$problem_size"
fi

exec "$ncu_path" --set basic -c 1 "$binary_path" "$problem_size"
