#!/usr/bin/env bash

set -euo pipefail

summarize_file() {
  local input_file=$1
  local input_dir
  local input_name
  local input_stem
  local output_file

  if [[ ! -f "$input_file" ]]; then
    echo "Input file not found: $input_file" >&2
    return 1
  fi

  input_dir=$(dirname "$input_file")
  input_name=$(basename "$input_file")
  input_stem=${input_name%.txt}

  if [[ "$input_stem" == "$input_name" ]]; then
    output_file="$input_dir/${input_name}_maxperf.txt"
  else
    output_file="$input_dir/${input_stem}_maxperf.txt"
  fi

  awk '
{
  n = ""
  eps = ""

  for (i = 1; i <= NF; ++i) {
    if ($i == "N" && (i + 1) <= NF) {
      n = $(i + 1)
    }
    if ($i == "elem_per_sec" && (i + 1) <= NF) {
      eps = $(i + 1) + 0
    }
  }

  if (n == "" || eps == "") {
    next
  }

  if (!(n in best_eps) || eps > best_eps[n]) {
    best_eps[n] = eps
    best_line[n] = $0
  }
}

END {
  count = 0
  for (n in best_line) {
    keys[count++] = n + 0
  }

  for (i = 0; i < count; ++i) {
    for (j = i + 1; j < count; ++j) {
      if (keys[i] > keys[j]) {
        tmp = keys[i]
        keys[i] = keys[j]
        keys[j] = tmp
      }
    }
  }

  for (i = 0; i < count; ++i) {
    print best_line[keys[i]]
  }
}
' "$input_file" > "$output_file"

  echo "Wrote $output_file"
}

if [[ $# -eq 0 ]]; then
  shopt -s nullglob
  run_files=(./kernels/*/results/*_run.txt)
  shopt -u nullglob

  if [[ ${#run_files[@]} -eq 0 ]]; then
    echo "No run files found under ./kernels/*/results/" >&2
    exit 1
  fi

  for input_file in "${run_files[@]}"; do
    summarize_file "$input_file"
  done
elif [[ $# -eq 1 ]]; then
  summarize_file "$1"
else
  echo "Usage: $0 [result_file]" >&2
  exit 1
fi
