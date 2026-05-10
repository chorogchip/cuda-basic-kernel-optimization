#!/usr/bin/env bash

set -euo pipefail

shopt -s nullglob

found=0
for results_dir in ./kernels/*/results; do
  run_files=()
  for candidate in "$results_dir"/*_run.txt; do
    if [[ -s "$candidate" ]]; then
      run_files+=("$candidate")
    fi
  done

  if [[ ${#run_files[@]} -eq 0 ]]; then
    continue
  fi

  found=1
  ./scripts/plot_elem_per_sec.py "${run_files[@]}"

  maxperf_files=()
  for candidate in "$results_dir"/*_run_maxperf.txt; do
    if [[ -s "$candidate" ]]; then
      maxperf_files+=("$candidate")
    fi
  done

  if [[ ${#maxperf_files[@]} -gt 0 ]]; then
    ./scripts/plot_elem_per_sec.py "${maxperf_files[@]}"
  fi
done

if [[ $found -eq 0 ]]; then
  echo "No run files found under ./kernels/*/results/" >&2
  exit 1
fi
