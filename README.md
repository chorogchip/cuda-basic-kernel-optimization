# CUDA Kernel Optimization

## Build

From the repository root:

Build all kernel binaries:

```bash
make
```

Remove all built kernel binaries:

```bash
make clean
```

## Run

From the repository root:

Run all configured kernel sweeps:

```bash
make run-all
```

Remove all saved run output files:

```bash
make clean-run-all
```

## Summarize Results

From the repository root:

Summarize one run file into a sibling `_maxperf.txt` file that keeps only the highest-performance line for each `N`:

```bash
./scripts/summarize_maxperf.sh kernels/reduction/results/reduction_1_run.txt
```

This writes:

```bash
kernels/reduction/results/reduction_1_run_maxperf.txt
```

Summarize all run files under `./kernels/*/results/*_run.txt`:

```bash
./scripts/summarize_maxperf.sh
```

## Plot Results

From the repository root:

Plot one raw run file into `./plots/`:

```bash
./scripts/plot_elem_per_sec.py kernels/reduction/results/reduction_1_run.txt
```

This writes:

```bash
plots/reduction_1_run_plot.png
```

Plot multiple raw run files together in one combined figure:

```bash
./scripts/plot_elem_per_sec.py \
  kernels/transpose/results/transpose_1_run.txt \
  kernels/transpose/results/transpose_2_run.txt \
  kernels/transpose/results/transpose_3_run.txt \
  kernels/transpose/results/transpose_4_run.txt \
  kernels/transpose/results/transpose_5_run.txt \
  kernels/transpose/results/transpose_6_run.txt
```

This writes:

```bash
plots/transpose_combined_plot.png
```

The plotting script accepts only raw `*_run.txt` files and writes PNG files under `./plots/`.
