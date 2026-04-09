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
