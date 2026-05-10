# Histogram

## Build

From `kernels/histogram/`:

```bash
make
```

## Run

Single run:

```bash
./bin/histogram_1_256_256 1048576
```

Batch run from `configs/sizes.txt`:

```bash
make run
```

## Implementation Notes

Kernel implementation is intentionally left as TODO. Suggested variants:

- global atomics
- shared-memory privatized bins
- per-block histograms plus merge
