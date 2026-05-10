# Sparse Matrix-Vector Multiply

## Build

```bash
make
```

## Run

```bash
./bin/spmv_1_256_8 1048576
make run
```

Input `N` is interpreted as the row count for a synthetic CSR matrix.

## Implementation Notes

Kernel implementation is intentionally left as TODO. Suggested variants:

- one thread per row
- one warp per row
- vector CSR or segmented reduction
