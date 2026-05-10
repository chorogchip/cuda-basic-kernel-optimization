# Softmax

## Build

```bash
make
```

## Run

```bash
./bin/softmax_1_256 1048576
make run
```

## Implementation Notes

Kernel implementation is intentionally left as TODO. Suggested variants:

- one row per block
- vectorized loads
- fused max, sum, and normalization where feasible
