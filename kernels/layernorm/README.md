# LayerNorm

## Build

```bash
make
```

## Run

```bash
./bin/layernorm_1_256 1048576
make run
```

## Implementation Notes

Kernel implementation is intentionally left as TODO. Suggested variants:

- mean and variance reduction per row
- RMSNorm-only variant
- vectorized load/store variant
