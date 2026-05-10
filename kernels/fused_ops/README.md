# Fused Elementwise Ops

## Build

```bash
make
```

## Run

```bash
./bin/fused_ops_1_256 1048576
make run
```

## Implementation Notes

Kernel implementation is intentionally left as TODO. Suggested variants:

- bias add plus ReLU
- affine transform plus GELU
- vectorized memory access
