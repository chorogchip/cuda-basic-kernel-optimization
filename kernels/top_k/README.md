# Top-K

## Build

```bash
make
```

## Run

```bash
./bin/top_k_1_256_16 1048576
make run
```

## Implementation Notes

Kernel implementation is intentionally left as TODO. Suggested variants:

- block-local top-k
- heap or insertion buffer per thread
- multi-stage merge
