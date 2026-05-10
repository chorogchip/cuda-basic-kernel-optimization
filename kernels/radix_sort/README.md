# Radix Sort

## Build

```bash
make
```

## Run

```bash
./bin/radix_sort_1_256_4 1048576
make run
```

## Implementation Notes

Kernel implementation is intentionally left as TODO. Suggested variants:

- per-pass histogram
- prefix scan of buckets
- scatter into ping-pong buffers
