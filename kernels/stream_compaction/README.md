# Stream Compaction

## Build

```bash
make
```

## Run

```bash
./bin/stream_compaction_1_256 1048576
make run
```

## Implementation Notes

Kernel implementation is intentionally left as TODO. Suggested variants:

- predicate plus prefix scan plus scatter
- block-local compaction
- sparse and dense keep-rate cases
