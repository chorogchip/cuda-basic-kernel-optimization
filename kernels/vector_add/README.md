# Vector Add

## Build

From `kernels/vector_add/`:

Build the binary:

```bash
make
```

Remove the built binary from `bin/`:

```bash
make clean
```

## Run

From `kernels/vector_add/`:

Single run:

```bash
./bin/vector_add 1048576
```

Batch run from `configs/sizes.txt`:

```bash
make run
```

Remove the saved run output file from `results/`:

```bash
make clean-run
```
