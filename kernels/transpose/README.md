# Matrix Transpose

## Build

From `kernels/transpose/`:

Build all transpose binaries:

```bash
make
```

Remove the built binaries from `bin/`:

```bash
make clean
```

## Run

From `kernels/transpose/`:

Single run:

```bash
./bin/transpose_5 4096
```

Batch run from `configs/sizes.txt`:

```bash
make run
```

Remove the saved run output file from `results/`:

```bash
make clean-run
```
