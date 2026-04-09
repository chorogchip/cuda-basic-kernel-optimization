# Reduction

## Build

From `kernels/reduction/`:

Build all reduction binaries:

```bash
make
```

Remove the built binaries from `bin/`:

```bash
make clean
```

## Run

From `kernels/reduction/`:

Single run:

```bash
./bin/reduction_1_128_32 8388608
```

Another example:

```bash
./bin/reduction_2_512_64 268435456
```

Batch run from `configs/sizes.txt`:

```bash
make run
```

Remove the saved run output file from `results/`:

```bash
make clean-run
```
