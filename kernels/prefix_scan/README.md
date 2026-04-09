# Prefix Scan

## Build

From `kernels/prefix_scan/`:

Build all prefix-scan binaries:

```bash
make
```

Remove the built binaries from `bin/`:

```bash
make clean
```

## Run

From `kernels/prefix_scan/`:

Single run:

```bash
./bin/prefix_scan_256 2097152
```

Batch run from `configs/sizes.txt`:

```bash
make run
```

Remove the saved run output file from `results/`:

```bash
make clean-run
```
