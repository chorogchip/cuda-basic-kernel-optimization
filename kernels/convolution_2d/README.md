# 2D Convolution

## Build

```bash
make
```

## Run

```bash
./bin/convolution_2d_1_16_1 1024
make run
```

The input `N` is interpreted as image width and height for a square image.

## Implementation Notes

Kernel implementation is intentionally left as TODO. Suggested variants:

- direct global-memory convolution
- constant-memory filter
- shared-memory tiled input with halo
