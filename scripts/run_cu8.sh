#!/bin/bash

# y = READPERTHREAD, x = BLOCKDIM
for y in 1 2 4 8
do
    for x in 64 128 256 512 1024
    do
        BIN="./bin/cu8_reduction_add_${x}_${y}"
        
        if [ -f "$BIN" ]; then
            echo "Executing: $BIN"
            ./range_run.sh "$BIN" -st 8 -fn 27
        else
            echo "Error: $BIN not found"
        fi
    done
done
