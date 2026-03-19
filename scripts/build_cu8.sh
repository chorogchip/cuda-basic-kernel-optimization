#!/bin/bash

SRC="cu8_reduction_add"

for block_dim in 64 128 256 512 1024
do
    for rpt in 1 2 4 8 16 32
    do
        SUFFIX="${block_dim}_${rpt}"
        echo "Building: ${SUFFIX}"

        ./comp_define.sh "$SRC" "$SUFFIX" "-DMY_BLOCKDIM=${block_dim}" "-DMY_READPERTHREAD=${rpt}"

        if [ $? -ne 0 ]; then
            echo "Error: ${SUFFIX} failed"
        fi
    done
done

echo "Done."
