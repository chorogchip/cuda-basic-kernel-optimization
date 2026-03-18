#!/bin/bash

if [ -z "$2" ]; then
    echo "Usage: $0 <filename> <name_suffix> [additional options]"
    echo "Example: $0 cu8 test1 -DBLOCK_SIZE=256 -DUNROLL_LOOP"
    exit 1
fi

SRC_NAME=$1
NAME_SUFFIX=$2
BIN_NAME=${SRC_NAME}_${NAME_SUFFIX}
SHIFT_ARGS="${@:3}"


nvcc "./src/${SRC_NAME}.cu" -O3 -o "./bin/${BIN_NAME}" ${SHIFT_ARGS}

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi
