#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <filename> <size>"
    echo "Example: $0 cu1 2097152"
    exit 1
fi

BIN_NAME=$1
DATA_SIZE=$2
NCU_PATH="/usr/local/cuda-13.1/bin/ncu"
DATE_STR=$(date +%H%M%S)
OUTPUT_DIR="./perf"
OUTPUT_FILE="${OUTPUT_DIR}/${BIN_NAME}_${DATA_SIZE}_${DATE_STR}.txt"

mkdir -p $OUTPUT_DIR

echo "Profiling $BIN_NAME with n=$DATA_SIZE..."
sudo $NCU_PATH --set basic -c 1 ./bin/$BIN_NAME $DATA_SIZE > $OUTPUT_FILE

if [ $? -eq 0 ]; then
    echo "Profiling succeed: $OUTPUT_FILE"
else
    echo "Profiling error"
fi
