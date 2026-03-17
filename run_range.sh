#!/bin/bash

START=10
FINISH=24

TARGET_BIN=$1
shift

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -st) START="$2"; shift ;;
        -fn) FINISH="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [ ! -f "$TARGET_BIN" ]; then
    echo "Error: $TARGET_BIN not found."
    exit 1
fi

for (( i=$START; i<=$FINISH; i++ ))
do
    SIZE=$((1 << i))
    
    printf "N = 2^%-2s (%-10s) | " "$i" "$SIZE"
    
    $TARGET_BIN $SIZE
done
