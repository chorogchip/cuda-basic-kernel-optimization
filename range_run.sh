#!/bin/bash

START=10
FINISH=24

TARGET_BIN=$1
shift

if [ "$#" -le 2 ]; then
    echo "Usage: $0 <target> -st <logn> -fn <logn>"
    echo "Example: $0 <target> -st 5 -fn 10"
    exit 1
fi

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

    RESULT=$("$TARGET_BIN" "$SIZE")
    printf "%30s | N = 2^%-2d = %-10d\n" "$RESULT" "$i" "$SIZE"
done
