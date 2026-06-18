#!/bin/bash
set -e

# Count existing .trt engines
trt_count=$(find /app/models -maxdepth 1 -name '*.trt' 2>/dev/null | wc -l)

if [ "$trt_count" -eq 0 ]; then
    echo "=== No TRT engines found, downloading ONNX models and building engines ==="
    python /app/scripts/build_engine.py
    echo "=== Build complete ==="
else
    echo "=== $trt_count TRT engine(s) found, skipping build ==="
fi

exec python /app/source/uvicorn.main.py "$@"
