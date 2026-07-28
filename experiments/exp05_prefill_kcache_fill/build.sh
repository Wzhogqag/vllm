#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# exp05 prefill K cache 灌充 bench build
set -euo pipefail
cd "$(dirname "$0")"

NVLIB=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
NVINC=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/include
[[ -e libnccl.so ]] || ln -s "$NVLIB/libnccl.so.2" libnccl.so

NVCC="/usr/local/cuda/bin/nvcc"

echo "=== compile kernel ==="
$NVCC -O2 -std=c++17 --expt-relaxed-constexpr \
    -gencode arch=compute_90,code=sm_90 \
    -I"$NVINC" -Xcompiler -fPIC \
    -c rix_fill_kernel.cu -o rix_fill_kernel.o

echo "=== compile host ==="
g++ -O2 -std=c++17 -fPIC -c \
    -I"$NVINC" -I/usr/local/cuda/include \
    rix_gin_host.cc -o rix_gin_host.o

echo "=== link ==="
g++ -O2 -std=c++17 -fPIC -shared \
    rix_gin_host.o rix_fill_kernel.o \
    -L. -lnccl -L/usr/local/cuda/lib64 -lcudart \
    -Xlinker -rpath -Xlinker "$NVLIB" \
    -Xlinker -rpath -Xlinker /usr/local/cuda/lib64 \
    -o librix_fill.so

nm -D librix_fill.so | grep -E "rix_(gin_|fill_)" | head
echo "built."
