#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")"

NVLIB=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
NVINC=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/include
[[ -e libnccl.so ]] || ln -s "$NVLIB/libnccl.so.2" libnccl.so

NVCC="/usr/local/cuda/bin/nvcc"

echo "=== kernel ==="
$NVCC -O2 -std=c++17 --expt-relaxed-constexpr \
    -gencode arch=compute_90,code=sm_90 \
    -I"$NVINC" -Xcompiler -fPIC \
    -c rix_multi_qp_kernel.cu -o rix_multi_qp_kernel.o

echo "=== host ==="
g++ -O2 -std=c++17 -fPIC -c \
    -I"$NVINC" -I/usr/local/cuda/include \
    rix_gin_host.cc -o rix_gin_host.o

echo "=== link ==="
g++ -O2 -std=c++17 -fPIC -shared \
    rix_gin_host.o rix_multi_qp_kernel.o \
    -L. -lnccl -L/usr/local/cuda/lib64 -lcudart \
    -Xlinker -rpath -Xlinker "$NVLIB" \
    -Xlinker -rpath -Xlinker /usr/local/cuda/lib64 \
    -o librix_multi_qp.so

nm -D librix_multi_qp.so | grep -E "rix_(gin_|multi_qp_)" | head
echo "built."
