#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build librix_replay.so: host init(.cc) + kernel-split device(.cu)
set -euo pipefail
cd "$(dirname "$0")"

NVLIB=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
NVINC=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/include

# 复用 exp04 libnccl.so 软链(2.30 wheel)
[[ -e libnccl.so ]] || ln -sf "$NVLIB/libnccl.so.2" libnccl.so

NVCC="/usr/local/cuda/bin/nvcc"

echo "=== compile kernel ==="
$NVCC -O2 -std=c++17 --expt-relaxed-constexpr \
    -gencode arch=compute_90,code=sm_90 \
    -I"$NVINC" \
    -Xcompiler -fPIC \
    -c rix_replay_kernel.cu -o rix_replay_kernel.o

echo "=== compile host ==="
g++ -O2 -std=c++17 -fPIC -c \
    -I"$NVINC" -I/usr/local/cuda/include \
    rix_gin_host.cc -o rix_gin_host.o

echo "=== link .so ==="
g++ -O2 -std=c++17 -fPIC -shared \
    rix_gin_host.o rix_replay_kernel.o \
    -L. -lnccl \
    -L/usr/local/cuda/lib64 -lcudart \
    -Xlinker -rpath -Xlinker "$NVLIB" \
    -Xlinker -rpath -Xlinker /usr/local/cuda/lib64 \
    -o librix_replay.so

echo "=== built ==="
ls -la librix_replay.so
nm -D librix_replay.so | grep -E "rix_(gin_|r0_|r1_|symmetric)" | head -20
