#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build librix_gin_rtt.so: host init (.cc) + device kernel (.cu)
set -euo pipefail
cd "$(dirname "$0")"

NVLIB=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
NVINC=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/include

# 新 NCCL wheel 里的 libnccl.so.2,先建本地无版本软链
[[ -e libnccl.so ]] || ln -s "$NVLIB/libnccl.so.2" libnccl.so

# device 编译:相当于 DeepEP V2 用的 --expt-relaxed-constexpr -rdc=true;
#   -rdc=true 因为要 link libnccl_device.bc(NVSHMEM device 那种一份预编好的 device 库)
# H200 是 SM90(不需 SM90a,官方通用即可)
NVCC="/usr/local/cuda/bin/nvcc"

echo "=== compile rtt kernel (header-only NCCL device API) ==="
$NVCC -O2 -std=c++17 --expt-relaxed-constexpr \
    -gencode arch=compute_90,code=sm_90 \
    -I"$NVINC" \
    -Xcompiler -fPIC \
    -c rix_rtt_kernel.cu -o rix_rtt_kernel.o

echo "=== compile host init ==="
g++ -O2 -std=c++17 -fPIC -c \
    -I"$NVINC" -I/usr/local/cuda/include \
    rix_gin_host.cc -o rix_gin_host.o

echo "=== link .so ==="
g++ -O2 -std=c++17 -fPIC -shared \
    rix_gin_host.o rix_rtt_kernel.o \
    -L. -lnccl \
    -L/usr/local/cuda/lib64 -lcudart \
    -Xlinker -rpath -Xlinker "$NVLIB" \
    -Xlinker -rpath -Xlinker /usr/local/cuda/lib64 \
    -o librix_gin_rtt.so

echo "=== built ==="
ls -la librix_gin_rtt.so
nm -D librix_gin_rtt.so | grep -E "rix_(gin_|rtt_)" | head -10
