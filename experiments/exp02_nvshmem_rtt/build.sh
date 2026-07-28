#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the minimal NVSHMEM host wrapper into librix_nvshmem.so.
set -euo pipefail
cd "$(dirname "$0")"

NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
CUDA_INC=/usr/local/cuda/include

g++ -shared -fPIC -O2 rix_nvshmem.cc -o librix_nvshmem.so \
  -I"$NVSHMEM_DIR/include" \
  -I"$NVSHMEM_DIR/include/host" \
  -I"$CUDA_INC" \
  "$NVSHMEM_DIR/lib/libnvshmem_host.so.3" \
  -L/usr/local/cuda/lib64 -lcudart \
  -Wl,-rpath,"$NVSHMEM_DIR/lib"

echo "built: librix_nvshmem.so"
nm -D librix_nvshmem.so | grep -E " T rix_" | awk '{print "  export:", $3}'
