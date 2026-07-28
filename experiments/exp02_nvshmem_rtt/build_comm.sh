#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build indexer-comm framework with PROPER CUDA device linking.
#
# The bug this fixes: a plain `nvcc -shared -rdc=true` does NOT device-link our
# code against NVSHMEM's device lib, so nvshmemi_ibgda_device_state_d ended up
# as a LOCAL zeroed __constant__ copy (nm showed 'b', not 'U'). NVSHMEM filled
# its own copy; our kernel read the empty one -> cudaErrorIllegalAddress (700)
# on the very first access, before any printf.
#
# Fix (mirrors DeepEP setup.py:43-55): separate compile (-dc) then an explicit
# device-link (-dlink) against libnvshmem_device.a, then assemble the .so.
set -euo pipefail
cd "$(dirname "$0")"

NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
CUDA_INC=/usr/local/cuda/include
[ -e libnvshmem_host.so ] || ln -sf "$NVSHMEM_DIR/lib/libnvshmem_host.so.3" ./libnvshmem_host.so

ARCH="-gencode arch=compute_90a,code=sm_90a"
INCS="-I. -I$NVSHMEM_DIR/include -I$NVSHMEM_DIR/include/host -I$CUDA_INC"

# 1) compile to a relocatable device object
nvcc -dc -Xcompiler -fPIC -O2 -rdc=true $ARCH --expt-relaxed-constexpr \
  indexer_comm.cu -o indexer_comm.o $INCS

# 2) DEVICE-LINK against NVSHMEM's device lib -> resolves the __constant__
#    symbols to NVSHMEM's real instances (this is the step we were missing)
nvcc -dlink -Xcompiler -fPIC $ARCH \
  indexer_comm.o "$NVSHMEM_DIR/lib/libnvshmem_device.a" -o indexer_dlink.o

# 3) assemble the shared lib from both objects + host lib
nvcc -shared -Xcompiler -fPIC $ARCH \
  indexer_comm.o indexer_dlink.o -o librix_comm.so \
  "$NVSHMEM_DIR/lib/libnvshmem_device.a" \
  -L. -lnvshmem_host -lcudart -lcuda \
  -Xlinker -rpath -Xlinker "$NVSHMEM_DIR/lib"

echo "built: librix_comm.so"
echo "=== verify ibgda state symbol is now EXTERNAL ref (U), not local copy (b) ==="
nm librix_comm.so 2>/dev/null | grep -iE "ibgda_device_state_d" | head
nm -D librix_comm.so 2>/dev/null | grep -E " T rix_run_rtt_bench" && echo "  ✓ entry exported"
