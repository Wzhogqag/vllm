#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the NVSHMEM device-initiated RTT bench into librix_rtt.so.
# Device code (kernel calling nvshmem device APIs) requires:
#   -rdc=true          relocatable device code (NVSHMEM device calls need it)
#   -gencode sm_90a    H200 (Hopper)
#   link libnvshmem_device.a (device runtime) + libnvshmem_host.so.3 (host)
set -euo pipefail
cd "$(dirname "$0")"

NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
CUDA_INC=/usr/local/cuda/include

# nvcc won't accept the versioned libnvshmem_host.so.3 directly; use a local
# unversioned symlink (created here if missing) so -lnvshmem_host resolves.
[ -e libnvshmem_host.so ] || ln -sf "$NVSHMEM_DIR/lib/libnvshmem_host.so.3" ./libnvshmem_host.so

nvcc -shared -Xcompiler -fPIC -O2 \
  -rdc=true -gencode arch=compute_90a,code=sm_90a \
  rix_rtt_kernel.cu -o librix_rtt.so \
  -I"$NVSHMEM_DIR/include" -I"$NVSHMEM_DIR/include/host" -I"$CUDA_INC" \
  "$NVSHMEM_DIR/lib/libnvshmem_device.a" \
  -L. -lnvshmem_host \
  -lcudart -lcuda \
  -Xlinker -rpath -Xlinker "$NVSHMEM_DIR/lib"

echo "built: librix_rtt.so"
nm -D librix_rtt.so 2>/dev/null | grep -E " T rix_run_rtt_bench" && echo "  ✓ entry symbol exported"
