#!/usr/bin/env bash
# 编 GIN 探测 .so
# 用新装的 nvidia-nccl-cu13>=2.30.4 里的 header + .so(2.30.7),
# 系统 /usr/include 里的老 header 保持不动
set -euo pipefail
cd "$(dirname "$0")"

NVLIB=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
NVINC=/export/home/weizhongqiang.3/vllm/.venv/lib/python3.12/site-packages/nvidia/nccl/include

# 新 NCCL 只有 libnccl.so.2, 没无版本软链;给 -l 用一个本地软链
[[ -e libnccl.so ]] || ln -s "$NVLIB/libnccl.so.2" libnccl.so

g++ -O2 -fPIC -shared \
    -I"$NVINC" \
    -I/usr/local/cuda/include \
    -o librix_gin_probe.so probe_gin.cc \
    -L. -lnccl \
    -Xlinker -rpath -Xlinker "$NVLIB"

echo "built librix_gin_probe.so"
nm -D librix_gin_probe.so | grep -E "rix_probe_gin" || { echo "ERROR: rix_probe_gin not exported"; exit 1; }
echo "symbols OK"
