#!/usr/bin/env bash
# exp15 clean vLLM 启动(无 dump_hook)。单独脚本,避免复合命令脆弱性。
cd "$(dirname "$0")"
# CRITICAL: server runs from this dir, whose sys.path picks /usr/local/.../dist-packages/vllm
# (an OLD separate copy) NOT the repo. Force PYTHONPATH to the repo so our edited vLLM loads.
export PYTHONPATH="/export/home/weizhongqiang.3/vllm:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH="$(readlink -f ../../.venv/lib/python3.12/site-packages/nvidia/nccl/lib):${LD_LIBRARY_PATH:-}"
export NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=ib1 NCCL_DEBUG=WARN
export VLLM_INDEXER_REMOTE=1
export VLLM_INDEXER_REMOTE_HOST=6.102.176.55
export VLLM_INDEXER_REMOTE_PORT=29934
export VLLM_INDEXER_REMOTE_DIR="$(pwd)"
export VLLM_INDEXER_REMOTE_DEBUG=1
mkdir -p runs
find ../../vllm/model_executor/layers -name "*.pyc" -delete 2>/dev/null || true
setsid ../../.venv/bin/python launch_clean.py \
  --model /models/DeepSeek-V3.2 --tensor-parallel-size 8 --enforce-eager \
  --max-num-seqs 4 --max-model-len 1024 --port 8000 \
  --no-enable-prefix-caching \
  > runs/clean_server.log 2>&1 &
echo $! > runs/clean.pid
echo "launched pid=$(cat runs/clean.pid) log=runs/clean_server.log"
