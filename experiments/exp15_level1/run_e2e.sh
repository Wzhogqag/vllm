#!/usr/bin/env bash
# exp15 端到端一键跑通(单机 loopback):serve_remote(GIN rank1)+ 真 vLLM TP=8(rank0)
# → 打一个固定 prompt → 比对 baseline 逐 token 一致。
#
# 拓扑:vLLM 吃 GPU0-7(TP=8)= GIN rank0;serve_remote 也放 cuda:0 co-resident = rank1。
# rendezvous:StatelessProcessGroup host=6.102.176.55(本机 ib1)port=29934,两侧同 host:port。
# 时序:先起 serve(它 create(rank=1) 阻塞等 rank0),再起 vLLM(load 时 eager-init create(rank=0)
# 汇合)。vLLM ready 后打请求,indexer 每层外发 serve 打分,回 topk。
set -u
cd "$(dirname "$0")"
export PYTHONPATH="/export/home/weizhongqiang.3/vllm:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(readlink -f ../../.venv/lib/python3.12/site-packages/nvidia/nccl/lib):${LD_LIBRARY_PATH:-}"
export NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=ib1 NCCL_DEBUG=WARN
export VLLM_INDEXER_REMOTE_DEBUG_V=1
HOST=6.102.176.55
PORT=29934
MAXLEN=1024
PY=../../.venv/bin/python
mkdir -p runs

# 远端打分开关(worker 读文件,不靠 env 转发)
cat > /tmp/vllm_indexer_remote.json <<JSON
{"enabled": true, "debug": true, "dir": "$(pwd)", "host": "$HOST", "port": $PORT}
JSON
: > /tmp/vllm_indexer_remote_debug   # touch 调试打点开关

# 时序:rank0(vLLM)binds TCPStore master(StatelessProcessGroup.create:launch_server=rank==0),
# rank1(serve)是 client 连它。所以**先起 vLLM** 让它在 model 构造时(load 一开始)bind master,
# 再起 serve 连上。两侧 create 谁先到都阻塞等对方(store_timeout=1800s),但 rank0-first 最稳,
# 避开 client 端"连不上重试超限"(serve 单独起、没有 vLLM 时就是这么失败的)。
echo "[run] launching vLLM TP=8 (GIN rank0, binds master) ..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid $PY launch_clean.py \
  --model /models/DeepSeek-V3.2 --tensor-parallel-size 8 --enforce-eager \
  --max-num-seqs 4 --max-model-len "$MAXLEN" --port 8000 \
  --no-enable-prefix-caching \
  > runs/e2e_server.log 2>&1 &
echo $! > runs/e2e_server.pid
echo "[run] vLLM pid=$(cat runs/e2e_server.pid) — log: runs/e2e_server.log"

# 等 rank0 bind master(eager-init 打 "building GIN client")再起 serve,确保 client 有 master 可连。
echo "[run] waiting for rank0 to bind GIN master ..."
for i in $(seq 1 60); do
  grep -q "building GIN client" runs/e2e_server.log 2>/dev/null && break
  sleep 5
done
echo "[run] launching serve_remote (GIN rank1) on cuda:0 ..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid $PY serve_remote.py "$HOST" "$PORT" "$MAXLEN" \
  > runs/e2e_serve.log 2>&1 &
echo $! > runs/e2e_serve.pid
echo "[run] serve pid=$(cat runs/e2e_serve.pid) — waiting for readiness (log: runs/e2e_server.log)"
