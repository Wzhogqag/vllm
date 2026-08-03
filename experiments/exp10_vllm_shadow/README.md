# exp10 — 接 vLLM 影子对拍(Level 0.1:先本地 dump)

## 目的

在 C 机跑 V3.2 (TP=8),hook 每次 `sparse_attn_indexer` 调用,dump payload 到磁盘。
**vLLM 主路径完全不受影响**,只是旁挂观察者。

## Level 阶段

- **Level 0.1(本文件)**:hook + 本地 dump,不涉网络。验证:
  - hook 挂上 vLLM 后能否触发(cudagraph 兼容性)
  - payload 的 shape/dtype 与 exp09 假设是否一致
  - 每 decode step / 每 request 各调用几次(层数验证)
- Level 0.2(下一步):hook 同时把 payload 送到远端 A 机,远端算 topk,回来比 recall,vLLM 仍使用原生
- Level 1(未来):hook 用远端 topk 替换 vLLM 原生,真实分离

## 前置

- **C 机**:8 卡 H200 空闲、`/models/DeepSeek-V3.2` 挂载可读
- vLLM 需要 `--enforce-eager` — cudagraph 模式下 Python monkey-patch 追不到
- `torch >= 2.4`,含 `float8_e4m3fn` 支持

## 用法

```bash
# C 机上
cd /export/home/weizhongqiang.3/vllm/experiments/exp10_vllm_shadow

# 1) 挑 8 张空卡
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits

# 2) 启动 vLLM,preload dump_hook
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_DUMP_DIR=$(pwd)/dumps
export VLLM_DUMP_FULL_CALLS=5   # 前 N 次完整 dump,之后只打摘要

TS=$(TZ='Asia/Shanghai' date '+%Y%m%d-%H%M%S')
mkdir -p run_${TS}_CST
export VLLM_DUMP_DIR=$(pwd)/run_${TS}_CST

setsid ../../.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import dump_hook
dump_hook.install()  # 先 install,再启动 vLLM,避免 attention.py 已绑定
from vllm.entrypoints.openai.api_server import main
main()
" \
--model /models/DeepSeek-V3.2 \
--tensor-parallel-size 8 \
--enforce-eager \
--max-num-seqs 16 \
--max-model-len 8192 \
--port 8000 \
> run_${TS}_CST/server.log 2>&1 &

# 3) 发一次 B=16 的 request(等 server ready ~10 分钟后)
curl -sN localhost:8000/v1/completions -H 'content-type: application/json' -d '{
  "model": "/models/DeepSeek-V3.2",
  "prompt": ["你好"],
  "max_tokens": 4,
  "n": 16
}'
```

## dump 产物

`run_<CST>_CST/call_XXXX.pt`:torch.save 的 dict,含:
- `sig`: shape/dtype 摘要
- `hidden_states`, `q_quant`, `q_scale`, `k`, `weights`: 完整 tensor(前 N 个 call)
- `topk_indices`: 该 call 的 vLLM 原生 topk 结果

## 用 dump 验证的事

启动完 + 收到几个 call 后,cd 回 A 机:

```bash
scp weizhongqiang.3@C_business_ip:.../run_<CST>_CST/call_*.pt /tmp/
../../.venv/bin/python -c "
import torch, glob
for f in sorted(glob.glob('/tmp/call_*.pt'))[:5]:
    d = torch.load(f, weights_only=False)
    print(f, d['sig'])
"
```

确认:
- `hidden_states.shape[0]` = num_tokens(B × 每 request 该 step token 数)
- `q_quant.shape` = `[num_tokens, 64, 128]` fp8
- `weights.shape` = `[num_tokens, 64]` bf16
- `k.shape` = `[num_tokens, 128 + 4]` uint8(fp8 + scale 打包)
- `topk_indices.shape` = `[num_tokens, 2048]` int32

若这些和 exp09 假设一致 → 可以进 Level 0.2。

## 关联

- 前置:exp09(GIN + torch 打分链路正确性验证过 recall=1.0)
- 数据来源:[[project-remote-indexer]] 记忆里的方案 A payload 布局
