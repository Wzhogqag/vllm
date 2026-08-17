"""exp15 loopback 协议验证:client(rank0)↔ serve(rank1)真 GIN 往返,不接 vLLM。

用 exp10 抓取的 decode payload 驱动 client.roundtrip,serve_remote.py 当 rank1 打分回传,
校验 recall vs native topk。目的:在接真 vLLM(占 8 卡、跨机)之前,先把
「client 打包 → GIN → serve 解包 → 整块写 cache → 真 op 打分 → 回传」整条协议链在单机验通。

用法:
    rank1:  python serve_remote.py <host_ip> <port> <max_model_len>   (先起,GPU1)
    rank0:  python loopback_test.py <host_ip> <port> <payload.pt>     (后起,GPU0)
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INDEX_TOPK = 2048


def main():
    host, port, payload_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")

    # client 的 host/port 从 env 读(它内部 create rank0)
    os.environ["VLLM_INDEXER_REMOTE"] = "1"
    os.environ["VLLM_INDEXER_REMOTE_HOST"] = host
    os.environ["VLLM_INDEXER_REMOTE_PORT"] = str(port)
    from indexer_remote_client import get_client

    p = torch.load(payload_path, weights_only=False, map_location="cpu")
    sc = p["score"]
    used = sc["kv_cache_used"]
    blocks = used["blocks"].to(dev)                 # [nb,bs,132]
    seq_len = int(sc["seq_lens"].max().item())
    q = p["inputs"]["q_quant"].to(dev)
    w = p["inputs"]["weights"].to(dev).float()
    # 单请求:逻辑块号 arange(nb)(和 serve 端 resident cache 对齐)
    block_ids = torch.arange(blocks.shape[0], dtype=torch.int32)

    client = get_client()
    print("[loopback] client ready, sending one round-trip...", flush=True)
    topk = client.roundtrip(q, w, blocks, block_ids, seq_len)

    ref = p["output"].cpu()
    got = topk.cpu()
    a = set(x for x in got[0].tolist() if 0 <= x < seq_len)
    b = set(x for x in ref[0].tolist() if 0 <= x < seq_len)
    rec = len(a & b) / max(len(b), 1)
    print(f"[loopback] seq_len={seq_len} recall={rec:.4f}", flush=True)
    print("PASS ✓ full client↔serve GIN protocol" if rec >= 0.99 else "FAIL ✗",
          flush=True)


if __name__ == "__main__":
    main()
