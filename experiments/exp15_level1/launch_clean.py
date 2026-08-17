"""exp15 干净 launcher —— 起真 vLLM,不装 exp10 dump_hook。

Level 1 不需要 dump_hook(那是 exp10 抓数据用的)。dump_hook patch 了 forward_cuda,
会挡住 sparse_attn_indexer op body,导致我的 remote 分支进不去。这里直接交给 vllm CLI,
只让 core 里的 remote 分支(env 门控)生效。

用法同 vllm serve:
    python launch_clean.py --model ... --tensor-parallel-size 8 --enforce-eager ...
"""
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "serve":
        sys.argv.insert(1, "serve")
    sys.argv[0] = "vllm"
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()
