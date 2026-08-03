"""C 机启动 vLLM server 的包装:先 install dump_hook,再交给 vllm CLI 处理 argv。

用法:
    python launch_with_hook.py --model /models/DeepSeek-V3.2 --tensor-parallel-size 8 \
        --enforce-eager --max-num-seqs 16 --max-model-len 8192 --port 8000

或用 module 形式(entrypoint 相同):
    python -m launch_with_hook <相同 args>
"""
import os
import sys
from pathlib import Path

# ---- 1) install hook 在 vllm import 之前 ----
sys.path.insert(0, str(Path(__file__).parent))
import dump_hook
dump_hook.install()

# ---- 2) 用 vllm CLI 的 argv 处理 ----
# vllm 的 CLI 入口在 vllm.entrypoints.cli.main:main,它按 `vllm serve <args>` 模式接受 argv
if __name__ == "__main__":
    # 若没显式说 serve,补上
    if len(sys.argv) < 2 or sys.argv[1] != "serve":
        sys.argv.insert(1, "serve")

    # 直接把 argv 交给 vllm CLI
    sys.argv[0] = "vllm"
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()
