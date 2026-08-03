"""Python 启动脚本 —— 每个使用 PYTHONSTARTUP 的 Python 进程会自动执行。

vLLM V1 spawn worker subprocess 时,通过 env 变量继承 PYTHONSTARTUP,所以主进程和
worker 都会跑这里的 install()。
"""
import sys
from pathlib import Path

_dump_dir = Path(__file__).parent
if str(_dump_dir) not in sys.path:
    sys.path.insert(0, str(_dump_dir))

try:
    import dump_hook
    dump_hook.install()
except Exception as e:
    import os
    print(f"[startup pid={os.getpid()}] dump_hook install failed: {e}", flush=True)
