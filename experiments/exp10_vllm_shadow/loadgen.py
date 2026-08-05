"""exp10 持续压测 —— 零依赖并发 loadgen。

用标准库 urllib + 线程池并发发 completion 请求,持续制造负载。
不产生 pt 文件(现有 server 进程的 _DUMPED 计数早已超过 VLLM_DUMP_FULL_CALLS,
do_full 恒为 False),脚本自身也监控 pt 数以佐证不增长。

env 可调:
    LOADGEN_CONCURRENCY  并发数(默认 24,>max-num-seqs=16 保证排队打满 batch)
    LOADGEN_MAX_TOKENS   每请求生成 token 数(默认 128)
    LOADGEN_DURATION     总时长秒,0=无限(默认 0)
    VLLM_DUMP_DIR        pt 落盘目录(用于监控计数)
"""
import json
import os
import signal
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("LOADGEN_URL", "http://localhost:8000/v1/completions")
MODEL = os.environ.get("LOADGEN_MODEL", "/models/DeepSeek-V3.2")
CONCURRENCY = int(os.environ.get("LOADGEN_CONCURRENCY", "24"))
MAX_TOKENS = int(os.environ.get("LOADGEN_MAX_TOKENS", "128"))
DURATION = int(os.environ.get("LOADGEN_DURATION", "0"))
DUMP_DIR = os.environ.get("VLLM_DUMP_DIR", ".")

PROMPTS = [
    "请介绍一下分布式系统的基本原理",
    "写一首关于秋天的诗",
    "解释什么是张量并行以及它如何加速大模型推理",
    "描述一下你理想中的未来城市",
    "如何系统性地优化大语言模型的推理吞吐?",
]

stop = threading.Event()
lock = threading.Lock()
stats = {"done": 0, "fail": 0, "tokens": 0, "lat_sum": 0.0}


def worker(idx):
    i = 0
    while not stop.is_set():
        prompt = PROMPTS[(idx + i) % len(PROMPTS)]
        body = json.dumps(
            {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.7,
            }
        ).encode()
        req = urllib.request.Request(
            URL, data=body, headers={"Content-Type": "application/json"}
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            dt = time.time() - t0
            n = d.get("usage", {}).get("completion_tokens", 0)
            with lock:
                stats["done"] += 1
                stats["tokens"] += n
                stats["lat_sum"] += dt
        except Exception:
            with lock:
                stats["fail"] += 1
        i += 1


def count_pt():
    try:
        return sum(
            1 for f in os.listdir(DUMP_DIR) if f.endswith(".pt")
        )
    except Exception:
        return -1


def main():
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    print(
        f"[loadgen pid={os.getpid()}] concurrency={CONCURRENCY} "
        f"max_tokens={MAX_TOKENS} url={URL}",
        flush=True,
    )
    print(f"[loadgen] pt baseline = {count_pt()}", flush=True)
    start = time.time()
    last_t, last_done, last_tok = start, 0, 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for k in range(CONCURRENCY):
            ex.submit(worker, k)
        while not stop.is_set():
            time.sleep(10)
            now = time.time()
            with lock:
                done, fail = stats["done"], stats["fail"]
                tok, lat = stats["tokens"], stats["lat_sum"]
            dt = now - last_t
            qps = (done - last_done) / dt if dt else 0
            tps = (tok - last_tok) / dt if dt else 0
            avg_lat = (lat / done) if done else 0
            print(
                f"[loadgen +{int(now - start)}s] done={done} fail={fail} "
                f"win_qps={qps:.2f} win_tok/s={tps:.0f} "
                f"avg_lat={avg_lat:.2f}s pt={count_pt()}",
                flush=True,
            )
            last_t, last_done, last_tok = now, done, tok
            if DURATION and (now - start) >= DURATION:
                stop.set()
    print("[loadgen] stopped.", flush=True)


if __name__ == "__main__":
    main()
