# SPDX-License-Identifier: Apache-2.0
"""exp05 prefill K cache 灌充 bench launcher。

三种 mode 扫多个 seq_len:
  BULK      = 一次 put 全部 payload
  STREAMING = 每 token 一个 put
  CHUNKED N = 每 N token 一个 put

Payload per token = 132B(vllm 源码坐实,见 exp05/README)。
"""
import argparse
import ctypes
import json
import os
import sys

import torch
import torch.distributed as dist


PER_TOKEN_BYTES = 132   # vllm DeepseekV32IndexerCache: 128 fp8 + 4 fp32 scale (uint8 packed)

MODE_BULK      = 0
MODE_STREAMING = 1
MODE_CHUNKED   = 2

MODE_NAMES = {0: "BULK", 1: "STREAMING", 2: "CHUNKED"}


def _get_native_nccl_comm(group):
    backend = group._get_backend(torch.device("cuda"))
    for attr in ("_comm_ptr", "_get_comm_handle", "_ncclComm", "get_nccl_comm"):
        if hasattr(backend, attr):
            v = getattr(backend, attr)
            return int(v() if callable(v) else v)
    raise RuntimeError(f"no comm handle on {type(backend).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=str, default="2048,8192,32768,131072",
                    help="逗号分隔;方案 A per-token 132B")
    ap.add_argument("--modes", type=str, default="0,2,1",
                    help="逗号分隔:0=BULK 1=STREAMING 2=CHUNKED")
    ap.add_argument("--chunk-tokens", type=int, default=64,
                    help="CHUNKED mode 每块几个 token")
    ap.add_argument("--iters", type=int, default=20,
                    help="每个 (seq_len, mode) 组合重复次数,取 p50")
    ap.add_argument("--symmetric-bytes", type=int, default=64 << 20,   # 64 MiB
                    help="对称堆总大小 -- 128k token × 132B = 16.5 MiB,预留 4x")
    ap.add_argument("--num-qps", type=int, default=1)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    modes = [int(m) for m in args.modes.split(",")]

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world == 2

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    dist.all_reduce(torch.zeros(1, device="cuda"))
    torch.cuda.synchronize()

    comm_ptr = _get_native_nccl_comm(dist.group.WORLD)

    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librix_fill.so")
    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    lib.rix_gin_init.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int64, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_void_p)]
    lib.rix_gin_init.restype = ctypes.c_int
    lib.rix_gin_finalize.argtypes = [ctypes.c_void_p]
    lib.rix_gin_finalize.restype = ctypes.c_int
    lib.rix_fill_run.argtypes = [ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int,
                                 ctypes.c_uint64,
                                 ctypes.POINTER(ctypes.c_double)]
    lib.rix_fill_run.restype = ctypes.c_int

    ctx_ptr = ctypes.c_void_p(0)
    rc = lib.rix_gin_init(comm_ptr, rank, world,
                          args.symmetric_bytes, args.num_qps,
                          ctypes.byref(ctx_ptr))
    if rc != 0:
        sys.exit(f"[rank {rank}] rix_gin_init rc={rc}")

    dist.barrier()

    all_results = []
    signal_target = 0
    for mode in modes:
        for seq_len in seq_lens:
            payload_bytes = seq_len * PER_TOKEN_BYTES
            # STREAMING 长序列会很慢,给 warning + 跳过极端组合
            if mode == MODE_STREAMING and seq_len > 32768:
                if rank == 0:
                    print(f"  [skip] STREAMING seq_len={seq_len} too slow, skipping", flush=True)
                continue

            samples_us = []
            # warmup 1 次(不计)
            signal_target += 1
            out = ctypes.c_double(0)
            rc = lib.rix_fill_run(ctx_ptr, seq_len, PER_TOKEN_BYTES,
                                  args.chunk_tokens, mode,
                                  signal_target, ctypes.byref(out))
            if rc != 0:
                sys.exit(f"[rank {rank}] rix_fill_run rc={rc}")
            dist.barrier()

            for _ in range(args.iters):
                signal_target += 1
                out = ctypes.c_double(0)
                rc = lib.rix_fill_run(ctx_ptr, seq_len, PER_TOKEN_BYTES,
                                      args.chunk_tokens, mode,
                                      signal_target, ctypes.byref(out))
                if rc != 0:
                    sys.exit(f"[rank {rank}] rix_fill_run rc={rc}")
                dist.barrier()
                if rank == 0:
                    samples_us.append(out.value)

            if rank == 0:
                samples_us.sort()
                n = len(samples_us)
                p50 = samples_us[n // 2]
                p95 = samples_us[min(n - 1, int(n * 0.95))]
                p99 = samples_us[min(n - 1, int(n * 0.99))]
                gbps = payload_bytes / (p50 * 1e-6) / 1e9
                entry = {
                    "mode": MODE_NAMES[mode],
                    "seq_len": seq_len,
                    "payload_bytes": payload_bytes,
                    "p50_us": p50, "p95_us": p95, "p99_us": p99,
                    "gbps": gbps,
                }
                if mode == MODE_CHUNKED:
                    entry["chunk_tokens"] = args.chunk_tokens
                all_results.append(entry)
                print(f"  mode={MODE_NAMES[mode]:>9}  seq_len={seq_len:>6}  "
                      f"payload={payload_bytes/1024:>7.1f} KiB  "
                      f"p50={p50:>10.2f} us  p95={p95:>10.2f}  p99={p99:>10.2f}  "
                      f"eff={gbps:>5.2f} GB/s", flush=True)

    if rank == 0:
        print("\n============ prefill K-cache fill ===============")
        print(f"{'mode':>9}  {'seq_len':>7}  {'payload':>10}  {'p50(us)':>11}  {'p95':>10}  {'p99':>10}  {'GB/s':>6}")
        for r in all_results:
            print(f"{r['mode']:>9}  {r['seq_len']:>7}  "
                  f"{r['payload_bytes']/1024:>7.1f}KiB  "
                  f"{r['p50_us']:>11.2f}  {r['p95_us']:>10.2f}  {r['p99_us']:>10.2f}  {r['gbps']:>6.2f}")
        print("=================================================\n")
        if args.out:
            d = os.path.dirname(args.out)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"transport": "NCCL GIN GDAKI",
                           "per_token_bytes": PER_TOKEN_BYTES,
                           "iters_per_point": args.iters,
                           "chunk_tokens_cfg": args.chunk_tokens,
                           "results": all_results}, f, indent=2)
            print(f"  wrote {args.out}")

    lib.rix_gin_finalize(ctx_ptr)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
