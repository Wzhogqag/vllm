# SPDX-License-Identifier: Apache-2.0
"""GIN 跨机 indexer 单层 RTT bench launcher。

两机各 1 rank,torch.distributed 起 NCCL comm → 从 ProcessGroupNCCL._comm_ptr 挖出
原生 comm 指针 → 传给 librix_gin_rtt.so:
  rix_gin_init(comm_ptr, rank, world, symmetric_bytes, num_qps) -> ctx
  rix_rtt_run(ctx, up_bytes, down_bytes, iters, warmup) -> avg/p50/p95/p99 us
  rix_gin_finalize(ctx)

方案 A 默认:up=8580, down=8192, iters=1000, warmup=50, B=1(单 token)。
"""
import argparse
import ctypes
import json
import os
import sys

import torch
import torch.distributed as dist


def _get_native_nccl_comm(group: dist.ProcessGroup) -> int:
    backend = group._get_backend(torch.device("cuda"))
    # torch 版本间差异:_comm_ptr 有时是属性,有时是方法(2.11 跨机场景是 method)
    for attr in ("_comm_ptr", "_get_comm_handle", "_ncclComm", "get_nccl_comm"):
        if hasattr(backend, attr):
            v = getattr(backend, attr)
            handle = v() if callable(v) else v
            return int(handle)
    raise RuntimeError(
        f"can't find NCCL comm handle on {type(backend).__name__}; "
        f"attrs: {[a for a in dir(backend) if 'omm' in a.lower()]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up-per-token",   type=int, default=8580, help="方案A 单 token 上行(B*this)")
    ap.add_argument("--down-per-token", type=int, default=8192, help="方案A 单 token 下行(B*this)")
    ap.add_argument("--batch-sizes", type=str, default="1,4,16,64,256",
                    help="扫描的 B 值,逗号分隔;方案 A payload = B * per-token bytes")
    ap.add_argument("--iters",  type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--layers", type=int, default=61)
    ap.add_argument("--symmetric-bytes", type=int, default=16 << 20,  # 16 MiB
                    help="对称堆总大小,必须 >= 2*max(payload) 上限,默认 16 MiB")
    ap.add_argument("--num-qps", type=int, default=1)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    batch_list = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world == 2, f"这个 bench 只做 2-rank 跨机,{world=}"

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    # 触发 NCCL comm lazy-init
    x = torch.zeros(1, device="cuda")
    dist.all_reduce(x)
    torch.cuda.synchronize()

    comm_ptr = _get_native_nccl_comm(dist.group.WORLD)
    if rank == 0:
        print(f"[bench] world={world} native NCCL comm ptr = 0x{comm_ptr:x}", flush=True)

    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "librix_gin_rtt.so")
    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

    lib.rix_gin_init.argtypes = [
        ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.rix_gin_init.restype = ctypes.c_int
    lib.rix_gin_finalize.argtypes = [ctypes.c_void_p]
    lib.rix_gin_finalize.restype = ctypes.c_int
    lib.rix_rtt_run.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rix_rtt_run.restype = ctypes.c_int

    ctx_ptr = ctypes.c_void_p(0)
    rc = lib.rix_gin_init(comm_ptr, rank, world,
                          args.symmetric_bytes, args.num_qps,
                          ctypes.byref(ctx_ptr))
    if rc != 0:
        print(f"[rank {rank}] rix_gin_init rc={rc}", file=sys.stderr, flush=True)
        sys.exit(1)

    dist.barrier()

    all_results = []
    for B in batch_list:
        up = B * args.up_per_token
        down = B * args.down_per_token
        avg = ctypes.c_double(0)
        p50 = ctypes.c_double(0)
        p95 = ctypes.c_double(0)
        p99 = ctypes.c_double(0)
        rc = lib.rix_rtt_run(ctx_ptr, up, down,
                             args.iters, args.warmup,
                             ctypes.byref(avg), ctypes.byref(p50),
                             ctypes.byref(p95), ctypes.byref(p99))
        dist.barrier()
        if rc != 0:
            print(f"[rank {rank}] rix_rtt_run B={B} rc={rc}", file=sys.stderr, flush=True)
            lib.rix_gin_finalize(ctx_ptr)
            sys.exit(1)
        if rank == 0:
            entry = {
                "B": B, "up_bytes": up, "down_bytes": down,
                "rtt_avg_us": avg.value, "rtt_p50_us": p50.value,
                "rtt_p95_us": p95.value, "rtt_p99_us": p99.value,
                "serial_layers_ms": avg.value * args.layers / 1e3,
                "bytes_per_rtt": up + down,
                "gbps_effective": (up + down) / (avg.value * 1e-6) / 1e9,
            }
            all_results.append(entry)
            print(f"  B={B:4d}  up={up:>8d}B  down={down:>8d}B  "
                  f"avg={avg.value:7.2f}us  p50={p50.value:7.2f}us  p99={p99.value:7.2f}us  "
                  f"eff={entry['gbps_effective']:5.2f} GB/s", flush=True)

    if rank == 0:
        print("\n=========== NCCL GIN RTT bench (batch sweep) ===========")
        print(f"{'B':>4}  {'up(B)':>7}  {'down(B)':>7}  {'avg(us)':>8}  {'p50':>7}  {'p95':>7}  {'p99':>7}  "
              f"{'GB/s':>6}  {'61layer(ms)':>11}")
        for r in all_results:
            print(f"{r['B']:>4}  {r['up_bytes']:>7}  {r['down_bytes']:>7}  "
                  f"{r['rtt_avg_us']:>8.2f}  {r['rtt_p50_us']:>7.2f}  {r['rtt_p95_us']:>7.2f}  {r['rtt_p99_us']:>7.2f}  "
                  f"{r['gbps_effective']:>6.2f}  {r['serial_layers_ms']:>11.3f}")
        print("========================================================\n")
        if args.out:
            d = os.path.dirname(args.out)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"transport": "NCCL GIN GDAKI",
                           "iters": args.iters, "warmup": args.warmup,
                           "layers": args.layers,
                           "up_per_token": args.up_per_token,
                           "down_per_token": args.down_per_token,
                           "results": all_results}, f, indent=2)
            print(f"  wrote {args.out}")

    lib.rix_gin_finalize(ctx_ptr)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
