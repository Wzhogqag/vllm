# SPDX-License-Identifier: Apache-2.0
"""exp07 multi-QP concurrency bench launcher.

扫两个维度:
  N (n_blocks)  ∈ {1, 2, 4, 8}   模拟并发的主实例数
  num_qps       ∈ {1, 2, 4, 8}   indexer 侧分配的 QP 数

期望:
  - N <= num_qps:每 CTA 独占 QP,单 CTA 的 avg RTT ≈ exp04 baseline(~21us B=1 或 ~4us B=16)
  - N > num_qps:多 CTA 挤 QP,竞争 SQ,avg RTT 显著上升,max 抖动大
"""
import argparse, ctypes, json, os, sys
import torch, torch.distributed as dist


def _get_native_nccl_comm(group):
    backend = group._get_backend(torch.device("cuda"))
    for attr in ("_comm_ptr", "_get_comm_handle", "_ncclComm", "get_nccl_comm"):
        if hasattr(backend, attr):
            v = getattr(backend, attr)
            return int(v() if callable(v) else v)
    raise RuntimeError(f"no comm handle on {type(backend).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up-per-token",   type=int, default=8580)
    ap.add_argument("--down-per-token", type=int, default=8192)
    ap.add_argument("--batch-sizes",  type=str, default="1,16",
                    help="每主实例一次发多少 token,逗号分隔;实际 payload = B × per-token")
    ap.add_argument("--n-blocks",   type=str, default="1,2,4,8",
                    help="并发发送 CTA 数(模拟主实例数),逗号分隔")
    ap.add_argument("--num-qps-list", type=str, default="1,2,4,8",
                    help="QP 池大小,逗号分隔")
    ap.add_argument("--iters",  type=int, default=100)   # 之前 500,先小,超时保护也降低总时间
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--symmetric-bytes", type=int, default=64 << 20)  # 64 MiB (覆盖 8 CTA × 4MB × 2)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    n_blocks_list = [int(n) for n in args.n_blocks.split(",")]
    num_qps_list = [int(q) for q in args.num_qps_list.split(",")]
    batch_list = [int(b) for b in args.batch_sizes.split(",")]

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world == 2

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    dist.all_reduce(torch.zeros(1, device="cuda"))
    torch.cuda.synchronize()

    comm_ptr = _get_native_nccl_comm(dist.group.WORLD)

    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librix_multi_qp.so")
    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    lib.rix_gin_init.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int64, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_void_p)]
    lib.rix_gin_init.restype = ctypes.c_int
    lib.rix_gin_finalize.argtypes = [ctypes.c_void_p]
    lib.rix_gin_finalize.restype = ctypes.c_int
    lib.rix_multi_qp_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_int)]
    lib.rix_multi_qp_run.restype = ctypes.c_int

    all_results = []
    for num_qps in num_qps_list:
        ctx_ptr = ctypes.c_void_p(0)
        rc = lib.rix_gin_init(comm_ptr, rank, world,
                              args.symmetric_bytes, num_qps,
                              ctypes.byref(ctx_ptr))
        if rc != 0:
            sys.exit(f"[rank {rank}] gin_init num_qps={num_qps} rc={rc}")

        dist.barrier()

        for B in batch_list:
            up_bytes = B * args.up_per_token
            down_bytes = B * args.down_per_token
            for N in n_blocks_list:
                # 跳过 N > num_qps:上一版实测这条会 SQ 死锁,数据无意义
                if N > num_qps:
                    if rank == 0:
                        print(f"  [skip] B={B} N={N} > num_qps={num_qps} (SQ contention, kernel deadlocks)", flush=True)
                    continue
                avg = ctypes.c_double(0)
                mx = ctypes.c_double(0)
                mn = ctypes.c_double(0)
                to = ctypes.c_int(0)
                rc = lib.rix_multi_qp_run(ctx_ptr, up_bytes, down_bytes,
                                          args.iters, args.warmup, N,
                                          ctypes.byref(avg),
                                          ctypes.byref(mx),
                                          ctypes.byref(mn),
                                          ctypes.byref(to))
                dist.barrier()
                if rc != 0:
                    sys.exit(f"[rank {rank}] multi_qp_run B={B} N={N} num_qps={num_qps} rc={rc}")
                if rank == 0:
                    entry = {"B": B, "n_blocks": N, "num_qps": num_qps,
                             "up_bytes": up_bytes, "down_bytes": down_bytes,
                             "avg_us": avg.value, "min_us": mn.value, "max_us": mx.value,
                             "timeouts": to.value}
                    all_results.append(entry)
                    print(f"  B={B:3d}  N={N}  num_qps={num_qps}  "
                          f"avg={avg.value:7.2f}us  min={mn.value:7.2f}  max={mx.value:7.2f}  "
                          f"(spread={mx.value-mn.value:6.2f}us, timeouts={to.value})", flush=True)

        lib.rix_gin_finalize(ctx_ptr)
        dist.barrier()

    if rank == 0:
        print("\n============ multi-QP concurrency ==============")
        print(f"{'B':>3}  {'N':>3}  {'QP':>3}  {'avg(us)':>9}  {'min':>8}  {'max':>8}  {'spread':>7}  {'timeouts':>8}")
        for r in all_results:
            print(f"{r['B']:>3}  {r['n_blocks']:>3}  {r['num_qps']:>3}  {r['avg_us']:>9.2f}  "
                  f"{r['min_us']:>8.2f}  {r['max_us']:>8.2f}  "
                  f"{r['max_us']-r['min_us']:>7.2f}  {r['timeouts']:>8d}")
        print("================================================\n")
        if args.out:
            d = os.path.dirname(args.out)
            if d: os.makedirs(d, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"transport": "NCCL GIN GDAKI multi-QP",
                           "up_per_token": args.up_per_token,
                           "down_per_token": args.down_per_token,
                           "iters": args.iters, "results": all_results}, f, indent=2)
            print(f"  wrote {args.out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
