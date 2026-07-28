# SPDX-License-Identifier: Apache-2.0
"""NVSHMEM host-API put RTT bench (first version, UPPER BOUND).

Scheme A indexer round-trip modeled with NVSHMEM host-side puts:
  PE0 (local)  putmem 8580B (index_q) -> PE1 (remote) symmetric buffer
  PE1 (remote) putmem 8192B (top-k)   -> PE0 symmetric buffer
  both          barrier_all()          (sync point -- see caveat)

CAVEAT (why this is an upper bound): wait_until is __device__-only, so the host
path cannot do point-to-point signal handoff. We use barrier_all as the sync,
whose collective cost inflates RTT. Compare against NCCL cross-node ~130us: if
this is clearly lower, the device-initiated version is worth writing to reach
the ~4us RDMA floor.

Launch (both machines, complete commands in README):
  A: torchrun --nnodes=2 --node_rank=0 --master_addr=<A_ib_ip> bench_nvshmem_put.py
  B: torchrun --nnodes=2 --node_rank=1 --master_addr=<A_ib_ip> bench_nvshmem_put.py
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

from nvshmem_ctypes import Nvshmem

# Scheme A payloads (bytes), from exp01 common.py (verified from config)
UP_BYTES = 8580     # index_q_fp8 8192 + weights 256 + index_k 132
DOWN_BYTES = 8192   # top-k indices 2048 x int32
NUM_LAYERS = 61


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    # --- torch.distributed bootstrap (over IB, gloo for the UID broadcast) ---
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"need 2 ranks, got {world}"
    torch.cuda.set_device(0)  # launcher pins one visible GPU per node
    device = torch.device("cuda:0")

    # --- NVSHMEM UID bootstrap ---
    sh = Nvshmem()
    if rank == 0:
        uid = sh.get_uniqueid_bytes()
    else:
        uid = b""
    obj = [uid]
    dist.broadcast_object_list(obj, src=0)   # send 128-byte UID to rank1
    uid = obj[0]
    sh.init_with_uid(uid, myrank=rank, nranks=world)

    mype = sh.my_pe()
    npes = sh.n_pes()
    peer = 1 - mype

    # --- symmetric buffers (one per direction, sized to the larger payload) ---
    buf = sh.malloc(max(UP_BYTES, DOWN_BYTES))   # remote-writable landing buffer
    # local source tensors (regular cuda memory is fine as put SOURCE)
    src_up = torch.empty(UP_BYTES, dtype=torch.uint8, device=device)
    src_down = torch.empty(DOWN_BYTES, dtype=torch.uint8, device=device)
    stream = torch.cuda.current_stream().cuda_stream

    sh.barrier_all()

    def one_rtt():
        # PE0 sends up (8580) to PE1; PE1 sends down (8192) to PE0; barrier syncs.
        if mype == 0:
            sh.putmem_on_stream(buf, src_up.data_ptr(), UP_BYTES, peer, stream)
        else:
            sh.putmem_on_stream(buf, src_down.data_ptr(), DOWN_BYTES, peer, stream)
        sh.quiet()
        sh.barrier_all()

    for _ in range(args.warmup):
        one_rtt()
    sh.barrier_all()
    torch.cuda.synchronize()

    # CUDA-event timing on rank0
    samples_ms = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        one_rtt()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))

    sh.barrier_all()

    if rank == 0:
        xs = sorted(samples_ms)
        n = len(xs)
        pct = lambda q: xs[min(n - 1, int(q * n))] * 1e3  # us
        res = {
            "transport": "NVSHMEM host putmem_on_stream + barrier (UPPER BOUND)",
            "up_bytes": UP_BYTES, "down_bytes": DOWN_BYTES,
            "iters": args.iters, "npes": npes,
            "rtt_p50_us": pct(0.50), "rtt_p90_us": pct(0.90),
            "rtt_p99_us": pct(0.99), "rtt_min_us": xs[0] * 1e3,
            "serial61_ms": pct(0.50) * NUM_LAYERS / 1e3,
        }
        print("\n=== NVSHMEM host-put RTT (cross-machine, UPPER BOUND) ===")
        print(f"  npes={npes}  up={UP_BYTES}B down={DOWN_BYTES}B")
        print(f"  rtt_p50={res['rtt_p50_us']:.1f}us  p99={res['rtt_p99_us']:.1f}us  "
              f"min={res['rtt_min_us']:.1f}us")
        print(f"  61-layer serial = {res['serial61_ms']:.3f} ms")
        print(f"  (ref: cross-node NCCL ~130us, RDMA floor ~4us)")
        if args.out:
            d = os.path.dirname(args.out)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
            print("  wrote", args.out)

    sh.free(buf)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
