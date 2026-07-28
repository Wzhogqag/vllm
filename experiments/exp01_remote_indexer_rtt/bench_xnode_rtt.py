# SPDX-License-Identifier: Apache-2.0
"""Scheme A single-layer RTT ACROSS TWO MACHINES over RoCE (NCCL send/recv).

rank0 = local node (machine A), rank1 = remote node (machine B). Each node uses
one local GPU. This is the production-shaped measurement: local and remote are
separate machines, traffic crosses the RoCE fabric via GPUDirect RDMA (if the
NCCL_IB_* env is set correctly) rather than NVLink.

Comparison ladder (H200, B=1 single-layer p50):
  same-machine NVLink IPC peer copy   ~8 us   (bench_ipc_cross_proc.py)
  same-machine NCCL send/recv        ~100 us  (bench_p2p_rtt.py)
  cross-machine RoCE NCCL             ???      (this)

The bench body is identical to bench_p2p_rtt.py (transport-agnostic dist calls);
only device selection differs: each rank uses its LOCAL GPU, chosen via the
CUDA_VISIBLE_DEVICES the launcher pins (so torchrun's LOCAL_RANK maps to a free
card on each node independently).

Launch with run_xnode.sh on BOTH machines (node 0 on A, node 1 on B).
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

import common


def init_xnode():
    """Cross-node init. Each node runs 1 process (1 rank) using its local GPU.

    The launcher pins CUDA_VISIBLE_DEVICES to a single free card per node, so
    the visible device index is always 0 here.
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"need exactly 2 ranks (one per machine), got {world}"
    torch.cuda.set_device(0)  # launcher pinned one visible card per node
    return rank, torch.device("cuda:0")


def _sync_all():
    dist.barrier()
    torch.cuda.synchronize()


def bench_rtt(batch, iters, warmup, device):
    rank = dist.get_rank()
    up = common.make_scheme_a_uplink(batch, device)
    down = common.make_scheme_a_downlink(batch, device)

    def one_iter():
        if rank == 0:
            for t in up:
                dist.send(t, dst=1)
            dist.recv(down, src=1)
        else:
            for t in up:
                dist.recv(t, src=0)
            dist.send(down, dst=0)

    for _ in range(warmup):
        one_iter()
    _sync_all()
    samples = []
    for _ in range(iters):
        timer = common.CudaTimer()
        with timer:
            one_iter()
        samples.append(timer.ms())
    _sync_all()
    return samples if rank == 0 else None


def bench_oneway(batch, iters, warmup, device, direction):
    rank = dist.get_rank()
    up = common.make_scheme_a_uplink(batch, device)
    down = common.make_scheme_a_downlink(batch, device)
    sender = 0 if direction == "up" else 1
    tensors = up if direction == "up" else [down]
    dst = 1 - sender

    def one_iter():
        if rank == sender:
            for t in tensors:
                dist.send(t, dst=dst)
        else:
            for t in tensors:
                dist.recv(t, src=sender)

    for _ in range(warmup):
        one_iter()
    _sync_all()
    samples = []
    for _ in range(iters):
        timer = common.CudaTimer()
        with timer:
            one_iter()
        samples.append(timer.ms())
    _sync_all()
    return samples if rank == sender else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    rank, device = init_xnode()
    common._assert_dims()

    results = {"meta": {"scheme": "A", "transport": "cross-machine RoCE NCCL",
                        "up_bytes_per_token": common.UP_BYTES_PER_TOKEN,
                        "down_bytes_per_token": common.DOWN_BYTES_PER_TOKEN,
                        "num_indexer_layers": common.NUM_INDEXER_LAYERS,
                        "iters": args.iters, "warmup": args.warmup,
                        "gpu_name": torch.cuda.get_device_name(0),
                        "nccl_ib_hca": os.environ.get("NCCL_IB_HCA", "(unset)"),
                        "nccl_ib_gid": os.environ.get("NCCL_IB_GID_INDEX", "(unset)")},
               "by_batch": {}}

    for b in args.batches:
        rtt = bench_rtt(b, args.iters, args.warmup, device)
        up = bench_oneway(b, args.iters, args.warmup, device, "up")
        down = bench_oneway(b, args.iters, args.warmup, device, "down")
        # downlink is timed on rank1 (the sender); ship its percentiles to rank0
        # over the process group (cross-machine: no shared /tmp).
        down_pct = common.percentiles(down) if (rank == 1 and down is not None) else None
        gathered = [None, None]
        dist.all_gather_object(gathered, down_pct)
        if rank == 0:
            results["by_batch"][str(b)] = {
                "up_bytes_total": common.UP_BYTES_PER_TOKEN * b,
                "down_bytes_total": common.DOWN_BYTES_PER_TOKEN * b,
                "rtt": common.percentiles(rtt),
                "uplink": common.percentiles(up),
                "downlink": gathered[1],
            }

    dist.barrier()

    if rank == 0:
        m = results["meta"]
        print(f"\n=== Cross-machine RoCE RTT on {m['gpu_name']} ===")
        print(f"(NCCL_IB_HCA={m['nccl_ib_hca']} GID={m['nccl_ib_gid']}; "
              f"same-machine ref: NVLink IPC ~8us, NCCL ~100us)")
        print(f"{'B':>4} {'up_B':>8} {'down_B':>8} "
              f"{'rtt_p50':>9} {'rtt_p99':>9} {'up_p50':>8} {'down_p50':>8}  (us)")
        for b in args.batches:
            e = results["by_batch"][str(b)]
            dn = e.get("downlink", {})
            print(f"{b:>4} {e['up_bytes_total']:>8} {e['down_bytes_total']:>8} "
                  f"{e['rtt']['p50_us']:>9.1f} {e['rtt']['p99_us']:>9.1f} "
                  f"{e['uplink']['p50_us']:>8.1f} {dn.get('p50_us', float('nan')):>8.1f}")
        b1 = results["by_batch"].get("1")
        if b1:
            per = b1["rtt"]["p50_us"]
            results["serial61_ms_from_b1_p50"] = per * common.NUM_INDEXER_LAYERS / 1e3
            print(f"\n61-layer serial (B=1 p50 x61) = "
                  f"{results['serial61_ms_from_b1_p50']:.3f} ms  "
                  f"(same-machine NVLink ~0.5 ms, H800 ~4.3 ms)")
        if args.out:
            d = os.path.dirname(args.out)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print("wrote", args.out)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
