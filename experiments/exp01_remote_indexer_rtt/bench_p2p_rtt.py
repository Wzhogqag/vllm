# SPDX-License-Identifier: Apache-2.0
"""Scheme A single-layer round-trip latency on H200 NVLink.

rank0 = local (main compute), rank1 = remote indexer.

Per measured iteration (one indexer layer, one decode step):
  rank0 sends uplink  (index_q_fp8 + weights + index_k, 8580 B/token)  -> rank1
  rank1 immediately sends downlink (top-k ids, 8192 B/token)           -> rank0

We measure the rank0-side RTT with CUDA events. The remote does NO scoring here
(that is bench_compute_sim's job, and it grows with seq_len); this isolates the
pure transport floor to compare against the H800 ~70us baseline.

We also measure uplink-only and downlink-only to see which half dominates.

Sweep batch B (concurrent decode tokens). H800 was flat 256B-32KB
(launch-bound); this checks whether H200 stays flat at these payloads.
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

import common


def _sync_all():
    dist.barrier()
    torch.cuda.synchronize()


def bench_rtt(batch: int, iters: int, warmup: int, device):
    """Full round-trip: rank0 up -> rank1 down -> rank0. Timed on rank0."""
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


def bench_oneway(batch: int, iters: int, warmup: int, device, direction: str):
    """Uplink-only or downlink-only, timed on the sender."""
    rank = dist.get_rank()
    up = common.make_scheme_a_uplink(batch, device)
    down = common.make_scheme_a_downlink(batch, device)

    # uplink: rank0 -> rank1 ; downlink: rank1 -> rank0
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
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    rank, local_rank, _ = common.init_dist()
    device = torch.device(f"cuda:{local_rank}")
    common._assert_dims()

    results = {
        "meta": {
            "scheme": "A",
            "up_bytes_per_token": common.UP_BYTES_PER_TOKEN,
            "down_bytes_per_token": common.DOWN_BYTES_PER_TOKEN,
            "num_indexer_layers": common.NUM_INDEXER_LAYERS,
            "iters": args.iters,
            "warmup": args.warmup,
            "gpu_name": torch.cuda.get_device_name(local_rank),
        },
        "by_batch": {},
    }

    for b in args.batches:
        rtt = bench_rtt(b, args.iters, args.warmup, device)
        up = bench_oneway(b, args.iters, args.warmup, device, "up")
        down = bench_oneway(b, args.iters, args.warmup, device, "down")
        if rank == 0:
            entry = {
                "up_bytes_total": common.UP_BYTES_PER_TOKEN * b,
                "down_bytes_total": common.DOWN_BYTES_PER_TOKEN * b,
                "rtt": common.percentiles(rtt),
                "uplink": common.percentiles(up),
            }
            # downlink is timed on rank1 (the sender); gather it to rank0.
            results["by_batch"][str(b)] = entry
        # downlink samples live on rank1; print there and also stash via file.
        if rank == 1 and down is not None:
            dp = common.percentiles(down)
            os.makedirs("/tmp/_rtt_down", exist_ok=True)
            with open(f"/tmp/_rtt_down/b{b}.json", "w") as f:
                json.dump(dp, f)

    dist.barrier()

    if rank == 0:
        # fold in downlink percentiles produced by rank1
        for b in args.batches:
            p = f"/tmp/_rtt_down/b{b}.json"
            if os.path.exists(p):
                with open(p) as f:
                    results["by_batch"][str(b)]["downlink"] = json.load(f)

        print("\n=== Scheme A RTT on", results["meta"]["gpu_name"], "===")
        print(f"{'B':>4} {'up_B':>8} {'down_B':>8} "
              f"{'rtt_p50':>9} {'rtt_p99':>9} {'up_p50':>8} {'down_p50':>8}  (us)")
        for b in args.batches:
            e = results["by_batch"][str(b)]
            dn = e.get("downlink", {})
            print(f"{b:>4} {e['up_bytes_total']:>8} {e['down_bytes_total']:>8} "
                  f"{e['rtt']['p50_us']:>9.1f} {e['rtt']['p99_us']:>9.1f} "
                  f"{e['uplink']['p50_us']:>8.1f} {dn.get('p50_us', float('nan')):>8.1f}")

        # 61-layer serial extrapolation from single-layer RTT p50 at B=1
        b1 = results["by_batch"].get("1")
        if b1:
            per_layer_us = b1["rtt"]["p50_us"]
            results["serial_61_layer_ms_from_b1_p50"] = per_layer_us * common.NUM_INDEXER_LAYERS / 1e3
            print(f"\n61-layer serial (B=1 p50 x 61) = "
                  f"{results['serial_61_layer_ms_from_b1_p50']:.3f} ms  (H800 ref ~4.3 ms)")

        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print("wrote", args.out)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
