# SPDX-License-Identifier: Apache-2.0
"""Raw P2P copy RTT + uplink packing, single-process two-GPU.

Complements bench_p2p_rtt.py (NCCL send/recv, 2-process). This one uses cupy's
cudaMemcpyPeerAsync directly to measure the hardware/driver P2P floor with no
NCCL communicator and no cross-process sync. Comparing the two isolates
"NCCL + multi-process overhead" from "raw NVLink P2P latency", and lets us
check the H800 ~70us baseline on the same kind of measurement (peer copy).

Two questions:
  (1) raw peer-copy RTT vs the ~100us NCCL RTT and the H800 ~70us baseline.
  (2) uplink packing: 3 separate copies (q_fp8 / weights / index_k) vs a single
      copy of one contiguous 8580 B/token buffer.

Single process owns both GPUs, so we do NOT use torchrun here.
"""
import argparse
import json

import cupy as cp
import torch

import common

UP0 = common.BYTES_INDEX_Q_FP8      # 8192
UP1 = common.BYTES_INDEX_WEIGHTS    # 256
UP2 = common.BYTES_INDEX_K          # 132
UP_TOTAL = common.UP_BYTES_PER_TOKEN    # 8580
DOWN_TOTAL = common.DOWN_BYTES_PER_TOKEN  # 8192


def _alloc(dev: int, nbytes: int):
    with cp.cuda.Device(dev):
        return cp.cuda.alloc(nbytes)


def _time_loop(fn, stream, iters: int, warmup: int) -> list[float]:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    for _ in range(warmup):
        fn()
    stream.synchronize()
    samples = []
    for _ in range(iters):
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        samples.append(cp.cuda.get_elapsed_time(start, end))  # ms
    return samples  # ms; percentiles() converts to us


def bench_raw_rtt(batch: int, iters: int, warmup: int, packed: bool):
    """RTT = uplink (dev0->dev1) + downlink (dev1->dev0), timed on dev0 stream.

    packed=False: 3 separate uplink copies. packed=True: single merged buffer.
    """
    dev0, dev1 = 0, 1
    with cp.cuda.Device(dev0):
        stream = cp.cuda.Stream(non_blocking=True)

    up_total_b = UP_TOTAL * batch
    down_total_b = DOWN_TOTAL * batch

    # uplink source buffers on dev0, dest buffers on dev1
    if packed:
        src_up = [_alloc(dev0, up_total_b)]
        dst_up = [_alloc(dev1, up_total_b)]
        sizes_up = [up_total_b]
    else:
        sizes_up = [UP0 * batch, UP1 * batch, UP2 * batch]
        src_up = [_alloc(dev0, s) for s in sizes_up]
        dst_up = [_alloc(dev1, s) for s in sizes_up]

    # downlink: dev1 -> dev0 (single buffer, topk ids)
    src_down = _alloc(dev1, down_total_b)
    dst_down = _alloc(dev0, down_total_b)

    def one_rtt():
        # uplink dev0->dev1
        for s, d, n in zip(src_up, dst_up, sizes_up):
            cp.cuda.runtime.memcpyPeerAsync(d.ptr, dev1, s.ptr, dev0, n, stream.ptr)
        # downlink dev1->dev0
        cp.cuda.runtime.memcpyPeerAsync(
            dst_down.ptr, dev0, src_down.ptr, dev1, down_total_b, stream.ptr
        )

    with cp.cuda.Device(dev0):
        samples = _time_loop(one_rtt, stream, iters, warmup)
    return samples


def bench_raw_oneway(batch: int, iters: int, warmup: int, direction: str, packed: bool):
    dev0, dev1 = 0, 1
    with cp.cuda.Device(dev0):
        stream = cp.cuda.Stream(non_blocking=True)

    if direction == "up":
        if packed:
            sizes = [UP_TOTAL * batch]
        else:
            sizes = [UP0 * batch, UP1 * batch, UP2 * batch]
        src = [_alloc(dev0, s) for s in sizes]
        dst = [_alloc(dev1, s) for s in sizes]
        sdev, ddev = dev0, dev1
    else:  # down
        sizes = [DOWN_TOTAL * batch]
        src = [_alloc(dev1, s) for s in sizes]
        dst = [_alloc(dev0, s) for s in sizes]
        sdev, ddev = dev1, dev0

    def one():
        for s, d, n in zip(src, dst, sizes):
            cp.cuda.runtime.memcpyPeerAsync(d.ptr, ddev, s.ptr, sdev, n, stream.ptr)

    with cp.cuda.Device(dev0):
        samples = _time_loop(one, stream, iters, warmup)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    common._assert_dims()
    # enable P2P access both ways
    for a, b in ((0, 1), (1, 0)):
        with cp.cuda.Device(a):
            try:
                cp.cuda.runtime.deviceEnablePeerAccess(b)
            except cp.cuda.runtime.CUDARuntimeError:
                pass  # already enabled

    gpu_name = torch.cuda.get_device_name(0)
    results = {
        "meta": {
            "method": "raw cudaMemcpyPeerAsync, single-process 2-GPU",
            "up_bytes_per_token": UP_TOTAL,
            "down_bytes_per_token": DOWN_TOTAL,
            "iters": args.iters, "warmup": args.warmup, "gpu_name": gpu_name,
        },
        "unpacked": {}, "packed": {},
    }

    for packed in (False, True):
        key = "packed" if packed else "unpacked"
        for b in args.batches:
            rtt = bench_raw_rtt(b, args.iters, args.warmup, packed)
            up = bench_raw_oneway(b, args.iters, args.warmup, "up", packed)
            down = bench_raw_oneway(b, args.iters, args.warmup, "down", packed)
            results[key][str(b)] = {
                "rtt": common.percentiles(rtt),
                "uplink": common.percentiles(up),
                "downlink": common.percentiles(down),
            }

    print(f"\n=== Raw peer-copy RTT on {gpu_name} ===")
    print("(single-process 2-GPU, cudaMemcpyPeerAsync; H800 ref ~70us peer copy)")
    for key in ("unpacked", "packed"):
        tag = "3 sends" if key == "unpacked" else "1 send (packed)"
        print(f"\n-- uplink = {tag} --")
        print(f"{'B':>4} {'rtt_p50':>9} {'rtt_p99':>9} {'up_p50':>8} {'down_p50':>9}  (us)")
        for b in args.batches:
            e = results[key][str(b)]
            print(f"{b:>4} {e['rtt']['p50_us']:>9.1f} {e['rtt']['p99_us']:>9.1f} "
                  f"{e['uplink']['p50_us']:>8.1f} {e['downlink']['p50_us']:>9.1f}")

    # packing win at B=1 and 61-layer extrapolation
    u1 = results["unpacked"]["1"]["rtt"]["p50_us"]
    p1 = results["packed"]["1"]["rtt"]["p50_us"]
    results["b1_rtt_unpacked_us"] = u1
    results["b1_rtt_packed_us"] = p1
    results["serial61_unpacked_ms"] = u1 * common.NUM_INDEXER_LAYERS / 1e3
    results["serial61_packed_ms"] = p1 * common.NUM_INDEXER_LAYERS / 1e3
    print(f"\nB=1 single-layer RTT: unpacked {u1:.1f} us -> packed {p1:.1f} us")
    print(f"61-layer serial: unpacked {results['serial61_unpacked_ms']:.3f} ms"
          f" -> packed {results['serial61_packed_ms']:.3f} ms  (H800 ref ~4.3 ms)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
