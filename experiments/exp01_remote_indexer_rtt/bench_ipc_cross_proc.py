# SPDX-License-Identifier: Apache-2.0
"""Cross-process raw peer copy via CUDA IPC handles (same machine, 2 GPUs).

Isolates the "separate process" variable. bench_raw_peer_rtt.py measured a
single process owning both GPUs (~16us RTT). Here rank0 owns GPU0 and rank1
owns GPU1 as two independent processes; they exchange CUDA IPC memory handles
once at setup, then rank0 does cudaMemcpyPeerAsync straight into rank1's memory
(uplink) and reads back from rank1's memory (downlink).

Transport path is still NVLink DMA; the only new variable vs the single-process
bench is process separation. Comparison ladder:
  single-process peer copy   ~16 us   (bench_raw_peer_rtt.py)
  cross-process IPC peer copy    ?     (this)
  NCCL send/recv            ~100 us    (bench_p2p_rtt.py)

Timing method: each direction is timed independently on the initiator's stream
with CUDA events, matching the single-process bench's up_p50 / down_p50 columns.
The handle exchange (via the gloo CPU side-channel) happens once and is NOT
timed. rank0 is the initiator for both directions (it opens rank1's handles),
so all copies are launched from one process — the peer copy itself is what we
measure, not cross-process kernel handshakes.
"""
import argparse
import json
import os

import cupy as cp
import torch
import torch.distributed as dist

import common

UP0 = common.BYTES_INDEX_Q_FP8
UP1 = common.BYTES_INDEX_WEIGHTS
UP2 = common.BYTES_INDEX_K
UP_TOTAL = common.UP_BYTES_PER_TOKEN
DOWN_TOTAL = common.DOWN_BYTES_PER_TOKEN


def _handle_to_bytes(mem) -> bytes:
    return bytes(cp.cuda.runtime.ipcGetMemHandle(mem.ptr))


def _exchange_handles(local_handles: list[bytes], rank: int):
    """All-gather raw IPC handle bytes over the gloo CPU group."""
    obj = [local_handles]
    gathered = [None, None]
    dist.all_gather_object(gathered, local_handles)
    return gathered  # gathered[r] = list of handle-bytes from rank r


def bench_cross_proc(batch: int, iters: int, warmup: int, packed: bool,
                     rank: int, dev: int):
    """rank0 initiates both directions into/from rank1's IPC-shared buffers.

    rank1 only allocates its buffers, shares handles, and waits at barriers.
    """
    stream = cp.cuda.Stream(non_blocking=True)

    if packed:
        up_sizes = [UP_TOTAL * batch]
    else:
        up_sizes = [UP0 * batch, UP1 * batch, UP2 * batch]
    down_sizes = [DOWN_TOTAL * batch]

    # rank1 owns the "remote" side: uplink DEST buffers + downlink SRC buffers.
    # rank0 owns the "local" side: uplink SRC buffers + downlink DEST buffers.
    if rank == 1:
        remote_up_dst = [cp.cuda.alloc(s) for s in up_sizes]   # uplink lands here
        remote_down_src = [cp.cuda.alloc(s) for s in down_sizes]  # downlink starts here
        my_handles = [_handle_to_bytes(m) for m in remote_up_dst + remote_down_src]
    else:
        my_handles = []

    all_h = _exchange_handles(my_handles, rank)

    if rank == 0:
        local_up_src = [cp.cuda.alloc(s) for s in up_sizes]
        local_down_dst = [cp.cuda.alloc(s) for s in down_sizes]
        # open rank1's handles: first len(up_sizes) are uplink dst, rest downlink src
        r1 = all_h[1]
        opened = [cp.cuda.runtime.ipcOpenMemHandle(h) for h in r1]
        remote_up_dst_ptr = opened[: len(up_sizes)]
        remote_down_src_ptr = opened[len(up_sizes):]

        def do_up():
            for src, dptr, n in zip(local_up_src, remote_up_dst_ptr, up_sizes):
                cp.cuda.runtime.memcpyPeerAsync(dptr, 1, src.ptr, 0, n, stream.ptr)

        def do_down():
            for dst, sptr, n in zip(local_down_dst, remote_down_src_ptr, down_sizes):
                cp.cuda.runtime.memcpyPeerAsync(dst.ptr, 0, sptr, 1, n, stream.ptr)

        def time_dir(fn):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            for _ in range(warmup):
                fn()
            stream.synchronize()
            s = []
            for _ in range(iters):
                start.record(stream)
                fn()
                end.record(stream)
                end.synchronize()
                s.append(cp.cuda.get_elapsed_time(start, end))  # ms
            return s

        up_ms = time_dir(do_up)
        down_ms = time_dir(do_down)
        # RTT estimate = up + down back-to-back (same launch-per-copy structure)

        def do_rtt():
            do_up()
            do_down()
        rtt_ms = time_dir(do_rtt)

        # close handles
        for p in opened:
            cp.cuda.runtime.ipcCloseMemHandle(p)

        dist.barrier()
        return {
            "uplink": common.percentiles(up_ms),
            "downlink": common.percentiles(down_ms),
            "rtt": common.percentiles(rtt_ms),
        }
    else:
        dist.barrier()  # match rank0's post-loop barrier; keep buffers alive
        del remote_up_dst, remote_down_src
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    # 2-process init; use gloo for the CPU-side handle exchange (NCCL not needed).
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, world
    dev = rank
    cp.cuda.Device(dev).use()
    torch.cuda.set_device(dev)
    common._assert_dims()

    results = {"meta": {"method": "cross-process CUDA IPC peer copy",
                        "gpu_name": torch.cuda.get_device_name(dev),
                        "iters": args.iters, "warmup": args.warmup},
               "unpacked": {}, "packed": {}}

    for packed in (False, True):
        key = "packed" if packed else "unpacked"
        for b in args.batches:
            r = bench_cross_proc(b, args.iters, args.warmup, packed, rank, dev)
            if rank == 0:
                results[key][str(b)] = r

    if rank == 0:
        print(f"\n=== Cross-process IPC peer copy on {results['meta']['gpu_name']} ===")
        print("(2 processes, CUDA IPC handles; single-proc ref ~16us, NCCL ~100us)")
        for key in ("unpacked", "packed"):
            tag = "3 copies" if key == "unpacked" else "1 copy (packed)"
            print(f"\n-- uplink = {tag} --")
            print(f"{'B':>4} {'rtt_p50':>9} {'rtt_p99':>9} {'up_p50':>8} {'down_p50':>9}  (us)")
            for b in args.batches:
                e = results[key][str(b)]
                print(f"{b:>4} {e['rtt']['p50_us']:>9.1f} {e['rtt']['p99_us']:>9.1f} "
                      f"{e['uplink']['p50_us']:>8.1f} {e['downlink']['p50_us']:>9.1f}")
        u1 = results["unpacked"]["1"]["rtt"]["p50_us"]
        p1 = results["packed"]["1"]["rtt"]["p50_us"]
        results["serial61_unpacked_ms"] = u1 * common.NUM_INDEXER_LAYERS / 1e3
        results["serial61_packed_ms"] = p1 * common.NUM_INDEXER_LAYERS / 1e3
        print(f"\nB=1 RTT: unpacked {u1:.1f} us, packed {p1:.1f} us")
        print(f"61-layer serial: unpacked {results['serial61_unpacked_ms']:.3f} ms, "
              f"packed {results['serial61_packed_ms']:.3f} ms  (single-proc ~0.82-1.0 ms)")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print("wrote", args.out)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
