# SPDX-License-Identifier: Apache-2.0
"""Indexer remote-comm RTT bench (DeepEP IBGDA primitives).

Round-trip loop runs in a CUDA kernel using DeepEP's low-level IBGDA device
functions (put_nbi_warp + amo_nonfetch_add + quiet + ld_acquire poll), which
bypass the standard nvshmem_putmem_signal API that failed with rc=700 here.

Payload is parameterized (--up-bytes / --down-bytes) so the same framework
covers Scheme A (8580/8192), Scheme B (17408/8192), or any future variant.
"""
import argparse
import ctypes
import json
import os

import torch
import torch.distributed as dist

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librix_comm.so")
UID_BYTES = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up-bytes", type=int, default=8580)    # Scheme A uplink
    ap.add_argument("--down-bytes", type=int, default=8192)  # Scheme A downlink
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--layers", type=int, default=61)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    lib = ctypes.CDLL(_LIB, mode=ctypes.RTLD_GLOBAL)
    lib.rix_get_uniqueid.argtypes = [ctypes.c_void_p]
    lib.rix_init_with_uid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.rix_my_pe.restype = ctypes.c_int
    lib.rix_n_pes.restype = ctypes.c_int
    lib.rix_run_rtt_bench.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    lib.rix_run_rtt_bench.restype = ctypes.c_int

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, world
    torch.cuda.set_device(0)

    if rank == 0:
        buf = (ctypes.c_byte * UID_BYTES)()
        assert lib.rix_get_uniqueid(ctypes.byref(buf)) == 0
        uid = bytes(buf)
    else:
        uid = b""
    obj = [uid]
    dist.broadcast_object_list(obj, src=0)
    uid = obj[0]
    ubuf = (ctypes.c_byte * UID_BYTES).from_buffer_copy(uid)
    assert lib.rix_init_with_uid(ctypes.byref(ubuf), rank, world) == 0

    npes = lib.rix_n_pes()
    avg_us = ctypes.c_double(0.0)
    rc = lib.rix_run_rtt_bench(args.up_bytes, args.down_bytes,
                               args.iters, args.warmup, ctypes.byref(avg_us))
    assert rc == 0, f"run_rtt_bench rc={rc}"

    if rank == 0:
        res = {
            "transport": "DeepEP IBGDA primitives (put_nbi + amo + poll)",
            "up_bytes": args.up_bytes, "down_bytes": args.down_bytes,
            "iters": args.iters, "npes": npes,
            "rtt_avg_us": avg_us.value,
            "serial_layers_ms": avg_us.value * args.layers / 1e3,
            "layers": args.layers,
        }
        print(f"\n=== Indexer-comm RTT (IBGDA, up={args.up_bytes} down={args.down_bytes}) ===")
        print(f"  npes={npes}  rtt_avg = {res['rtt_avg_us']:.2f} us")
        print(f"  {args.layers}-layer serial = {res['serial_layers_ms']:.3f} ms")
        print(f"  (ref: NCCL ~130us, RDMA floor ~4us, IPC same-machine ~8us)")
        if args.out:
            d = os.path.dirname(args.out)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
            print("  wrote", args.out)

    lib.rix_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
