# SPDX-License-Identifier: Apache-2.0
"""NVSHMEM device-initiated RTT bench launcher (Scheme A).

The round-trip loop runs entirely inside a CUDA kernel (rix_rtt_kernel.cu):
put_signal + signal_wait_until, no CPU in the data path. Python only does the
UID bootstrap (via torch.distributed) and calls rix_run_rtt_bench once.

This is the CLEAN latency number (device-initiated over IBGDA), the target the
whole exp02 was aiming for. Compare vs cross-node NCCL ~130us and RDMA floor
~4us.

Launch: run_nvshmem_dev.sh on both machines.
"""
import argparse
import ctypes
import json
import os

import torch
import torch.distributed as dist

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librix_rtt.so")
UID_BYTES = 128
NUM_LAYERS = 61


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    lib = ctypes.CDLL(_LIB, mode=ctypes.RTLD_GLOBAL)
    lib.rix_get_uniqueid.argtypes = [ctypes.c_void_p]
    lib.rix_init_with_uid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.rix_my_pe.restype = ctypes.c_int
    lib.rix_n_pes.restype = ctypes.c_int
    lib.rix_run_rtt_bench.argtypes = [ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_double)]
    lib.rix_run_rtt_bench.restype = ctypes.c_int

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, world
    torch.cuda.set_device(0)

    # UID bootstrap: rank0 generates, broadcast the 128-byte blob to rank1
    if rank == 0:
        buf = (ctypes.c_byte * UID_BYTES)()
        rc = lib.rix_get_uniqueid(ctypes.byref(buf))
        assert rc == 0, f"get_uniqueid rc={rc}"
        uid = bytes(buf)
    else:
        uid = b""
    obj = [uid]
    dist.broadcast_object_list(obj, src=0)
    uid = obj[0]

    ubuf = (ctypes.c_byte * UID_BYTES).from_buffer_copy(uid)
    rc = lib.rix_init_with_uid(ctypes.byref(ubuf), rank, world)
    assert rc == 0, f"init_with_uid rc={rc}"

    mype = lib.rix_my_pe()
    npes = lib.rix_n_pes()

    avg_us = ctypes.c_double(0.0)
    rc = lib.rix_run_rtt_bench(args.iters, args.warmup, ctypes.byref(avg_us))
    assert rc == 0, f"run_rtt_bench rc={rc}"

    if rank == 0:
        res = {
            "transport": "NVSHMEM device-initiated put_signal (IBGDA)",
            "up_bytes": 8580, "down_bytes": 8192,
            "iters": args.iters, "npes": npes,
            "rtt_avg_us": avg_us.value,
            "serial61_ms": avg_us.value * NUM_LAYERS / 1e3,
        }
        print("\n=== NVSHMEM device-initiated RTT (cross-machine, IBGDA) ===")
        print(f"  npes={npes}  up=8580B down=8192B  iters={args.iters}")
        print(f"  rtt_avg = {res['rtt_avg_us']:.2f} us")
        print(f"  61-layer serial = {res['serial61_ms']:.3f} ms")
        print(f"  (ref: cross-node NCCL ~130us, RDMA floor ~4us, same-machine IPC ~8us)")
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
