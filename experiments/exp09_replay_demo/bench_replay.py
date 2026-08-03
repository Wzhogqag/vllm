# SPDX-License-Identifier: Apache-2.0
"""exp09 replay demo — kernel-split GIN + torch host 打分/topk + recall 验证。

两机各 1 rank。B=16 固定,S 扫 [1024, 4096, 16384]。61 层串行,每层 host 循环。

Rank 0:
  produce_payload(layer_l)   # 本地随机生成 index_q_fp8/w/index_k 写进对称堆 up 区
  rix_r0_put_up(up_bytes)    # GIN put payload
  rix_r0_wait_down(cnt)      # 等 rank1 送回 topk
  read topk from down region
  同时本地也算 reference topk,diff 计算 recall

Rank 1:
  rix_r1_wait_up(cnt)        # 等 payload 到
  torch 读对称堆 up 区,做 bmm/weighted-sum/topk,写回 down 区
  rix_r1_put_down(down_bytes)

正确性:rank0 用同一份 payload + 同一份 K cache 本地 torch 算 topk,和 rank1 送回的
对比 recall@2048。
"""
import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from common import (
    INDEX_HEAD_DIM,
    INDEX_N_HEADS,
    INDEX_TOPK,
    UP_CAP,
    down_bytes,
    payload_offsets,
    up_bytes,
)


def _get_native_nccl_comm(group: dist.ProcessGroup) -> int:
    backend = group._get_backend(torch.device("cuda"))
    for attr in ("_comm_ptr", "_get_comm_handle", "_ncclComm", "get_nccl_comm"):
        if hasattr(backend, attr):
            v = getattr(backend, attr)
            handle = v() if callable(v) else v
            return int(handle)
    raise RuntimeError(f"can't find NCCL comm handle on {type(backend).__name__}")


def _load_lib() -> ctypes.CDLL:
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "librix_replay.so")
    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

    lib.rix_gin_init.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int64, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_void_p)]
    lib.rix_gin_init.restype = ctypes.c_int
    lib.rix_gin_finalize.argtypes = [ctypes.c_void_p]
    lib.rix_gin_finalize.restype = ctypes.c_int
    lib.rix_r0_put_up.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rix_r0_put_up.restype = ctypes.c_int
    lib.rix_r0_wait_down.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.rix_r0_wait_down.restype = ctypes.c_int
    lib.rix_r1_wait_up.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.rix_r1_wait_up.restype = ctypes.c_int
    lib.rix_r1_put_down.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rix_r1_put_down.restype = ctypes.c_int
    lib.rix_symmetric_buffer.argtypes = [ctypes.c_void_p]
    lib.rix_symmetric_buffer.restype = ctypes.c_void_p
    return lib


def _view_up_region(sym_base: int, B: int) -> dict:
    """对称堆 up 区上的 torch view(zero-copy)。返回 dict of tensors。"""
    offs = payload_offsets(B)
    # q_fp8: [B, 64, 128] as uint8 (fp8 e4m3 存字节相同)
    q_ptr = sym_base + offs["q_fp8"]
    w_ptr = sym_base + offs["weights"]
    k_ptr = sym_base + offs["index_k"]

    # 用 from_dlpack 造 view 太麻烦;直接用 torch.frombuffer(int -> tensor)-> view 是安全的
    # 但更可靠:UntypedStorage.from_buffer 从原始指针。这里我们干脆先 malloc 一个 device buffer,
    # 每层 memcpy 进对称堆(host driver 写清晰,零 aliasing 风险)。
    raise NotImplementedError("payload 生成不用 view 对称堆,改成明确的 D2D memcpy")


def _rand_payload(B: int, device: torch.device, gen: torch.Generator) -> dict:
    """生成一层的真实形状 payload。数值随机,shape/dtype 与真实一致。"""
    q_fp8 = torch.randint(0, 240, (B, INDEX_N_HEADS, INDEX_HEAD_DIM),
                          dtype=torch.uint8, device=device, generator=gen)
    weights = torch.randn(B, INDEX_N_HEADS, dtype=torch.bfloat16,
                          device=device, generator=gen)
    # index_k+scale:128 fp8 值(uint8)+ 4 字节 scale(fp32)
    # 简化:一整段 [B, 132] uint8,前 128 是 fp8 数据,后 4 是 fp32 scale 的字节
    idx_k = torch.randint(0, 240, (B, INDEX_HEAD_DIM), dtype=torch.uint8,
                          device=device, generator=gen)
    idx_k_scale = torch.rand(B, dtype=torch.float32, device=device, generator=gen) * 0.1 + 0.01
    return {"q_fp8": q_fp8, "weights": weights,
            "idx_k": idx_k, "idx_k_scale": idx_k_scale}


def _write_payload_to_symmetric(sym_ptr: int, B: int, payload: dict) -> None:
    """把生成的 payload D2D 拷贝到对称堆 up 区。"""
    offs = payload_offsets(B)
    # 三段拼接:q_fp8 | weights | (idx_k || idx_k_scale)
    q_fp8 = payload["q_fp8"].contiguous()
    weights = payload["weights"].contiguous()
    idx_k = payload["idx_k"].contiguous()
    scale = payload["idx_k_scale"].contiguous().view(torch.uint8).view(B, 4)  # fp32 bytes
    # idx_k_scale 每 B 有 1 个 fp32 → 4 字节,和 idx_k 各 128 字节拼成每 B 132 字节
    idx_k_full = torch.cat([idx_k, scale], dim=1).contiguous()

    dst_q = sym_ptr + offs["q_fp8"]
    dst_w = sym_ptr + offs["weights"]
    dst_k = sym_ptr + offs["index_k"]

    def _cudamemcpy(dst_ptr: int, src: torch.Tensor):
        n = src.numel() * src.element_size()
        # torch does not expose a public API for raw pointer copy;
        # cudart via ctypes is the cleanest here.
        cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
        # cudaMemcpy(dst, src, count, kind) with kind=cudaMemcpyDeviceToDevice=3
        cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_size_t, ctypes.c_int]
        cudart.cudaMemcpy.restype = ctypes.c_int
        rc = cudart.cudaMemcpy(dst_ptr, src.data_ptr(), n, 3)
        assert rc == 0, f"cudaMemcpy rc={rc}"

    _cudamemcpy(dst_q, q_fp8.view(-1))
    _cudamemcpy(dst_w, weights.view(-1))
    _cudamemcpy(dst_k, idx_k_full.view(-1))


def _read_payload_from_symmetric(sym_ptr: int, B: int, device: torch.device) -> dict:
    """rank1 从对称堆 up 区读出 payload;返回可参与 torch 计算的 tensor。"""
    offs = payload_offsets(B)
    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int

    q_fp8 = torch.empty(B, INDEX_N_HEADS, INDEX_HEAD_DIM,
                        dtype=torch.uint8, device=device)
    weights = torch.empty(B, INDEX_N_HEADS,
                          dtype=torch.bfloat16, device=device)
    idx_k_full = torch.empty(B, INDEX_HEAD_DIM + 4,
                             dtype=torch.uint8, device=device)

    def _cp(dst: torch.Tensor, src_ptr: int):
        rc = cudart.cudaMemcpy(dst.data_ptr(), src_ptr,
                               dst.numel() * dst.element_size(), 3)
        assert rc == 0, f"cudaMemcpy rc={rc}"

    _cp(q_fp8.view(-1), sym_ptr + offs["q_fp8"])
    _cp(weights.view(-1), sym_ptr + offs["weights"])
    _cp(idx_k_full.view(-1), sym_ptr + offs["index_k"])
    idx_k = idx_k_full[:, :INDEX_HEAD_DIM]
    scale = idx_k_full[:, INDEX_HEAD_DIM:].contiguous().view(-1).view(torch.float32).view(B)
    return {"q_fp8": q_fp8, "weights": weights, "idx_k": idx_k, "idx_k_scale": scale}


def _write_topk_to_symmetric(sym_ptr: int, B: int, topk: torch.Tensor) -> None:
    """topk [B, 2048] int32 → 对称堆下行区。"""
    assert topk.dtype == torch.int32 and topk.shape == (B, INDEX_TOPK)
    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int
    n = topk.numel() * topk.element_size()
    rc = cudart.cudaMemcpy(sym_ptr + UP_CAP, topk.contiguous().data_ptr(), n, 3)
    assert rc == 0


def _read_topk_from_symmetric(sym_ptr: int, B: int, device: torch.device) -> torch.Tensor:
    """rank0 从对称堆下行区读回 topk。"""
    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int
    topk = torch.empty(B, INDEX_TOPK, dtype=torch.int32, device=device)
    rc = cudart.cudaMemcpy(topk.data_ptr(), sym_ptr + UP_CAP,
                           topk.numel() * topk.element_size(), 3)
    assert rc == 0
    return topk


def _fp8_uint8_to_bf16(u: torch.Tensor) -> torch.Tensor:
    """把 fp8 e4m3 的 raw bytes(装成 uint8)近似 upcast 到 bf16。
    这里为简单起见:用 torch 内建的 float8_e4m3fn view。"""
    return u.view(torch.float8_e4m3fn).to(torch.bfloat16)


def _score_and_topk(payload: dict, k_cache: torch.Tensor,
                    softmax_scale: float, n_head_scale: float) -> torch.Tensor:
    """MQA 打分 + top-2048。

    payload:   q_fp8 [B, 64, 128] uint8, weights [B, 64] bf16,
               idx_k [B, 128] uint8, idx_k_scale [B] fp32
    k_cache:   [S+1, 128] bf16 (已包含当前 idx_k 追加)
    返回:      topk [B, 2048] int32
    """
    B = payload["q_fp8"].shape[0]
    q = _fp8_uint8_to_bf16(payload["q_fp8"])   # [B, 64, 128] bf16
    # k_cache: [S+1, 128] bf16
    # score[b, h, s] = q[b,h,:] · k_cache[s,:]
    # = einsum "bhd,sd->bhs"
    score = torch.einsum("bhd,sd->bhs", q, k_cache.to(torch.bfloat16))  # [B,64,S+1]
    # softmax_scale (1/sqrt(head_dim)) 和 n_head_scale 是配置常量;为了 recall 对比,
    # 只要 rank0 rank1 用同一份就行。
    logit = (payload["weights"].unsqueeze(-1) * softmax_scale * n_head_scale * score
             ).sum(dim=1)  # [B, S+1]
    # 真实场景 top-2048 需要 K cache 足够长;短序列时退化为全选(不满 2048 pad 到 2048)。
    k = min(INDEX_TOPK, logit.shape[1])
    idx = torch.topk(logit, k, dim=1).indices.to(torch.int32)  # [B, k]
    if k < INDEX_TOPK:
        pad = idx.new_zeros(idx.shape[0], INDEX_TOPK - k)  # 用 0 填充,recall 对比也一致
        idx = torch.cat([idx, pad], dim=1)
    return idx


def _recall(a: torch.Tensor, b: torch.Tensor) -> float:
    """两组 topk 索引集合的 recall(平均 over batch)。"""
    B = a.shape[0]
    total_hit = 0
    for i in range(B):
        set_a = set(a[i].tolist())
        set_b = set(b[i].tolist())
        total_hit += len(set_a & set_b)
    return total_hit / (B * INDEX_TOPK)


def rank0_loop(lib, ctx_ptr, B: int, S: int, layers: int,
               iters: int, warmup: int, sym_ptr: int,
               device: torch.device):
    """rank0 每层 = produce payload + put + wait + read topk (+ diff)。"""
    softmax_scale = INDEX_HEAD_DIM ** -0.5
    n_head_scale = INDEX_N_HEADS ** -0.5
    gen = torch.Generator(device=device).manual_seed(1234)

    # 本地 K cache:S 行历史 + 1 行当前(逐层追加)。这里我们简化 —— 每层重新
    # 生成 (S+1) 的 k_cache 用于本地 reference 打分,不去关心跨层增长。
    # 只关心"给定同一 payload 和 k_cache 时,两侧算的 topk 是否一致"。
    #
    # 但对时间对比,我们希望 rank0 也做同样的本地打分,占用一样多的时间,
    # 以便时序 apples-to-apples。所以每层 rank0 也算 reference topk。

    up = up_bytes(B)
    down = down_bytes(B)

    total_iters = warmup + iters
    up_signal = 0  # rank0 侧 SIG_DOWN 期望值(每层递增)
    recalls = []
    layer_times_us = []

    torch.cuda.synchronize()
    dist.barrier()

    import time
    for it in range(total_iters):
        t_iter0 = time.perf_counter_ns()

        for layer in range(layers):
            payload = _rand_payload(B, device, gen)
            _write_payload_to_symmetric(sym_ptr, B, payload)

            # 本地 reference K cache:再随机造一个(rank1 那边同 seed 会重造同样的)
            # 注意:这里 seed 是 layer+it 派生,确保两侧确定一致
            k_gen = torch.Generator(device=device).manual_seed(
                999 + it * layers + layer)
            k_cache = torch.randn(S + 1, INDEX_HEAD_DIM, dtype=torch.bfloat16,
                                  device=device, generator=k_gen)

            rc = lib.rix_r0_put_up(ctx_ptr, up)
            assert rc == 0, f"r0_put_up rc={rc}"
            up_signal += 1
            rc = lib.rix_r0_wait_down(ctx_ptr, up_signal)
            assert rc == 0, f"r0_wait_down rc={rc}"

            # 读回 rank1 送来的 topk
            topk_remote = _read_topk_from_symmetric(sym_ptr, B, device)

            # 本地也算 reference topk
            if it >= warmup:
                topk_local = _score_and_topk(payload, k_cache,
                                             softmax_scale, n_head_scale)
                r = _recall(topk_local.cpu(), topk_remote.cpu())
                recalls.append(r)

        t_iter1 = time.perf_counter_ns()
        if it >= warmup:
            layer_times_us.append((t_iter1 - t_iter0) / 1e3 / layers)

    return {
        "mean_layer_us": sum(layer_times_us) / len(layer_times_us),
        "min_layer_us": min(layer_times_us),
        "max_layer_us": max(layer_times_us),
        "mean_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "min_recall": min(recalls) if recalls else 0.0,
    }


def rank1_loop(lib, ctx_ptr, B: int, S: int, layers: int,
               iters: int, warmup: int, sym_ptr: int,
               device: torch.device):
    softmax_scale = INDEX_HEAD_DIM ** -0.5
    n_head_scale = INDEX_N_HEADS ** -0.5

    down = down_bytes(B)
    total_iters = warmup + iters
    down_signal = 0  # rank1 侧 SIG_UP 期望值

    torch.cuda.synchronize()
    dist.barrier()

    for it in range(total_iters):
        for layer in range(layers):
            down_signal += 1
            rc = lib.rix_r1_wait_up(ctx_ptr, down_signal)
            assert rc == 0, f"r1_wait_up rc={rc}"

            # 读 payload
            payload = _read_payload_from_symmetric(sym_ptr, B, device)

            # 与 rank0 同种子重造 k_cache
            k_gen = torch.Generator(device=device).manual_seed(
                999 + it * layers + layer)
            k_cache = torch.randn(S + 1, INDEX_HEAD_DIM, dtype=torch.bfloat16,
                                  device=device, generator=k_gen)

            topk = _score_and_topk(payload, k_cache,
                                   softmax_scale, n_head_scale)
            _write_topk_to_symmetric(sym_ptr, B, topk)

            rc = lib.rix_r1_put_down(ctx_ptr, down)
            assert rc == 0, f"r1_put_down rc={rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-lens", type=str, default="1024,4096,16384")
    ap.add_argument("--layers", type=int, default=61)
    ap.add_argument("--iters", type=int, default=20,
                    help="外循环 iters,每 iter 走完 61 层,收 mean/min/max")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--symmetric-bytes", type=int, default=16 << 20)
    ap.add_argument("--num-qps", type=int, default=1)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world == 2, f"exp09 只做 2-rank 跨机,{world=}"

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)

    x = torch.zeros(1, device=device)
    dist.all_reduce(x)
    torch.cuda.synchronize()

    comm_ptr = _get_native_nccl_comm(dist.group.WORLD)
    if rank == 0:
        print(f"[bench09] world=2 native NCCL comm ptr = 0x{comm_ptr:x}", flush=True)

    lib = _load_lib()
    ctx_ptr = ctypes.c_void_p(0)
    rc = lib.rix_gin_init(comm_ptr, rank, world,
                          args.symmetric_bytes, args.num_qps,
                          ctypes.byref(ctx_ptr))
    if rc != 0:
        print(f"[rank {rank}] rix_gin_init rc={rc}", file=sys.stderr, flush=True)
        sys.exit(1)
    sym_ptr = int(lib.rix_symmetric_buffer(ctx_ptr))
    if rank == 0:
        print(f"[bench09] symmetric buffer @ 0x{sym_ptr:x}", flush=True)
    dist.barrier()

    seq_lens = [int(s) for s in args.seq_lens.split(",") if s.strip()]
    all_results = []

    for S in seq_lens:
        if rank == 0:
            print(f"\n=== B={args.batch}  S={S}  layers={args.layers}  "
                  f"iters={args.iters}  warmup={args.warmup} ===", flush=True)
        dist.barrier()

        if rank == 0:
            stats = rank0_loop(lib, ctx_ptr, args.batch, S, args.layers,
                               args.iters, args.warmup, sym_ptr, device)
            entry = {"B": args.batch, "S": S, "layers": args.layers,
                     **stats,
                     "serial_layers_ms": stats["mean_layer_us"] * args.layers / 1e3}
            all_results.append(entry)
            print(f"  mean {stats['mean_layer_us']:.2f}μs/层  "
                  f"[min {stats['min_layer_us']:.2f} max {stats['max_layer_us']:.2f}]  "
                  f"61 层 {entry['serial_layers_ms']:.3f}ms  "
                  f"recall mean={stats['mean_recall']:.4f}  min={stats['min_recall']:.4f}",
                  flush=True)
        else:
            rank1_loop(lib, ctx_ptr, args.batch, S, args.layers,
                       args.iters, args.warmup, sym_ptr, device)
        dist.barrier()

    if rank == 0:
        print("\n============= exp09 replay demo ============")
        for r in all_results:
            print(r)
        if args.out:
            Path(os.path.dirname(args.out) or ".").mkdir(exist_ok=True, parents=True)
            with open(args.out, "w") as f:
                json.dump({"transport": "NCCL GIN GDAKI (kernel-split)",
                           "B": args.batch,
                           "iters": args.iters, "warmup": args.warmup,
                           "layers": args.layers,
                           "note": "torch host 打分 + topk;payload/K cache 每层随机",
                           "results": all_results}, f, indent=2)
            print(f"  wrote {args.out}")

    lib.rix_gin_finalize(ctx_ptr)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
