# SPDX-License-Identifier: Apache-2.0
"""Remote lightning indexer RTT bench — shared helpers.

Dimensions are hard-coded from /models/deepseek-v3.2/config.json (verified
2026-07-24) so the bench does not depend on the model being loadable:

  hidden_size=7168  index_head_dim=128  index_n_heads=64  index_topk=2048
  q_lora_rank=1536  qk_rope_head_dim=64  num_hidden_layers=61
  first_k_dense_replace=3  index_topk_freq default 1  ->  61 indexer layers

Scheme A (projection stays local): every decode step, per layer, per token the
local rank sends the indexer query and the freshly-projected index K up to the
remote rank, which scores against its K cache and returns the top-k ids.
"""
import torch
import torch.distributed as dist

# --- verified model dims ---
HIDDEN_SIZE = 7168
INDEX_HEAD_DIM = 128
INDEX_N_HEADS = 64
INDEX_TOPK = 2048
Q_LORA_RANK = 1536
QK_ROPE_HEAD_DIM = 64
NUM_HIDDEN_LAYERS = 61
FIRST_K_DENSE_REPLACE = 3
QUANT_BLOCK_SIZE = 128
NUM_INDEXER_LAYERS = 61  # index_topk_freq=1 -> every layer carries an indexer

# --- Scheme A over-wire byte accounting (per token, per layer) ---
# fp8 e4m3 = 1 byte/elem; fp32 = 4 bytes/elem; int32 = 4 bytes/elem.
BYTES_INDEX_Q_FP8 = INDEX_N_HEADS * INDEX_HEAD_DIM * 1          # 8192
BYTES_INDEX_WEIGHTS = INDEX_N_HEADS * 4                         # 256
# index_k: one new key per token = head_dim fp8 + one fp32 scale per 128 elems
BYTES_INDEX_K = INDEX_HEAD_DIM * 1 + (INDEX_HEAD_DIM // QUANT_BLOCK_SIZE) * 4  # 132
UP_BYTES_PER_TOKEN = BYTES_INDEX_Q_FP8 + BYTES_INDEX_WEIGHTS + BYTES_INDEX_K   # 8580
DOWN_BYTES_PER_TOKEN = INDEX_TOPK * 4                                          # 8192


def _assert_dims():
    assert BYTES_INDEX_Q_FP8 == 8192, BYTES_INDEX_Q_FP8
    assert BYTES_INDEX_WEIGHTS == 256, BYTES_INDEX_WEIGHTS
    assert BYTES_INDEX_K == 132, BYTES_INDEX_K
    assert UP_BYTES_PER_TOKEN == 8580, UP_BYTES_PER_TOKEN
    assert DOWN_BYTES_PER_TOKEN == 8192, DOWN_BYTES_PER_TOKEN
    assert NUM_INDEXER_LAYERS == 61, NUM_INDEXER_LAYERS


def init_dist():
    """torchrun-launched 2-rank init. Returns (rank, local_rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"need exactly 2 ranks, got {world_size}"
    local_rank = rank  # single node, one process per GPU
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def make_scheme_a_uplink(batch: int, device) -> list[torch.Tensor]:
    """The three tensors the local rank sends up per step (batched over B tokens).

    Kept as separate tensors (not one flat buffer) to mirror the real dtypes;
    NCCL launches one send per tensor, matching production where they are
    distinct allocations. index_k scale is folded into the fp8 buffer's trailing
    bytes in production, but for wire timing a single fp8 tensor of the right
    total byte count is what matters.
    """
    index_q_fp8 = torch.empty(
        (batch, INDEX_N_HEADS, INDEX_HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
    )
    index_weights = torch.empty((batch, INDEX_N_HEADS), dtype=torch.float32, device=device)
    # index_k: 132 bytes/token -> represent as 132 uint8 to get exact wire size.
    index_k = torch.empty((batch, BYTES_INDEX_K), dtype=torch.uint8, device=device)
    return [index_q_fp8, index_weights, index_k]


def make_scheme_a_downlink(batch: int, device) -> torch.Tensor:
    """The top-k indices the remote rank returns per step."""
    return torch.empty((batch, INDEX_TOPK), dtype=torch.int32, device=device)


class CudaTimer:
    """Device-side timing via CUDA events. Returns milliseconds."""

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *a):
        self.end.record()

    def ms(self) -> float:
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)


def percentiles(xs_ms: list[float]) -> dict:
    xs = sorted(xs_ms)
    n = len(xs)

    def p(q):
        return xs[min(n - 1, int(q * n))]

    return {
        "n": n,
        "min_us": xs[0] * 1e3,
        "p50_us": p(0.50) * 1e3,
        "p90_us": p(0.90) * 1e3,
        "p99_us": p(0.99) * 1e3,
        "max_us": xs[-1] * 1e3,
    }


if __name__ == "__main__":
    _assert_dims()
    print("dims OK:")
    print(f"  uplink   = {UP_BYTES_PER_TOKEN} B/token "
          f"(q_fp8 {BYTES_INDEX_Q_FP8} + weights {BYTES_INDEX_WEIGHTS} + k {BYTES_INDEX_K})")
    print(f"  downlink = {DOWN_BYTES_PER_TOKEN} B/token (topk {INDEX_TOPK} x int32)")
    print(f"  layers   = {NUM_INDEXER_LAYERS}")
