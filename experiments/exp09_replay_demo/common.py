# SPDX-License-Identifier: Apache-2.0
# exp09 common — 维度常量、payload 布局、对称堆 offset

# --- V3.2 indexer 死维度(config.json 坐实) ---
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048
QUANT_BLOCK_SIZE = 128  # index_head_dim // 128 = 1 个 fp32 scale
NUM_HIDDEN_LAYERS = 61

# --- payload 布局(方案 A) ---
# 上行 per-token bytes:
#   index_q_fp8: 64 heads × 128 dim × 1 B  = 8192
#   index_weights: 64 heads × 2 B(bf16)     = 128
#   index_k + scale: 128 fp8 + 1 fp32       = 132
#   合计 = 8452 B(记忆里写 8580 是记 padding,严格按元素算是 8452)
INDEX_Q_FP8_PER_TOKEN = INDEX_N_HEADS * INDEX_HEAD_DIM         # 8192
INDEX_W_PER_TOKEN = INDEX_N_HEADS * 2                          # 128
INDEX_K_PER_TOKEN = INDEX_HEAD_DIM + 4                          # 132
UP_PER_TOKEN = INDEX_Q_FP8_PER_TOKEN + INDEX_W_PER_TOKEN + INDEX_K_PER_TOKEN  # 8452

# 下行 per-token bytes:
#   top-2048 int32 = 2048 × 4 = 8192
DOWN_PER_TOKEN = INDEX_TOPK * 4  # 8192


def up_bytes(batch: int) -> int:
    return batch * UP_PER_TOKEN


def down_bytes(batch: int) -> int:
    return batch * DOWN_PER_TOKEN


def payload_offsets(batch: int) -> dict:
    """对称堆内 payload 各段的 byte offset(rank0/rank1 同布局)。

    上行区 [0 .. UP_CAP):
      [0                                     ..) index_q_fp8   (B × 8192)
      [B*8192                                ..) index_weights (B × 128)
      [B*8192 + B*128                        ..) index_k+scale (B × 132)
    下行区 [UP_CAP .. UP_CAP+DOWN_CAP):
      [UP_CAP                                ..) topk indices  (B × 8192)
    """
    q_off = 0
    w_off = batch * INDEX_Q_FP8_PER_TOKEN
    k_off = w_off + batch * INDEX_W_PER_TOKEN
    return {
        "q_fp8": q_off,
        "weights": w_off,
        "index_k": k_off,
        "up_end": k_off + batch * INDEX_K_PER_TOKEN,
    }


# 对称堆总量(exp04 里 UP_CAP=DOWN_CAP=4MiB,足够 B=256)
UP_CAP = 4 * 1024 * 1024
DOWN_CAP = 4 * 1024 * 1024


def assert_batch_fits(batch: int) -> None:
    total_up = up_bytes(batch)
    total_down = down_bytes(batch)
    assert total_up <= UP_CAP, f"上行 {total_up} > UP_CAP {UP_CAP}"
    assert total_down <= DOWN_CAP, f"下行 {total_down} > DOWN_CAP {DOWN_CAP}"


if __name__ == "__main__":
    for b in (1, 4, 16, 64, 256):
        assert_batch_fits(b)
        offs = payload_offsets(b)
        print(f"B={b:3d}  up={up_bytes(b):>9d}  down={down_bytes(b):>9d}  offsets={offs}")
