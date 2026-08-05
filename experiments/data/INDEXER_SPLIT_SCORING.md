# Indexer 分离:打分复用 + 完整输入清单(带行号证据)

> 结论来自逐行读 `vllm/model_executor/layers/sparse_attn_indexer.py` 真实函数体
> 和 `vllm/v1/attention/backends/mla/indexer.py` 的 metadata builder,不是记忆或推测。
> 日期:2026-08-04。

---

## 1. 核心结论:打分是两个可直接重用的 vLLM op,零 kernel 拆分

融合 kernel(`fused_qkv_a_proj` / `fused_norm_rope` / `fused_q`)干的是
**投影 + norm + rope + 量化 + 写 cache** —— 它们**生产** index_q / index_k。
打分**不在**这些融合 kernel 里。

打分在 `sparse_attn_indexer` 函数内部,是两个界限分明、独立的 op:

| 步骤 | decode 路径的 op | prefill 路径的 op |
|------|------------------|-------------------|
| ① 算 logits(MQA 点积 + 加权) | `fp8_fp4_paged_mqa_logits`(:582) | `fp8_fp4_mqa_logits`(:479) |
| ② 选 top-2048 | `cooperative_topk` / `persistent_topk`(:613/626) | `top_k_per_row_prefill`(:488) |

**验证过的事实(远端可直接用):**

- 两个 logits op 都在 `vllm/utils/deep_gemm.py`(`fp8_fp4_mqa_logits:499`、
  `fp8_fp4_paged_mqa_logits:565`),是 DeepGEMM 的薄封装,`import` 即可调。
- topk op 是 `torch.ops._C.cooperative_topk` / `persistent_topk`,编译进 vLLM C 扩展,
  `import vllm` 后即在。
- **三个 op 都是无状态纯函数**:输出只由入参决定,无隐藏的模块级/实例级状态。
- 所以远端 `import vllm` + 调这两个 op,给同样的输入就能得到正确的 topk,
  **不需要拆任何融合 kernel**,也**不依赖主 MLA 路径**。

**推论:** exp09 里那个 pytorch 参照打分(`_score_and_topk`,einsum + torch.topk)
**不是**远端真正要跑的东西,远端跑的是上面这两个 vLLM op。所以拿 exp10 dump 去做
"pytorch 对拍"是无用功 —— 同一段 kernel,没有正确性风险,不需要对拍。

---

## 2. 完整输入清单(decode 稳态路,每项带行号)

decode 走 `fp8_fp4_paged_mqa_logits`(:582)+ `persistent_topk`(:626)。
把两个 op 的入参全部展开,逐项标注来源:

| # | 参数 | 形状/类型 | 来源(行号) | 归类 |
|---|------|-----------|-------------|------|
| 1 | `q_quant`(index Q) | [B,next_n,64,128] fp8 | 融合 kernel 产物,入参 | **每步传** ✓已验证 |
| 2 | `weights` | [B·next_n,64] fp32 | 入参 | **每步传** ✓已验证 |
| 3 | `k`(当前 index K) | [num_tokens,128] fp8 | 入参(:322/383) | **每步传**(仅当前 token) |
| 4 | `kv_cache`(历史 index K) | [num_blocks,block_size,1,132] uint8 | `self.k_cache.kv_cache`(:399/512) | **远端存**(核心状态) |
| 5 | `slot_mapping` | [num_tokens] int32 | metadata(:364) | 派生(当前 token 写入位置) |
| 6 | `block_table` | [B,max_blocks] int32 | metadata(:587/966) | **远端建/存**(分页映射) |
| 7 | `seq_lens`(context_lens) | [B,next_n] int32 | metadata(:586/967) | **每步传**(每步 +1) |
| 8 | `schedule_metadata` | 派生 tensor | `get_paged_mqa_logits_metadata`(:959) | **远端派生**(从 #7) |
| 9 | `max_model_len` | int | 静态配置(:589) | 一次性同步 |
| 10 | `topk_tokens` | int = 2048 | 静态配置(:593) | 一次性同步 |
| 11 | `topk_indices_buffer` | [B,2048] int32 | 输出 buffer(:593) | **远端自分配** |
| 12 | `max_seq_len` | int | metadata(:619) | **每步传** 或远端从 #7 求 max |
| 13 | `quant_block_size`,`scale_fmt` | int / str | 静态(:401/402) | 一次性(仅写 cache 用) |

---

## 3. 归类总结("到底需要多少东西")

**A. 每步必传(随 token 变,量小)—— 4 项**
- `q_quant`、`weights`、`k`(当前)、`seq_lens`
- 对应方案 A 的 **8452 B/token 上行**:q_fp8(8192)+ weights(128)+ k+scale(132)
- #1/#2/#3 已在 exp10 验证;#7 seq_lens 是新增的小量(B 个 int32)

**B. 远端持有的状态(不每步传)—— 2 项 ← 全部工程复杂度在这**
- `kv_cache`(历史 index K,#4):最大一块,靠 exp05 的 BULK 灌充 +
  每步用 `ops.indexer_k_quant_and_cache`(:397)把当前 k 追加进去
- `block_table`(#6):分页映射,**这就是"K cache 一致性"的具象** ——
  远端要维护它、且和主实例的 block 分配/evict 同步

**C. 远端可自行派生,不必传 —— 2 项**
- `schedule_metadata`(#8,从 seq_lens 算)
- `max_seq_len`(#12,= seq_lens.max())

**D. 启动时一次性同步(常量)—— 4 项**
- `max_model_len`、`topk_tokens`、`quant_block_size`、`scale_fmt`

**E. 远端自己分配 —— 1 项**
- `topk_indices_buffer`(输出承接)

**prefill 路**额外要 `cu_seqlen_ks` / `cu_seqlen_ke`(:429/430),同样从 seq_lens 派生,不传。

---

## 4. 一句话总结

**每步真正要传的只有 4 样(q / weights / k / seq_lens),其中 3 样已验证。**
其余要么是远端持有的状态(kv_cache + block_table),要么是常量或可派生量。

工程复杂度**全部压在 B 的两项**(kv_cache + block_table)——也就是
"这些放哪、怎么检索、怎么和主实例保持一致"。**打分本身零风险、零拆分。**

这与既定方向吻合:**KVConnector 控制面管生命周期,GIN 搬字节。**

---

## 5. 下一步(未做)

读 vLLM 的 KV cache 分配器 + block_table 管理(`gpu_model_runner` + `block_pool`),
设计远端的 paged index-K cache 放在哪、block_table 怎么和主实例同步/检索。
**这是真正的硬骨头。**

---

## 6. 进程模型 / 架构决策(2026-08-04)

### PD / attention-FFN 分离在 vLLM 里怎么实现(读代码坐实)

统一范式:**独立进程 + 外部 proxy 协调 + 连接器传数据面**。

- **PD 分离**:prefiller 和 decoder 是两个独立的 `vllm serve` 进程
  (`--kv-transfer-config` 里 `kv_role=kv_producer` vs `kv_consumer`),各占一张卡、各自端口。
  上面一个 HTTP proxy 转发请求(控制面:先发 prefiller 且把 `max_tokens` 改 1 只算 KV,
  再转 decoder 生成)。几 GB 的 KV 由两进程间经 KVConnector 走 RDMA **直传,不经过 proxy**。
- **两条路彻底分开**:proxy 只发号施令,连接器搬货。
- **attention/FFN(MoE EP)分离**:同一骨架,换个刀口(按层类型切而非按阶段),
  连接器传的是 hidden_states。

### 为什么 indexer 远端不能照抄 PD 的"远端 = 完整 vllm serve"

PD 拆的是整段计算,远端可以是完整引擎。indexer 远端只需要**两个无状态 op**
(`fp8_fp4_paged_mqa_logits` + `persistent_topk`)+ 一份 index-K cache + block_table。
让它起完整 `vllm serve` 会拉起一整套用不到的引擎(权重、采样、API server)。

### 决策(三部分)

1. **主实例侧 = 必须集成,无从选择。** `sparse_attn_indexer` 嵌在每层 attention 内部,
   无法搬出进程,只能在原地把计算外包(就是 exp10 hook 那个位置)。**这不是"A/B 选项",是强制的。**
2. **远端 = 一个轻量专用 indexer 进程,不是 vllm serve。** 它 `import vllm` 只为调那两个 op
   + 复用 vLLM 的 block_pool/kv-cache 数据结构;GIN 传输;KVConnector 控制面同步 block_table。
3. **部署 = 先 1 远端 : 1 主实例**(用户决定 2026-08-04)。池化(1:N,对应 exp07 多 QP)是下一步演进。
   1:1 让 block_table 同步最简单(远端只镜像一个主实例的调度状态)。

### 粒度:PD 给的 vs indexer 要的

- **PD 在请求级协调**(整个请求去 prefiller 还是 decoder,proxy 在 HTTP 层决定)。
- **indexer 在每层每步级协调**(decode 主循环里,每层同步等一次 GIN RTT,在 CUDA 热路径内)。
- 所以:借 PD 的"数据面直传 + 控制面同步"骨架,但**不要**它的 HTTP proxy
  (indexer 没有请求路由问题),也**不要**完整 serve 的远端。

### 钩子粒度错配 —— 已验证 + 修法

上一轮"KVConnector 控制面钩子直接对应 block_table 生命周期"的说法**部分错**,在 scheduler.py 验证:

| 路径 | 代码 | 调 connector 钩子吗 |
|------|------|------|
| running 请求(= decode 每步) | `scheduler.py:564` `allocate_slots(request, num_new_tokens, ...)` | **不调** |
| new 请求(prefill/首次) | `scheduler.py:946` `allocate_slots(...)` | 调 `update_state_after_alloc`(:974) |

`update_state_after_alloc` 的 docstring(`base.py:495`)明说是给"异步加载的 block"用的
——即 prefix 命中 / PD 传输那种 BULK 语义,**不是** decode 每步 +1 block 的增量。

**修法(代码自己指出的):** `scheduler.py:289`
`connector.bind_gpu_block_pool(kv_cache_manager.block_pool)`(`base.py:443`)——
connector 能拿到 **block_pool 直接引用**(simple_cpu_offload_connector 就这么用)。
block_pool 是所有 `allocate_slots` 落地的真相源。

**两层控制面借:**
- 粗粒度生命周期(新请求、prefix 命中、请求结束)→ 事件钩子
  (`update_state_after_alloc` / `request_finished`),够用。
- decode 每步 block 增量 → `bind_gpu_block_pool` 引用 + 每步从
  `common_attn_metadata.block_table_tensor` 读增量(worker 侧落地在
  `gpu_model_runner.py:1467` `block_table.append_row(new_block_ids, req_index)`,
  new_block_ids 来自调度器 = block_table 真相源)。

### block_table:远端自建 allocator,不镜像(已验证 2026-08-04)

**决定:远端跑自己的 allocator 复现逻辑结构,不镜像主实例的物理 block id。**

证据:
- **物理 id 不可复现**(镜像需流式同步每次分配决策):`BlockPool.get_new_blocks` =
  `free_block_queue.popleft_n`(block_pool.py:661);释放的块按 LRU/eviction 顺序回队
  (block_pool.py:719-742),请求的块倒序释放(single_type_kv_cache_manager.py:527)。
  任何 churn 后,下一个 id 取决于 free/evict 历史,而非请求状态。
- **物理 id 奇偶性不是正确性要求,逻辑位置奇偶性才是。** 决定性证据:indexer K-cache
  **在单进程内今天就已是独立的 KV-cache group**。`DeepseekV32IndexerCache.get_kv_cache_spec`
  返回自己的 `MLAAttentionSpec(block_size=cache_config.block_size, num_kv_heads=1,
  head_size=head_dim)`(deepseek_v2.py:631-637);coordinator 按 group 独立分配块列表
  (kv_cache_coordinator.py:108-113);indexer 用自己的 `block_table_tensor` 建 metadata
  (indexer.py:784)。所以 indexer 的物理 id 本来就和主 MLA group 不同——物理奇偶性可证不必要。
- worker 侧 block_table 是纯镜像(block_table.py:114-130,worker 零分配);
  slot 地址 = `block_number*block_size+offset`(block_table.py:407)——物理 id 只需和
  "那份 cache 把 K 存哪了"自洽。
- prefix-cache 共享(refcount,block_pool.py:702-717)只是省显存;远端用私有非共享 cache,
  只要逻辑位置 P 存 token P 的 index-K 就正确。

**远端 allocator 极简**:index-K group 是全注意力 MLA(无滑窗),block i = tokens
[i*bs, (i+1)*bs),append-only,block_size 同主实例。

**远端确实需要的跨实例耦合(全是逻辑元数据,无物理 id):**
(a) 请求生命周期 + 每请求 token 数 + **prefix 命中长度**(用于定块大小和 seq_lens——
这是唯一必须跟踪的,它是 token 数不是 block id);(b) index-K 内容流(或 hidden states)走 GIN;
(c) free/preempt 信号,按 request_id + 逻辑范围。

**结论**:1:1 远端是个轻量进程,自带极简 append-only allocator,只靠逻辑元数据 + K 字节流
与主实例耦合。不镜像 block_pool。这道硬骨头已解。

---

## 附:关联文件

- payload 布局契约:`../exp09_replay_demo/common.py`(8452B 上行 / 8192B 下行、对称堆 offset)
- 远端 pytorch 参照(仅数学正确性,非生产实现):`../exp09_replay_demo/bench_replay.py:_score_and_topk`
- 时延/资源/部署决策依据:`PRODUCTION_DESIGN_INPUTS.md`(同目录)
- 真实 payload 采集机制:`../exp10_vllm_shadow/dump_hook.py`
