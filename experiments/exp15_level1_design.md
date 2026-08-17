# exp15 — Level 1 设计:Live vLLM 集成 remote indexer

> 目标:主实例跑真 vLLM,forward 每层 indexer 当场把 payload 发远端算、阻塞等回、
> 用返回的 topk 继续。区别于 exp12(文件回放):新增变量是**引擎集成**。
> exp13/14(prefill 对拍 + op 时延)由用户并行做。

## 已验证的关键事实(两个 agent + 我亲自核对)

**1. 拦截缝:必须改 `sparse_attn_indexer.py` 或模型 forward,没有 drop-in hook。**
- KVConnector 的 worker hook(`wait_for_layer_load` 等)是最接近的可复用模板,但它挂在
  **MLA attention** 上(`attention.py:817` 的 decorator),在 indexer **之后**触发、且是**另一个 op**。
  indexer 是独立的、未被 decorator 包裹的 custom op(`torch.ops.vllm.sparse_attn_indexer`)。
- 所以 Level 1 **必然要动** `sparse_attn_indexer` 的 body 或调用点。
- 产出契约:结果必须**原地写入** `topk_indices_buffer[:num_tokens,:topk]`(int32、request-local
  索引、-1 padding)。返回新 tensor 会破坏 cudagraph replay。且**必须保留 K-cache 插入**
  (`:386-403`),否则 sparse attn 读到陈旧 KV。

**2. CUDA graph:V3.2 默认 enforce-eager 路线(修正了 agent 的乐观说法)。**
- `@eager_break_during_capture` 确实装在 registered op 上(`:295`),理论上阻塞 GIN 调用可住在
  eager-break 段里。**但** breakable-cudagraph 自动启用只对 `DeepseekV4ForCausalLM`,
  **V3.2 是 `DeepseekV32ForCausalLM`,不在名单,默认 OFF**。
- 第一版决策:**用 `--enforce-eager`**(和 exp10/12 一致,零 cudagraph 复杂度)。
  breakable-cudagraph(手动开 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`)留作性能阶段,且需先验证
  V3.2 能在 breakable 模式跑通(不在自动名单 = 可能没测过)。

**3. TP:indexer 全复制,8 rank 算出**相同**topk。**
- `wq_b=ReplicatedLinear`,`wk_weights_proj disable_tp=True`,`n_head=64` 不除 TP,op 内无 TP 规约。
- 意味着:8 worker 各自发 = 8x 冗余流量;但结果相同 → **rank0 算 + TP broadcast** 合法且省 8x。
  第一版可先每 worker 各发(简单),优化留后。

**4. 配置注入:KVConnector 的 out-of-tree 注册可复用,不用 fork core。**
- `--kv-transfer-config` 的 `kv_connector_module_path`(`factory.py:105-114`)能加载站外自定义类。
- `ForwardContext.additional_kwargs`(`forward_context.py:188`)是每步自由 side-channel,
  indexer 已经在读 forward_context,可作为"接收远端 topk"的读侧通道。

## 设计决策(待和用户讨论定稿)

### 决策 A:改动落在哪
选项 A1(推荐):**在 `sparse_attn_indexer` body 开头加一个"remote 分支"** —— 若开启 remote 模式
(env/config),则:插 K-cache(保留 `:386-403`)→ 打包 q_quant/weights → GIN 送远端 → 阻塞等 →
把回传 topk 原地写入 `topk_indices_buffer` → return。否则走原逻辑。改动集中在一个函数、一个
`if remote:` 分支,主路径零影响。
选项 A2:改模型 forward 调用点(`nvidia/attention.py:482`)包一层。更散,不推荐。

### 决策 B:cudagraph
B1(推荐,第一版):`--enforce-eager`。B2(性能阶段):breakable-cudagraph。

### 决策 C:TP 流量 —— 定稿:rank0 单发 + TP broadcast(用户方案)
indexer 全复制,8 卡 topk 相同。所以 **rank0 单发 GIN 上行 + 单收下行,再
`get_tp_group().broadcast(topk, src=0)`(parallel_state.py:764)分给 8 卡**。跨机流量 1x
(而非每 worker 各发的 8x),卡间 broadcast 走 NVLink 近乎免费。
时序:8 卡对称执行 indexer op,`if tp_rank==0: 发GIN+等+收; topk = get_tp_group().broadcast(topk, src=0)`
—— 其余 7 卡直接到 broadcast 等 rank0,在 broadcast 处汇合,不死锁。
(get_tp_group 已可 import,op 内已用同款 get_dcp_group/get_pcp_group 模式。)

### 决策 D:传输 + 远端
复用 exp12 已验证的 GIN 传输(`librix_replay.so`)+ 远端 scorer(`remote_scorer.py`)。
远端 rank1 基本不变;主实例侧把 exp12 rank0 的"读文件发"换成"forward 里实时发"。

### 决策 E:远端 K-cache 生命周期(最大的新问题)
exp12 是单次回放,cache 是抓取时静态的。Level 1 是**活体多步 decode**:远端的 index-K cache
要随请求增长(每步追加新 k)、请求结束要回收。这就是之前定的"控制面"问题 ——
第一版可先**单请求、prefill 一次性灌 + decode 逐步追加**(不做多请求/evict),把端到端跑通;
多请求生命周期(KVConnector 控制面)留作 Level 1.5。

## 建议的 exp15 第一版范围(最小可跑的真集成)

**单请求、enforce-eager、每 worker 各发、单机先验(93 主 + 93 上另一进程当远端 or 91 远端):**
1. 改 `sparse_attn_indexer` 加 remote 分支(env `VLLM_INDEXER_REMOTE=1` 开启)。
2. 主实例侧:每层把 q_quant/weights 经 GIN 发远端,阻塞收 topk 写回 buffer。
3. 远端:复用 exp12 scorer,但 cache 随 decode 步追加。
4. 验证:开 remote 跑一个请求,输出的 token 和不开 remote(原生本地 indexer)**逐 token 一致**
   (这是比 recall 更强的端到端正确性 —— 整个模型输出不变才算真的对)。

## 风险 / 待讨论

- **改 vllm core**:AGENTS.md 对改 vllm 有规矩(domain guide、测试、eval)。`sparse_attn_indexer.py`
  的改动要作为"加分支不改原逻辑",且 remote 关闭时零影响 —— 这点要守住。
- **阻塞延迟串行**:downstream 同层立即消费(`mla.py:180→190`),每层阻塞等一次 GIN,61 层
  串起来的延迟就是 exp04 测的 RTT × 61。这决定 Level 1 是否实用 —— exp14 的 op 时延数据正好补这块。
- **远端 K-cache 追加的正确性**:decode 每步远端要把新 k 插进自己的 paged cache,slot 位置要和
  seq_len 对齐 —— 这块最容易错,要单独验。
