"""exp10 dump hook —— exec_module 包装版。

原理:
  推理真正的调用点在 vllm/models/deepseek_v32/nvidia/attention.py 的 _fused_attention
  里,直接按名字调用 `sparse_attn_indexer(...)`。这个名字是通过
      from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
  from-import 拷贝进 attention 模块命名空间的。

  要拦截它,只需在 *源模块* sparse_attn_indexer.py 的模块体执行完之后、attention.py
  执行 from-import 之前,把源模块里的 sparse_attn_indexer 换成 wrapper —— 之后拷贝过去
  的自然就是 wrapper。import 依赖顺序保证源模块一定先于调用点 exec 完成,所以只要在
  exec_module 完成的瞬间打补丁,时序就是确定的,不再依赖 import 事件的偶然触发。

  兜底:同时也 patch 调用点模块(万一某进程里源模块已被缓存、拷贝已发生)。
"""

from __future__ import annotations

import importlib.abc
import os
import sys
import time
from pathlib import Path

_SRC_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"
_CALLSITE_MODULE = "vllm.models.deepseek_v32.nvidia.attention"
_ATTR = "sparse_attn_indexer"
_OP_CLASS = "SparseAttnIndexer"
_METHOD = "forward_cuda"

_DUMPED = 0          # 只统计“真实运行”(非 profiling)的 forward_cuda 调用
_PROFILING_SEEN = 0  # 统计被跳过的 profiling/dummy 调用(仅用于日志)
_ROOT = None
_WRAPPER = None
_MAX_FULL_DUMPS = int(os.environ.get("VLLM_DUMP_FULL_CALLS", "5"))
_MAX_TENSOR_MIB = int(os.environ.get("VLLM_DUMP_MAX_TENSOR_MIB", "256"))
# full-score 模式:额外保存 kv_cache(用到的 block 切片)+ block_table + seq_lens,
# 让远端 scorer 能重建 cache 并调打分 op 对拍 recall。此模式下不做 prefill row-cap
# (保留全部 query 行,topk 才能忠实比对)。配短 prompt 用,历史小、文件小。
_FULL_SCORE = os.environ.get("VLLM_DUMP_FULL_SCORE", "0") == "1"
# 只在这个 TP rank 落盘(减小产物:8 卡 → 1 卡)。设为 -1 则所有 rank 都存。
_DUMP_ONLY_RANK = int(os.environ.get("VLLM_DUMP_ONLY_RANK", "0"))
# 采样策略:
#   "first_n"    —— 前 N 个真实 call(旧行为,结构上只能取到 layer 0..N-1 的 prefill)
#   "per_layer"  —— 每 (layer_name, phase_bucket) 首次出现各存一份(默认)。
#                   一个长请求即可覆盖全部层 × {prefill, decode 前几步}。
_DUMP_STRATEGY = os.environ.get("VLLM_DUMP_STRATEGY", "per_layer")
# per_layer 模式下,decode 阶段每层最多存前几步(prefill 恒为 1 份)。
_DECODE_STEPS = int(os.environ.get("VLLM_DUMP_DECODE_STEPS", "2"))
# per_layer 模式的总文件数上限(防跑飞;61 层 × (1 prefill + N decode))。
_MAX_SAMPLES = int(os.environ.get("VLLM_DUMP_MAX_SAMPLES", "256"))
# prefill 行数上限:长 prompt 的 prefill 可能几千行,全存会让单文件几十 MB。
# 只保留最后 N 行 —— 它们候选池最大、对验证打分语义最有信息量。0=不限。
_PREFILL_MAX_ROWS = int(os.environ.get("VLLM_DUMP_PREFILL_MAX_ROWS", "64"))
_SEEN_KEYS = {}      # (layer_name, bucket) -> 已存次数
_SAMPLES = 0         # per_layer 模式已落盘总数
_LOG_EVERY = 200

# 已知位置 → 名字(best-effort,仅用于可读性;落盘不依赖它)。
_ARG_NAMES = {
    0: "hidden_states", 1: "k_cache_prefix", 2: "kv_cache", 3: "q_quant",
    4: "q_scale", 5: "k", 6: "weights", 7: "quant_block_size", 8: "scale_fmt",
    9: "topk_tokens", 10: "head_dim", 11: "max_model_len", 12: "total_seq_lens",
    13: "topk_indices_buffer", 14: "skip_k_cache_insert", 15: "use_pcp",
}


def _dump_dir():
    global _ROOT
    if _ROOT is None:
        _ROOT = Path(os.environ.get("VLLM_DUMP_DIR", "/tmp/idx_dump"))
        _ROOT.mkdir(exist_ok=True, parents=True)
    return _ROOT


def _sig_of(x):
    import torch
    if isinstance(x, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "mib": round(x.numel() * x.element_size() / 2**20, 3),
        }
    if x is None or isinstance(x, (int, float, bool, str)):
        return {"kind": "scalar", "value": x}
    return {"kind": "other", "type": type(x).__name__}


def _maybe_cpu(x):
    """小张量落盘,巨大张量(如 kv_cache)只留 sig 不落盘。"""
    import torch
    if not isinstance(x, torch.Tensor):
        return None
    if x.numel() * x.element_size() > _MAX_TENSOR_MIB * 2**20:
        return None
    return x.detach().cpu()


def _make_wrapper(orig):
    def wrapper(*args, **kwargs):
        global _DUMPED
        call_id = _DUMPED
        _DUMPED += 1
        if call_id < 3:
            print(
                f"[WRAPPER-ENTER pid={os.getpid()}] call {call_id} "
                f"nargs={len(args)} nkw={len(kwargs)}",
                flush=True,
            )

        do_full = call_id < _MAX_FULL_DUMPS

        # 输入张量在调 orig 之前快照(topk_indices_buffer 会被原地写)。
        pre_tensors = None
        if do_full:
            pre_tensors = {
                f"arg{i}": _maybe_cpu(a) for i, a in enumerate(args)
            }
            pre_tensors.update(
                {f"kw_{k}": _maybe_cpu(v) for k, v in kwargs.items()}
            )

        out = orig(*args, **kwargs)

        if do_full:
            layer_name = str(args[1]) if len(args) > 1 else "unknown"
            sig = {
                "call_id": call_id,
                "pid": os.getpid(),
                "layer_name": layer_name,
                "ts": time.time(),
                "arg_names_hint": _ARG_NAMES,
                "args": [_sig_of(a) for a in args],
                "kwargs": {k: _sig_of(v) for k, v in kwargs.items()},
            }
            payload = {
                "sig": sig,
                "inputs": {k: v for k, v in (pre_tensors or {}).items()
                           if v is not None},
                "output": _maybe_cpu(out),
            }
            fpath = _dump_dir() / f"call_pid{os.getpid()}_{call_id:04d}.pt"
            import torch
            torch.save(payload, fpath)
            print(
                f"[dump_hook pid={os.getpid()}] call {call_id} "
                f"layer={layer_name} → {fpath}",
                flush=True,
            )

        if call_id % _LOG_EVERY == 0:
            print(f"[dump_hook pid={os.getpid()}] tick call {call_id}", flush=True)

        return out

    wrapper._is_dump_wrapper = True
    return wrapper


def _is_real_run():
    """区分真实运行 vs profiling/dummy run。

    依据 vllm 源码 sparse_attn_indexer.py:319 的官方注释:
        # careful! this will be None in dummy run
        attn_metadata = get_forward_context().attn_metadata
    profiling(显存预估的 dummy forward)时 attn_metadata 不是 dict;真实 prefill/decode
    时是 dict。返回 (is_real, phase) —— phase ∈ {"profiling","real","no_ctx"}。
    """
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        return True, "no_fc_module"
    if not is_forward_context_available():
        # 没有 forward context = 不在正常前向里,保守当作非真实,跳过。
        return False, "no_ctx"
    md = get_forward_context().attn_metadata
    # 真实运行:dict[str, meta](常规)或 list[dict](ubatch/microbatch)。
    # profiling/dummy run:None(见 sparse_attn_indexer.py:319 官方注释)。
    if isinstance(md, dict):
        return True, "real"
    if isinstance(md, (list, tuple)) and md and all(
        isinstance(x, dict) for x in md
    ):
        return True, "real_ubatch"
    return False, "profiling"


def _tp_rank():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return -1


def _capture_score_metadata(self):
    """full-score 模式:从 forward_context 抠出本层的 kv_cache + block_table + seq_lens
    等打分 op 需要的状态,让远端 scorer 能忠实重建并对拍。

    返回一个 dict(全部搬到 CPU),取不到就返回 {}(失败安全,绝不影响主推理)。
    定位方式和真实函数一致:attn_metadata 按 self.k_cache.prefix 这个 key 取本层
    (sparse_attn_indexer.py:362 `attn_metadata[k_cache_prefix]`)。
    """
    out = {}
    try:
        import torch
        from vllm.forward_context import get_forward_context

        md_all = get_forward_context().attn_metadata
        if not isinstance(md_all, dict):
            return out

        # 本层 key:真实函数用 _resolve_layer_name(k_cache_prefix),而 prefix 就是
        # self.k_cache.prefix。metadata 的 key 可能是 prefix 本身,做个兜底匹配。
        prefix = getattr(getattr(self, "k_cache", None), "prefix", None)
        md = md_all.get(prefix)
        if md is None:
            for kk, vv in md_all.items():
                if prefix is not None and str(prefix) in str(kk):
                    md = vv
                    break
        if md is None:
            return out

        kv_cache = getattr(getattr(self, "k_cache", None), "kv_cache", None)
        out["max_model_len"] = int(getattr(self, "max_model_len", 0))
        out["max_total_seq_len"] = int(getattr(self, "max_total_seq_len", 0))
        out["topk_tokens"] = int(getattr(self, "topk_tokens", 0))
        out["quant_block_size"] = int(getattr(self, "quant_block_size", 0))
        out["scale_fmt"] = getattr(self, "scale_fmt", None)
        out["num_decodes"] = int(getattr(md, "num_decodes", 0))
        out["num_prefills"] = int(getattr(md, "num_prefills", 0))
        out["slot_mapping"] = _maybe_cpu(getattr(md, "slot_mapping", None))

        # decode 路:block_table + seq_lens 在 md.decode 上。
        dec = getattr(md, "decode", None)
        if dec is not None:
            bt = getattr(dec, "block_table", None)
            out["block_table"] = _maybe_cpu(bt)
            out["seq_lens"] = _maybe_cpu(getattr(dec, "seq_lens", None))
            # 只保存 block_table 引用到的那些物理块,cache 就很小。
            out["kv_cache_used"] = _slice_used_kv_cache(kv_cache, bt)
        else:
            # 纯 prefill:block_table 在 chunk 里,直接存整段 kv_cache 的有效前缀。
            out["block_table"] = None
            out["seq_lens"] = None
            out["kv_cache_used"] = _maybe_cpu(kv_cache) if kv_cache is not None else None

        out["kv_cache_shape"] = (
            tuple(kv_cache.shape) if kv_cache is not None else None
        )
    except Exception as e:
        out["_capture_error"] = repr(e)
    return out


def _slice_used_kv_cache(kv_cache, block_table):
    """只保留 block_table 引用到的物理块(去重),连同一个 old→new 块号重映射表。
    这样远端能用重映射后的 block_table 索引这份小 cache。"""
    try:
        import torch
        if kv_cache is None or block_table is None:
            return None
        used = torch.unique(block_table.reshape(-1))
        used = used[used >= 0]
        if used.numel() == 0:
            return None
        sub = kv_cache.index_select(0, used.to(kv_cache.device)).detach().cpu()
        return {
            "blocks": sub,                       # [num_used_blocks, block_size, 132]
            "orig_block_ids": used.detach().cpu(),  # 原物理块号
        }
    except Exception:
        return None



def _make_method_wrapper(orig_method):
    """Wrap SparseAttnIndexer.forward_cuda —— 这是 CustomOp 实例真正执行的方法。
    收到干净的 4 个张量:(hidden_states, q_quant, k, weights)。
    """
    def method_wrapper(self, hidden_states, q_quant, k, weights, *extra):
        global _DUMPED, _PROFILING_SEEN, _SAMPLES

        is_real, phase = _is_real_run()
        if not is_real:
            # profiling/dummy run —— 不计数、不 dump,直接放行。
            # 关键:这样真实 decode 进来时计数器仍从 0 开始,不会被启动期吃掉。
            _PROFILING_SEEN += 1
            if _PROFILING_SEEN <= 3:
                print(
                    f"[dump_hook pid={os.getpid()}] skip {phase} call "
                    f"(rows={hidden_states.shape[0]})",
                    flush=True,
                )
            return orig_method(self, hidden_states, q_quant, k, weights, *extra)

        call_id = _DUMPED
        _DUMPED += 1

        rank = _tp_rank()
        layer_name = str(getattr(getattr(self, "k_cache", None), "prefix", "?"))
        num_rows = int(hidden_states.shape[0])
        # prefill: 多 token 一次前向(rows>1);decode: 单 token 稳态(rows==1)。
        bucket = "prefill" if num_rows > 1 else "decode"

        if call_id < 3:
            print(
                f"[WRAPPER-ENTER pid={os.getpid()}] REAL forward_cuda call {call_id} "
                f"rows={num_rows} bucket={bucket} rank={rank}",
                flush=True,
            )

        # ---- 采样决策:是否落盘本 call ----
        rank_ok = _DUMP_ONLY_RANK < 0 or rank == _DUMP_ONLY_RANK or rank < 0
        want = False
        if rank_ok:
            if _DUMP_STRATEGY == "first_n":
                want = call_id < _MAX_FULL_DUMPS
            else:  # per_layer:每 (layer, bucket) 首次(decode 允许前 _DECODE_STEPS 步)
                key = (layer_name, bucket)
                seen = _SEEN_KEYS.get(key, 0)
                cap = 1 if bucket == "prefill" else _DECODE_STEPS
                want = seen < cap and _SAMPLES < _MAX_SAMPLES

        do_full = want
        # prefill 长序列:只保留最后 N 行(候选池最大、最有信息量)。decode rows=1 不受影响。
        # full-score 模式下不做 row-cap —— 要全部 query 行才能忠实对拍 topk。
        row_lo = 0
        if (
            not _FULL_SCORE
            and bucket == "prefill"
            and _PREFILL_MAX_ROWS > 0
            and num_rows > _PREFILL_MAX_ROWS
        ):
            row_lo = num_rows - _PREFILL_MAX_ROWS

        def _row_slice(t):
            import torch as _t
            if isinstance(t, _t.Tensor) and t.dim() >= 1 and t.shape[0] == num_rows:
                return t[row_lo:]
            return t

        pre = None
        if do_full:
            if isinstance(q_quant, tuple):
                q_snap = [_maybe_cpu(_row_slice(t)) for t in q_quant]
            else:
                q_snap = _maybe_cpu(_row_slice(q_quant))
            pre = {
                "hidden_states": _maybe_cpu(_row_slice(hidden_states)),
                "q_quant": q_snap,
                "k": _maybe_cpu(_row_slice(k)),
                "weights": _maybe_cpu(_row_slice(weights)),
            }

        out = orig_method(self, hidden_states, q_quant, k, weights, *extra)

        if do_full:
            key = (layer_name, bucket)
            step = _SEEN_KEYS.get(key, 0)
            _SEEN_KEYS[key] = step + 1
            _SAMPLES += 1
            sig = {
                "call_id": call_id,
                "pid": os.getpid(),
                "rank": rank,
                "phase": phase,
                "bucket": bucket,
                "step": step,
                "layer_name": layer_name,
                "ts": time.time(),
                "source": "forward_cuda",
                "num_rows": num_rows,
                "row_lo": row_lo,
                "row_hi": num_rows,
                "kept_rows": num_rows - row_lo,
                "hidden_states": _sig_of(hidden_states),
                "q_quant": (
                    [_sig_of(t) for t in q_quant]
                    if isinstance(q_quant, tuple) else _sig_of(q_quant)
                ),
                "k": _sig_of(k),
                "weights": _sig_of(weights),
                "topk_indices_buffer": _sig_of(
                    getattr(self, "topk_indices_buffer", None)
                ),
            }
            # output 是 topk_indices_buffer,按 max_model_len(8192)预分配、跨 call 复用;
            # 只有前 num_rows 行是本次真实写入,其余是陈旧 scratch。与 inputs 对齐,
            # 切到 [row_lo:num_rows] —— prefill 长序列只留最后 _PREFILL_MAX_ROWS 行。
            out_sliced = out
            try:
                import torch as _t
                if isinstance(out, _t.Tensor) and out.dim() >= 1 and out.shape[0] >= num_rows:
                    out_sliced = out[row_lo:num_rows]
            except Exception:
                pass
            payload = {"sig": sig, "inputs": pre, "output": _maybe_cpu(out_sliced)}
            # full-score 模式:附上 kv_cache(用到的块)+ block_table + seq_lens,
            # 让远端 scorer 能重建并对拍。在 orig_method 之后抓 —— 此刻 kv_cache 已含
            # 本次 k 的插入,正是打分 op 读的那份状态。失败安全。
            if _FULL_SCORE:
                payload["score"] = _capture_score_metadata(self)
            # 文件名编码 layer 号 + 阶段 + step,避免 per_layer 模式下碰撞、便于分析端定位。
            import re as _re
            m = _re.search(r"layers\.(\d+)\.", layer_name)
            lidx = m.group(1).zfill(2) if m else "xx"
            tag = "FS_" if _FULL_SCORE else ""
            fpath = _dump_dir() / f"{tag}L{lidx}_{bucket}_s{step}_rank{rank}.pt"
            import torch
            torch.save(payload, fpath)
            print(
                f"[dump_hook pid={os.getpid()}] REAL dump {tag}L{lidx} {bucket} "
                f"step={step} rows={num_rows} rank={rank} "
                f"(#{_SAMPLES}) → {fpath.name}",
                flush=True,
            )

        if call_id % _LOG_EVERY == 0:
            print(f"[dump_hook pid={os.getpid()}] tick call {call_id}", flush=True)

        return out

    method_wrapper._is_dump_wrapper = True
    return method_wrapper


def _patch_module(module):
    """把 module 里的 sparse_attn_indexer 换成 wrapper(幂等)。
    在源模块里额外 patch SparseAttnIndexer.forward_cuda —— 这才是 CustomOp 实例
    真正执行的入口(self.indexer_op(...) → forward_cuda → torch.ops)。必须在任何
    实例构造之前 patch,因为 CustomOp.__init__ 会把 forward_cuda 存进 _forward_method。
    """
    global _WRAPPER
    fn = getattr(module, _ATTR, None)
    if fn is not None and not getattr(fn, "_is_dump_wrapper", False):
        if _WRAPPER is None:
            _WRAPPER = _make_wrapper(fn)
        setattr(module, _ATTR, _WRAPPER)
        tag = "src" if module.__name__ == _SRC_MODULE else "callsite"
        print(
            f"[dump_hook pid={os.getpid()}] patched {tag} {module.__name__}.{_ATTR} "
            f"→ dump_dir={_dump_dir()}",
            flush=True,
        )

    if module.__name__ == _SRC_MODULE:
        cls = getattr(module, _OP_CLASS, None)
        meth = getattr(cls, _METHOD, None) if cls is not None else None
        if meth is not None and not getattr(meth, "_is_dump_wrapper", False):
            setattr(cls, _METHOD, _make_method_wrapper(meth))
            print(
                f"[dump_hook pid={os.getpid()}] patched method "
                f"{_OP_CLASS}.{_METHOD} → dump_dir={_dump_dir()}",
                flush=True,
            )


class _WrapLoader(importlib.abc.Loader):
    """包装原 loader:exec_module 跑完(模块体已执行、算子已注册)后立即打补丁。"""

    def __init__(self, orig_loader):
        self._orig = orig_loader

    def create_module(self, spec):
        return self._orig.create_module(spec)

    def exec_module(self, module):
        self._orig.exec_module(module)
        if module.__name__ in (_SRC_MODULE, _CALLSITE_MODULE):
            _patch_module(module)


class _Finder(importlib.abc.MetaPathFinder):
    """仅对两个目标模块接管 loader,其余模块零干预。"""

    def find_spec(self, name, path, target=None):
        if name not in (_SRC_MODULE, _CALLSITE_MODULE):
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _WrapLoader(spec.loader)
                return spec
        return None


def install():
    """装 finder;若目标模块已在 sys.modules 里则立即补丁。多次调用幂等。"""
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
        print(f"[dump_hook pid={os.getpid()}] exec_module finder installed", flush=True)
    for name in (_SRC_MODULE, _CALLSITE_MODULE):
        mod = sys.modules.get(name)
        if mod is not None:
            _patch_module(mod)


if __name__ == "__main__":
    install()
