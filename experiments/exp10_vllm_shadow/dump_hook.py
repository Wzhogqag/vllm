"""exp10 dump hook —— 轻量,延迟 patch 版。

.pth 文件在 site-packages 加载阶段调 install_finder(),此时 torch 都可能没就绪,
所以 install_finder() 只装一个 MetaPathFinder,不做任何 import。
真正的 patch 发生在 attention module 被 import 完成后。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_DUMPED = 0
_ROOT = None
_MAX_FULL_DUMPS = int(os.environ.get("VLLM_DUMP_FULL_CALLS", "5"))
_LOG_EVERY = 200
_PATCHED = False


def _dump_dir():
    global _ROOT
    if _ROOT is None:
        root = os.environ.get("VLLM_DUMP_DIR", "/tmp/idx_dump")
        _ROOT = Path(root)
        _ROOT.mkdir(exist_ok=True, parents=True)
    return _ROOT


def _tensor_sig(x):
    # 延迟 import torch —— 只在被调用时 import
    try:
        import torch
        if not isinstance(x, torch.Tensor):
            return None
        return {"shape": tuple(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
    except Exception:
        return None


def _make_wrapper(orig):
    def wrapper(
        hidden_states, k_cache_prefix, kv_cache, q_quant, q_scale, k, weights,
        quant_block_size, scale_fmt, topk_tokens, head_dim, max_model_len,
        total_seq_lens, topk_indices_buffer, skip_k_cache_insert, use_pcp,
        use_fp4_cache=False, dcp_rank=0, dcp_world_size=1,
        cp_kv_cache_interleave_size=1, skip_topk_buffer_clear=False,
    ):
        global _DUMPED
        call_id = _DUMPED
        _DUMPED += 1
        # DEBUG: 每次进入都强制打一个 print,证明 wrapper 被调用
        if call_id < 3:
            print(f"[WRAPPER-ENTER pid={os.getpid()}] call {call_id}", flush=True)

        try:
            layer_name = str(k_cache_prefix)
        except Exception:
            layer_name = "unknown"

        import torch
        def _sig(x):
            if not isinstance(x, torch.Tensor):
                return None
            return {"shape": tuple(x.shape), "dtype": str(x.dtype), "device": str(x.device)}

        sig = {
            "call_id": call_id,
            "pid": os.getpid(),
            "layer_name": layer_name,
            "hidden_states": _sig(hidden_states),
            "q_quant": _sig(q_quant),
            "q_scale": _sig(q_scale),
            "k": _sig(k),
            "weights": _sig(weights),
            "topk_indices_buffer": _sig(topk_indices_buffer),
            "skip_k_cache_insert": bool(skip_k_cache_insert),
            "topk_tokens": int(topk_tokens),
            "head_dim": int(head_dim),
            "quant_block_size": int(quant_block_size),
            "scale_fmt": scale_fmt,
            "ts": time.time(),
        }

        out = orig(
            hidden_states, k_cache_prefix, kv_cache, q_quant, q_scale, k, weights,
            quant_block_size, scale_fmt, topk_tokens, head_dim, max_model_len,
            total_seq_lens, topk_indices_buffer, skip_k_cache_insert, use_pcp,
            use_fp4_cache, dcp_rank, dcp_world_size, cp_kv_cache_interleave_size,
            skip_topk_buffer_clear,
        )

        if call_id < _MAX_FULL_DUMPS:
            outdir = _dump_dir()
            fpath = outdir / f"call_pid{os.getpid()}_{call_id:04d}.pt"
            payload = {
                "sig": sig,
                "hidden_states": hidden_states.detach().cpu() if isinstance(hidden_states, torch.Tensor) else None,
                "q_quant": q_quant.detach().cpu() if isinstance(q_quant, torch.Tensor) else None,
                "q_scale": q_scale.detach().cpu() if isinstance(q_scale, torch.Tensor) else None,
                "k": k.detach().cpu() if isinstance(k, torch.Tensor) else None,
                "weights": weights.detach().cpu() if isinstance(weights, torch.Tensor) else None,
                "topk_indices": (
                    topk_indices_buffer[: hidden_states.shape[0]].detach().cpu()
                    if isinstance(topk_indices_buffer, torch.Tensor) else None
                ),
            }
            torch.save(payload, fpath)
            print(f"[dump_hook pid={os.getpid()}] call {call_id}  layer={layer_name}  → {fpath}", flush=True)

        if call_id % _LOG_EVERY == 0:
            print(f"[dump_hook pid={os.getpid()}] tick call {call_id}", flush=True)

        return out
    return wrapper


def _try_patch():
    global _PATCHED
    if _PATCHED:
        return True
    mod = sys.modules.get("vllm.model_executor.layers.sparse_attn_indexer")
    if mod is None or not hasattr(mod, "sparse_attn_indexer"):
        return False
    orig = mod.sparse_attn_indexer
    if getattr(orig, "_is_dump_wrapper", False):
        _PATCHED = True
        return True
    wrapper = _make_wrapper(orig)
    wrapper._is_dump_wrapper = True
    mod.sparse_attn_indexer = wrapper

    # 同步补丁已 from-import 过的 attention.py
    for name, m in list(sys.modules.items()):
        if m is None:
            continue
        if "deepseek_v32" in name and hasattr(m, "sparse_attn_indexer"):
            fn = getattr(m, "sparse_attn_indexer")
            if not getattr(fn, "_is_dump_wrapper", False):
                m.sparse_attn_indexer = wrapper
                print(f"[dump_hook pid={os.getpid()}] also patched {name}", flush=True)

    _PATCHED = True
    print(f"[dump_hook pid={os.getpid()}] patched. dump_dir={_dump_dir()}", flush=True)
    return True


class _Finder:
    """每次 import 时 poke 一下 _try_patch(),直到成功。零副作用。"""
    def find_spec(self, name, path, target=None):
        if not _PATCHED and "sparse_attn_indexer" in name or "deepseek_v32" in name:
            _try_patch()
        return None


def install_finder():
    """.pth 阶段调用这个 —— 只装 finder,不 import torch/vllm。"""
    for f in sys.meta_path:
        if isinstance(f, _Finder):
            return
    sys.meta_path.append(_Finder())


# 额外:装一个 sys.settrace-free 的兜底 —— 每次 import 完 poke 一下
_orig_import = __builtins__.__import__ if isinstance(__builtins__, dict) is False else __builtins__['__import__']

def _patched_import(name, *args, **kwargs):
    result = _orig_import(name, *args, **kwargs)
    if not _PATCHED and ("sparse_attn_indexer" in name or "deepseek_v32" in name):
        _try_patch()
    return result


def install():
    """完整 install:装 finder + 替换 __import__。多次调用幂等。"""
    install_finder()
    # 替换 builtins.__import__(仅当还没被替换过)
    import builtins
    if getattr(builtins.__import__, "_dump_hooked", False):
        return
    orig = builtins.__import__
    def hooked(name, *args, **kwargs):
        result = orig(name, *args, **kwargs)
        if not _PATCHED and ("sparse_attn_indexer" in name or "deepseek_v32" in name):
            _try_patch()
        return result
    hooked._dump_hooked = True
    builtins.__import__ = hooked
    print(f"[dump_hook pid={os.getpid()}] finder+import hook installed", flush=True)


if __name__ == "__main__":
    install()
