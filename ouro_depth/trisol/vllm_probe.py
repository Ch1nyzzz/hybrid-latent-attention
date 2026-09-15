"""Inspect the image's vLLM attention kernels for the latent cache: head-size limits per backend, MLA constraints,
and functional tests of FlashAttention / Triton unified attention at head sizes 256 and 512 on this GPU."""
import importlib, inspect, os, re, subprocess, sys, traceback
import torch, vllm
V = os.path.dirname(vllm.__file__)
print("VLLM", vllm.__version__, "GPU", torch.cuda.get_device_name(0), flush=True)

def grep(pat, path, ctx=0):
    p = os.path.join(V, path)
    if not os.path.exists(p): print(f"-- {path}: missing"); return
    out = subprocess.run(["grep", "-n", f"-A{ctx}", "-i", pat, p], capture_output=True, text=True).stdout
    print(f"-- {path} :: {pat}\n" + "\n".join(out.splitlines()[:40]), flush=True)

grep("def supports_head_size", "v1/attention/backends/flash_attn.py", 8)
grep("def supports_head_size|def get_supported_head_sizes", "v1/attention/backends/triton_attn.py", 8)
grep("head_size", "v1/attention/ops/triton_unified_attention.py", 0)
grep("def supports_head_size|def get_supported_head_sizes", "v1/attention/backends/flashinfer.py", 8)
grep("def is_fa_version_supported|def get_flash_attn_version", "v1/attention/backends/fa_utils.py", 12)
grep("kv_lora_rank|qk_rope_head_dim|head_size", "v1/attention/backends/mla/common.py", 0)
grep("assert|lora|head", "v1/attention/backends/mla/triton_mla.py", 0)
grep("class MLAAttention|def __init__|kv_lora_rank|qk_rope_head_dim|head_size", "model_executor/layers/attention/mla_attention.py", 0)
grep("MLAAttention(|kv_lora_rank=|qk_rope_head_dim=|v_head_dim=", "model_executor/models/deepseek_v2.py", 0)

from vllm.vllm_flash_attn import flash_attn_varlen_func
for hd in (256, 512):
    try:
        q = torch.randn(64, 16, hd, device="cuda", dtype=torch.bfloat16); k = torch.randn(64, 1, hd, device="cuda", dtype=torch.bfloat16); v = torch.randn(64, 1, hd, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
        o = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=64, max_seqlen_k=64, causal=True)
        print("FA varlen head", hd, "OK", tuple(o.shape), flush=True)
    except Exception as e:
        print("FA varlen head", hd, "FAIL", str(e)[:300], flush=True)
try:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend as TB
    for hd in (128, 256, 512, 1024):
        f = getattr(TB, "supports_head_size", None)
        print("Triton backend supports head", hd, ":", f(hd) if f else "n/a", flush=True)
    print("Triton backend supported list:", getattr(TB, "get_supported_head_sizes", lambda: "n/a")(), flush=True)
except Exception:
    traceback.print_exc()
try:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend as FB
    for hd in (256, 512): print("FA backend supports head", hd, ":", FB.supports_head_size(hd), flush=True)
except Exception:
    traceback.print_exc()
print("VLLM_PROBE_DONE", flush=True)
