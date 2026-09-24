"""Decode-time KV sharing for Ouro in vLLM (paper Table 14, 'last-step only' scheme).

Loop T-1 (final) layers: standard `Attention`, full paged cache.
Loops 0..T-2 layers: `OuroPrefixAttention` -> `OuroPrefixSpec` (an R-SWA spec). The R-SWA manager keeps only
the prompt blocks plus the last `rswa_window` tokens' blocks for these layers, so their KV memory is
O(prompt + window) instead of O(sequence).

Decode token at absolute position n, loop r < T-1, attends over three disjoint key ranges:
  A1 = [0, min(B, s))   from cache r      (prompt region; B = prompt length rounded up to a block)
  B  = [B, s)           from cache T-1    (older generated tokens, final-loop K/V)
  A2 = [s, n]           from cache r      (s = n - recent; recent tokens + self, own-loop K/V)
and the three partial softmax states are merged with log-sum-exp. Prefill tokens use cache r only.
Requires enforce_eager, no chunked prefill, no prefix caching, FlashAttention backend (FA2 is enough).
"""
from dataclasses import dataclass
import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends import flash_attn as _fa
from vllm.v1.core.single_type_kv_cache_manager import RSWAManager
from vllm.v1.kv_cache_interface import KVCacheSpec, RSWASpec, get_kv_quant_mode
try:
    from vllm.v1.kv_cache_interface import KVCacheSpecRegistry
except ImportError:  # registry lives next to the managers in some builds
    from vllm.v1.core.single_type_kv_cache_manager import KVCacheSpecRegistry


class OuroPrefixSpec(RSWASpec):
    """Distinct spec type so mixed full + prefix layers take vLLM's hybrid-group path (not merged)."""

    @classmethod
    def merge(cls, specs):
        assert all(isinstance(s, OuroPrefixSpec) for s in specs), "OuroPrefixSpec group must be uniform"
        base = RSWASpec.merge(specs)
        return cls(**{f: getattr(base, f) for f in base.__dataclass_fields__})


KVCacheSpecRegistry.register(OuroPrefixSpec, RSWAManager, uniform_type_base_spec=OuroPrefixSpec)


class OuroPrefixAttention(Attention):
    def __init__(self, *args, rswa_window: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._rswa_window = rswa_window

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        return OuroPrefixSpec(block_size=vllm_config.cache_config.block_size, num_kv_heads=self.num_kv_heads,
                              head_size=self.head_size, head_size_v=self.head_size_v, dtype=self.kv_cache_torch_dtype,
                              kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype), rswa_window=self._rswa_window)


@dataclass
class _StepPlan:
    prefill_rows: torch.Tensor | None = None
    p_cu: torch.Tensor | None = None
    p_max_q: int = 0
    p_seqused: torch.Tensor | None = None
    p_max_k: int = 0
    p_reqs: torch.Tensor | None = None
    decode_rows: torch.Tensor | None = None
    d_reqs: torch.Tensor | None = None
    d_cu: torch.Tensor | None = None
    d_seqlen: torch.Tensor | None = None
    d_max_seq: int = 0
    a1_len: torch.Tensor | None = None
    a1_empty: torch.Tensor | None = None
    a1_max: int = 0
    b_len: torch.Tensor | None = None
    b_empty: torch.Tensor | None = None
    b_max: int = 0
    b_shift_blocks: torch.Tensor | None = None


def build_step_plan(md, block_size: int, recent: int, device) -> _StepPlan:
    """Classify requests of this step into prefill / decode and precompute per-range key lengths (one sync)."""
    qsl = md.query_start_loc.cpu()
    sl = md.seq_lens.cpu()
    nreq = min(qsl.shape[0] - 1, sl.shape[0])
    qlen = (qsl[1:nreq + 1] - qsl[:nreq]).long()
    sl = sl[:nreq].long()
    assert md.rswa_prefix_lens is not None, "hf_overrides must set rswa_window so prompt lengths reach attention metadata"
    pl = md.rswa_prefix_lens.cpu()[:nreq].long()
    plan = _StepPlan()
    is_dec = (qlen == 1) & (sl > pl)
    is_pre = (qlen > 0) & ~is_dec
    if bool(is_pre.any()):
        reqs = torch.nonzero(is_pre).flatten()
        if not bool((qlen[reqs] == sl[reqs]).all()):
            raise NotImplementedError("chunked prefill / prefix caching are not supported with share_decode_kv")
        rows = torch.cat([torch.arange(int(qsl[i]), int(qsl[i + 1])) for i in reqs.tolist()])
        cu = torch.zeros(len(reqs) + 1, dtype=torch.int32); cu[1:] = torch.cumsum(qlen[reqs], 0)
        plan.prefill_rows, plan.p_reqs = rows.to(device), reqs.to(device)
        plan.p_cu, plan.p_max_q = cu.to(device), int(qlen[reqs].max())
        plan.p_seqused, plan.p_max_k = sl[reqs].to(torch.int32).to(device), int(sl[reqs].max())
    if bool(is_dec.any()):
        reqs = torch.nonzero(is_dec).flatten()
        n = sl[reqs] - 1                      # absolute position of the current token
        B = ((pl[reqs] + block_size - 1) // block_size) * block_size
        s = torch.clamp(n - recent, min=0)
        a1 = torch.minimum(B, s)              # keys [0, a1) from cache r
        b = torch.clamp(s - B, min=0)         # keys [B, s) from cache T-1
        plan.decode_rows, plan.d_reqs = qsl[reqs].long().to(device), reqs.to(device)
        plan.d_cu = torch.arange(len(reqs) + 1, dtype=torch.int32, device=device)
        plan.d_seqlen, plan.d_max_seq = sl[reqs].to(torch.int32).to(device), int(sl[reqs].max())
        plan.a1_empty, plan.b_empty = (a1 == 0).to(device), (b == 0).to(device)
        plan.a1_len, plan.a1_max = torch.clamp(a1, min=1).to(torch.int32).to(device), int(max(int(a1.max()), 1))
        plan.b_len, plan.b_max = torch.clamp(b, min=1).to(torch.int32).to(device), int(max(int(b.max()), 1))
        plan.b_shift_blocks = (B // block_size).to(device)
    return plan


def _merge(outs, lses):
    """LSE merge of partial attention outputs. outs: [n,H,D]; lses: [H,n] (fp32, -inf = empty)."""
    lse = torch.stack([l.t() for l in lses])             # [k, n, H]
    m = lse.max(0).values
    w = torch.exp(lse - m).unsqueeze(-1)                  # [k, n, H, 1]
    num = sum(w[i] * outs[i].float() for i in range(len(outs)))
    return (num / w.sum(0)).to(outs[0].dtype)


def _fa_call(q, kc, vc, cu, max_q, seqused, max_k, scale, causal, block_table, window=None, lse=False):
    return _fa.flash_attn_varlen_func(q=q, k=kc, v=vc, cu_seqlens_q=cu, max_seqlen_q=max_q, seqused_k=seqused,
                                      max_seqlen_k=max_k, softmax_scale=scale, causal=causal, window_size=list(window) if window else None,
                                      block_table=block_table, fa_version=2, return_softmax_lse=lse)


def shared_decode_attention(q, k, v, layer_r: Attention, layer_T: Attention, num_heads, num_kv_heads, head_dim,
                            scale, block_size, recent):
    """Attention for loop r < T-1 with decode-time KV sharing. q/k/v: [N, heads*D]. Returns [N, heads*D]."""
    ctx = get_forward_context()
    md_all = ctx.attn_metadata
    if not isinstance(md_all, dict) or layer_r.layer_name not in md_all or md_all[layer_r.layer_name] is None \
            or layer_r.kv_cache.numel() == 0:
        return layer_r(q, k, v)  # profiling / dummy runs: no real metadata or cache yet
    md_r, md_T = md_all[layer_r.layer_name], md_all[layer_T.layer_name]
    q3 = q.view(-1, num_heads, head_dim); k3 = k.view(-1, num_kv_heads, head_dim); v3 = v.view(-1, num_kv_heads, head_dim)
    layer_r.impl.do_kv_cache_update(layer_r, k3, v3, layer_r.kv_cache, ctx.slot_mapping[layer_r.layer_name])
    kc_r, vc_r = layer_r.kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    kc_T, vc_T = layer_T.kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    plan = getattr(ctx, "_ouro_plan", None)
    if plan is None:
        plan = build_step_plan(md_r, block_size, recent, q.device)
        ctx._ouro_plan = plan
    out = torch.zeros_like(q3)
    if plan.prefill_rows is not None:
        o = _fa_call(q3[plan.prefill_rows], kc_r, vc_r, plan.p_cu, plan.p_max_q, plan.p_seqused, plan.p_max_k, scale,
                     True, md_r.block_table[plan.p_reqs])
        out[plan.prefill_rows] = o
    if plan.decode_rows is not None:
        qd = q3[plan.decode_rows]
        bt_r = md_r.block_table[plan.d_reqs]
        bt_T = md_T.block_table[plan.d_reqs]
        idx = torch.arange(bt_T.shape[1], device=q.device).unsqueeze(0) + plan.b_shift_blocks.unsqueeze(1)
        bt_T_shift = torch.gather(bt_T, 1, idx.clamp(max=bt_T.shape[1] - 1))
        o1, l1 = _fa_call(qd, kc_r, vc_r, plan.d_cu, 1, plan.a1_len, plan.a1_max, scale, False, bt_r, lse=True)
        o2, l2 = _fa_call(qd, kc_T, vc_T, plan.d_cu, 1, plan.b_len, plan.b_max, scale, False, bt_T_shift, lse=True)
        o3, l3 = _fa_call(qd, kc_r, vc_r, plan.d_cu, 1, plan.d_seqlen, plan.d_max_seq, scale, True, bt_r,
                          window=(recent, 0), lse=True)
        l1 = l1.masked_fill(plan.a1_empty.unsqueeze(0), float("-inf"))
        l2 = l2.masked_fill(plan.b_empty.unsqueeze(0), float("-inf"))
        out[plan.decode_rows] = _merge([o1, o2, o3], [l1, l2, l3])
    return out.view(-1, num_heads * head_dim)
