"""HF reference for decode-time KV sharing in Ouro (paper Table 14 'last-step only' scheme).

Prefill: every loop keeps its own K/V for the prompt (slots ut*L + l).
Decode:  loops ut < T-1 attend over their own K/V for positions p < prefix_len or p >= n - recent
         (prompt region and the most recent tokens), the FINAL loop's K/V for prompt-boundary..n-recent
         (older generated tokens), and their own K/V for the current token. The final loop is unchanged.
prefix_len defaults to the prompt length; the vLLM implementation aligns it to the paged-cache block
boundary and keeps a short recent window per loop, so both knobs are exposed here for exact comparison.
Every loop still appends to its own slot (this reference has no memory constraint); only the attention
inputs are rewired.
"""
import torch
from .vendor.modeling_ouro import UniversalTransformerCache


class SharedDecodeCache(UniversalTransformerCache):
    def __init__(self, num_layers: int, total_ut_steps: int, prefix_len: int | None = None, recent: int = 0,
                 mode: str = "mixed"):
        """mode='mixed': prompt from own loop, older generated tokens from the final loop (block-aligned).
        mode='all_final': every previous token (prompt included) from the final loop; only self from own loop."""
        super().__init__(max_cache_size=num_layers * total_ut_steps)
        self.L, self.T, self.prefix_len, self.recent, self.mode = num_layers, total_ut_steps, prefix_len, recent, mode
        self._prompt_len = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        ut, l = divmod(layer_idx, self.L)
        prefill = self._prompt_len is None or key_states.shape[2] > 1
        k, v = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if prefill:
            self._prompt_len = k.shape[2]
            return k, v
        if ut == self.T - 1:
            return k, v
        n = k.shape[2] - 1  # position of the current token
        fslot = (self.T - 1) * self.L + l
        if self.mode == "all_final":
            fk, fv = self.key_cache[fslot], self.value_cache[fslot]  # positions [0, n)
            return torch.cat([fk, k[:, :, n:]], dim=2), torch.cat([fv, v[:, :, n:]], dim=2)
        B = self.prefix_len if self.prefix_len is not None else self._prompt_len
        s = max(n - self.recent, 0)
        if s <= B:
            return k, v  # nothing older than the recent window beyond the prompt boundary yet
        fk, fv = self.key_cache[fslot], self.value_cache[fslot]  # holds positions [0, n)
        k = torch.cat([k[:, :, :B], fk[:, :, B:s], k[:, :, s:]], dim=2)
        v = torch.cat([v[:, :, :B], fv[:, :, B:s], v[:, :, s:]], dim=2)
        return k, v
