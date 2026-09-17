"""Verify static full-KV cache writes and masking against native Ouro."""
import torch
from ouro_depth.latent.benchmark_inference import StaticOuroCache
from ouro_depth.tests.test_s6_engine import fixture


@torch.no_grad()
def test_static_ouro_cache_matches_growing_native_cache():
    model, _, _, ids = fixture()
    prefix = ids[:, :3]
    native = model(prefix, use_cache=True, logits_to_keep=1).past_key_values
    capacity = 12
    keys = [torch.nn.functional.pad(x, (0,0,0,capacity-3)) for x in native.key_cache]
    values = [torch.nn.functional.pad(x, (0,0,0,capacity-3)) for x in native.value_cache]
    static = StaticOuroCache(keys, values, 3)
    for pos in range(3, 8):
        token = ids[:, pos:pos+1]
        expected = model(token, past_key_values=native, use_cache=True, logits_to_keep=1)
        native = expected.past_key_values
        # Eager CPU attention uses additive masks, unlike CUDA SDPA's bool masks.
        mask = torch.zeros(1,1,1,capacity)
        mask[...,pos+1:] = float('-inf')
        actual = model(token, past_key_values=static, use_cache=True, logits_to_keep=1,
                       cache_position=torch.tensor([pos]),position_ids=torch.tensor([[pos]]),
                       attention_mask={'full_attention':mask})
        torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
        for a,b in zip(static.key_cache,native.key_cache):
            torch.testing.assert_close(a[:,:,:pos+1],b)
