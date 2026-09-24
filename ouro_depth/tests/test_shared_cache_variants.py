import torch
from ouro_depth.shared_decode_cache import SharedDecodeCache

def test_cache():
    L = 24
    T = 4
    batch_size = 4
    num_heads = 16
    head_dim = 64
    prompt_len = 10
    gen_len = 5

    for mode in ["all_final", "mean"]:
        print(f"Testing mode={mode}...")
        cache = SharedDecodeCache(num_layers=L, total_ut_steps=T, mode=mode)
        
        # Prefill: simulate layer updates for all L layers and T loops
        for ut in range(T):
            for l in range(L):
                layer_idx = ut * L + l
                k = torch.randn(batch_size, num_heads, prompt_len, head_dim)
                v = torch.randn(batch_size, num_heads, prompt_len, head_dim)
                ret_k, ret_v = cache.update(k, v, layer_idx)
                assert ret_k.shape == (batch_size, num_heads, prompt_len, head_dim)

        # Decode steps
        for step in range(gen_len):
            current_pos = prompt_len + step
            for ut in range(T):
                for l in range(L):
                    layer_idx = ut * L + l
                    k = torch.randn(batch_size, num_heads, 1, head_dim)
                    v = torch.randn(batch_size, num_heads, 1, head_dim)
                    ret_k, ret_v = cache.update(k, v, layer_idx)
                    expected_len = current_pos + 1
                    assert ret_k.shape == (batch_size, num_heads, expected_len, head_dim), \
                        f"Expected shape {(batch_size, num_heads, expected_len, head_dim)}, got {ret_k.shape}"
            if mode == "mean":
                for l in range(L):
                    expected_mean_k = torch.stack([cache.key_cache[u * L + l] for u in range(T)]).mean(dim=0)
                    assert torch.allclose(cache._mean_key_cache[l], expected_mean_k, atol=1e-6), \
                        f"Mean key mismatch at layer {l}"
        print(f"Mode {mode} passed!")

if __name__ == "__main__":
    test_cache()
