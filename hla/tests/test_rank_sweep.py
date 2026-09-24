def test_rank_sweep_init_matches_teacher_init():
    import numpy as np, torch
    from hla.vendor.configuration_ouro import OuroConfig
    from hla.vendor.modeling_ouro import OuroForCausalLM
    from hla.latent.register import LatentStudent
    from hla.latent.teacher import Teacher
    from hla.latent.init_teacher import teacher_init
    from hla.latent import rank_sweep
    torch.manual_seed(0)
    cfg = OuroConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=4, max_position_embeddings=128, total_ut_steps=4, use_cache=False,
                     pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg._attn_implementation = 'eager'
    model = OuroForCausalLM(cfg).eval().requires_grad_(False)
    teacher = Teacher.wrap(model, 4)
    H, D, nL = 4, 16, 2
    blocks = np.random.randint(3, 64, (3, 32))
    covs = rank_sweep.collect(teacher, blocks, torch.device('cpu'), H, D, nL, 3)
    for rk, rv, r1 in [(16, 16, 16), (32, 48, 16), (96, 192, 64)]:
        torch.manual_seed(1); a = LatentStudent(nL, 64, H, D, 4, rk, rv, r1)
        torch.manual_seed(1); b = LatentStudent(nL, 64, H, D, 4, rk, rv, r1)
        teacher_init(a, teacher, blocks, torch.device('cpu'), micro_batch=1)
        rank_sweep.init_from(b, teacher, covs, torch.device('cpu'))
        for (n, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
            assert torch.equal(x, y), (rk, rv, r1, n)
        print('identical', rk, rv, r1)
    s = rank_sweep.spectra(covs)
    print({k: v.shape for k, v in s.items()})
