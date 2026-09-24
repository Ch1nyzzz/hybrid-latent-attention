"""How many latent dimensions do K / V / loop-1 need?  Training-free rank analysis.

One teacher pass over the Stage1 calibration blocks (same 128 x 2048 as teacher_init) gives the
per-layer covariances teacher_init uses. From them:
  * spectra: per-frequency (loops 2..T x heads) K eigenvalues, joint V eigenvalues, loop-1 K/V;
  * init sweep: for each (rank_k, rank_v, rank1) build the PCA-initialised S6 student exactly as
    teacher_init does (split into collect/fill so covariances are reused) and measure the Stage1
    validation metrics (per-loop attention KL and output MSE) on dev records.
Independent single-GPU shards: --shard i --shards n splits the config list.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from .corpus_index import RecordIndex
from .register import LatentStudent
from .teacher import Teacher
from .train_stage1_recipe import evaluate


@torch.no_grad()
def collect(teacher, blocks, device, H, D, nL, S):
    """teacher_init's covariance pass, verbatim (micro_batch 1)."""
    nf = D // 2
    covk1 = torch.zeros(nL, nf, H, H, device=device, dtype=torch.float64)
    covv1 = torch.zeros(nL, H * D, H * D, device=device, dtype=torch.float64)
    covkJ = torch.zeros(nL, nf, S * H, S * H, device=device, dtype=torch.float64)
    covvJ = torch.zeros(nL, S * H * D, S * H * D, device=device, dtype=torch.float64)
    for i in range(len(blocks)):
        ids = torch.from_numpy(np.asarray(blocks[i:i + 1]).astype(np.int64)).to(device)
        teacher.run(ids)
        for l in range(nL):
            attn = teacher.layers[l].self_attn
            comps, vs = [], []
            for t, h in enumerate(teacher.h_in[l]):
                k = attn.k_proj(h).float().reshape(-1, H, D)
                comp = torch.cat([k[:, :, :nf], k[:, :, nf:]], 0)
                v = attn.v_proj(h).float().reshape(-1, H * D)
                if t == 0:
                    covk1[l] += torch.einsum("nhf,ngf->fhg", comp, comp).double()
                    covv1[l] += (v.T @ v).double()
                else:
                    comps.append(comp); vs.append(v)
            comp = torch.cat(comps, 1); v = torch.cat(vs, 1)
            covkJ[l] += torch.einsum("nhf,ngf->fhg", comp, comp).double()
            covvJ[l] += (v.T @ v).double()
    return covk1, covv1, covkJ, covvJ


@torch.no_grad()
def init_from(student, teacher, covs, device):
    """teacher_init's fill / fill_joint, verbatim, from precomputed covariances."""
    covk1, covv1, covkJ, covvJ = covs
    cfg = student.cfg
    H, D, nL = cfg["heads"], cfg["head_dim"], cfg["num_layers"]
    nf = D // 2; rK, rV, r1 = cfg["rank"], cfg["rank_v"], cfg["rank1"]
    assert (rK // 2) % nf == 0 and rK // 2 // nf <= (cfg["loops"] - 1) * H, (rK, nf, H)
    assert rV <= (cfg["loops"] - 1) * H * D and r1 <= H * D
    assert (r1 // 2) % nf == 0 and r1 // 2 // nf <= H, (r1, nf, H)
    S = cfg["loops"] - 1

    def fill(covk_l, covv_l, Wk, Wv, rk, rv):
        mm = rk // 2 // nf
        cand = torch.zeros(rk + rv, Wk.shape[1], device=device); A = torch.zeros(H, D, rk, device=device)
        for f in range(nf):
            evals, evecs = torch.linalg.eigh(covk_l[f])
            M = evecs[:, -mm:].flip(-1).T.float()
            for s_ in range(mm):
                p_ = s_ * nf + f
                cand[p_] = torch.einsum("h,hd->d", M[s_], Wk.view(H, D, -1)[:, f])
                cand[rk // 2 + p_] = torch.einsum("h,hd->d", M[s_], Wk.view(H, D, -1)[:, nf + f])
                A[:, f, p_] = M[s_]; A[:, nf + f, rk // 2 + p_] = M[s_]
        evals, evecs = torch.linalg.eigh(covv_l)
        Pv = evecs[:, -rv:].flip(-1).T.float()
        cand[rk:] = Pv @ Wv
        return cand, A, Pv.view(rv, H, D).permute(1, 0, 2)

    def fill_joint(covk_l, covv_l, Wk, Wv, rk, rv):
        mm = rk // 2 // nf; T_ = cfg["loops"]
        cand = torch.zeros(T_, rk + rv, Wk.shape[1], device=device); A = torch.zeros(T_, H, D, rk, device=device)
        Wk3 = Wk.view(H, D, -1)
        for f in range(nf):
            evals, evecs = torch.linalg.eigh(covk_l[f])
            M = evecs[:, -mm:].flip(-1).T.float().view(mm, S, H)
            for s_ in range(mm):
                p_ = s_ * nf + f
                for s in range(S):
                    cand[s + 1, p_] = torch.einsum("h,hd->d", M[s_, s], Wk3[:, f]); cand[s + 1, rk // 2 + p_] = torch.einsum("h,hd->d", M[s_, s], Wk3[:, nf + f])
                    A[s + 1, :, f, p_] = M[s_, s]; A[s + 1, :, nf + f, rk // 2 + p_] = M[s_, s]
        evals, evecs = torch.linalg.eigh(covv_l)
        Pv = evecs[:, -rv:].flip(-1).T.float().view(rv, S, H * D)
        for s in range(S):
            cand[s + 1, rk:] = Pv[:, s] @ Wv
        Bm = Pv.permute(1, 0, 2).reshape(S, rv, H, D).permute(0, 2, 1, 3)
        A[0] = A[1]; Bm = torch.cat([Bm[:1], Bm], 0)
        return cand, A, Bm

    for l in range(nL):
        sl = student.layers[l]; attn = teacher.layers[l].self_attn
        Wk = attn.k_proj.weight.float(); Wv = attn.v_proj.weight.float()
        cand, A, Bm = fill_joint(covkJ[l], covvJ[l], Wk, Wv, rK, rV)
        for t in range(1, cfg["loops"]):
            sl.cand_s[t - 1].weight.copy_(cand[t])
        sl.q_absorb.copy_(A[1:]); sl.out_absorb.copy_(Bm[1:])
        if r1:
            cand1, A1, B1 = fill(covk1[l], covv1[l], Wk, Wv, r1, r1)
            sl.cand1.weight.copy_(cand1.to(sl.cand1.weight.dtype)); sl.q_absorb1.copy_(A1); sl.out_absorb1.copy_(B1)


def spectra(covs):
    covk1, covv1, covkJ, covvJ = covs
    ev = lambda c: torch.linalg.eigvalsh(c).flip(-1).float().cpu().numpy()
    return dict(k_joint=np.stack([ev(c) for c in covkJ]),   # [L, nf, S*H]
                v_joint=np.stack([ev(c) for c in covvJ]),   # [L, S*H*D]
                k_loop1=np.stack([ev(c) for c in covk1]),   # [L, nf, H]
                v_loop1=np.stack([ev(c) for c in covv1]))   # [L, H*D]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True); p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--configs', required=True, help='K:V:R1 comma list (all shards share it)')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--init-blocks', type=int, default=128); p.add_argument('--calibration-length', type=int, default=2048)
    p.add_argument('--eval-records', type=int, default=64); p.add_argument('--micro-batch', type=int, default=4)
    p.add_argument('--seed', type=int, default=20260915); p.add_argument('--spectra', action='store_true')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    device = torch.device('cuda')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(a.model_path, 4, device, dtype=torch.bfloat16)
    c = teacher.cfg
    H, D, nL = c.num_attention_heads, c.hidden_size // c.num_attention_heads, c.num_hidden_layers
    data = Path(a.data_dir)
    calibration = RecordIndex(data / 'calibration.jsonl')
    blocks = [calibration.sample_at(i, seed=a.seed, stage='calibration', min_length=a.calibration_length)['input_ids'][:a.calibration_length]
              for i in range(a.init_blocks)]
    calibration.close()
    t0 = time.monotonic()
    covs = collect(teacher, blocks, device, H, D, nL, 3)
    print(json.dumps(dict(event='covariances', seconds=time.monotonic() - t0, shard=a.shard)), flush=True)
    if a.spectra:
        np.savez_compressed(out / 'spectra.npz', **spectra(covs))
        print(json.dumps(dict(event='spectra_saved')), flush=True)
    dev = RecordIndex(data / 'dev.jsonl')
    validation = [dev.sample_at(i, seed=20260915, stage='evaluation', min_length=64) for i in range(a.eval_records)]
    dev.close()
    configs = [tuple(int(x) for x in s.split(':')) for s in a.configs.split(',')]
    for rk, rv, r1 in configs[a.shard::a.shards]:
        t = time.monotonic()
        student = LatentStudent(nL, c.hidden_size, H, D, 4, rk, rv, r1).to(device).eval()
        init_from(student, teacher, covs, device)
        r = evaluate(student, teacher, validation, a.micro_batch, device)
        row = dict(rank_k=rk, rank_v=rv, rank1=r1, kl_per_loop=r['kl_per_loop'], out_per_loop=r['out_per_loop'],
                   kl_per_layer=r['kl_per_layer'], out_per_layer=r['out_per_layer'], records=r['records'],
                   cache_elems_per_token_layer=rk + rv + 2 * r1, seconds=time.monotonic() - t)
        with (out / f'sweep-shard{a.shard}.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print('RANK_SWEEP ' + json.dumps({k: row[k] for k in ('rank_k', 'rank_v', 'rank1', 'kl_per_loop', 'out_per_loop', 'seconds')}), flush=True)
        del student; torch.cuda.empty_cache()
    print('RANK_SWEEP_SHARD_DONE', a.shard, flush=True)


if __name__ == '__main__':
    main()
