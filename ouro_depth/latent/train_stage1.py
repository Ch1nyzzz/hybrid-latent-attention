"""S6 stage-one attention distillation; exact diagonal, terminal latent history."""
import torch
from torch.nn import functional as F


def layer_losses(sl, teacher, layer, *, backward=False, weight=1.0, output_weight=1.0):
    h = teacher.h_in[layer]
    cos, sin = teacher.pos
    regs = sl.write(h)
    packed = sl.pack(regs[-1], sl.write1(h[0]), cos, sin)
    n = h[0].shape[1]
    diagonal = torch.eye(n, device=h[0].device, dtype=torch.bool)[None, None]
    bias = teacher.causal_bias(n, h[0].device)
    kls, outs, losses = [], [], []
    for loop in range(sl.loops):
        with torch.no_grad():
            qr, key, value, q = teacher.qkv(layer, h[loop], cos, sin)
            teacher_scores = (qr @ key.transpose(-1, -2)).float() * teacher.layers[layer].self_attn.scaling
            target_logp = F.log_softmax(teacher_scores + bias, -1)
            target = teacher.out[layer][loop].float()
        scores = torch.where(diagonal, teacher_scores, sl.scores(loop, q, packed, cos, sin).float())
        logp = F.log_softmax(scores + bias, -1)
        kl = (target_logp.exp() * (target_logp - logp)).sum(-1).mean()
        p = logp.exp().to(value.dtype)
        output = sl.read_out(loop, p.masked_fill(diagonal, 0), packed)
        output = output + p.diagonal(dim1=-2, dim2=-1)[..., None] * value
        output = teacher.o_proj(layer, output.transpose(1, 2).reshape_as(h[loop]))
        error = (output.float() - target).square().mean(dim=(1, 2))
        energy = target.square().mean(dim=(1, 2)).clamp_min(1e-8)
        mse = (error / energy).mean()
        losses.append((kl + output_weight * mse) * weight / sl.loops)
        kls.append(kl.detach()); outs.append(mse.detach())
    if backward:
        sum(losses).backward()
    return dict(kl=torch.stack(kls), out=torch.stack(outs), loss=sum(losses))


if __name__ == '__main__':
    from .train_stage1_recipe import main
    main()
