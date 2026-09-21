"""Zero-pad an inference-only S6 export, preserving split-half RoPE pairs."""
from copy import deepcopy
import argparse
from pathlib import Path
import torch


def pad_export(checkpoint):
    if 'optimizer' in checkpoint or 'backbone' in checkpoint:
        raise ValueError('Expected a latent-only inference export')
    result=deepcopy(checkpoint);cfg=result['cfg']
    rk,rv=cfg['rank'],cfg['rank_v'];width=max(rk,rv)
    if rk%cfg['head_dim'] or width%cfg['head_dim']:
        raise ValueError('K widths must contain complete RoPE frequency cycles')
    ki=torch.cat((torch.arange(rk//2),torch.arange(rk//2)+width//2))
    for name,value in list(result['student'].items()):
        if '.cand_s.' in name:
            expanded=value.new_zeros(2*width,value.shape[1])
            expanded[ki]=value[:rk];expanded[width:width+rv]=value[rk:]
        elif name.endswith('.q_absorb'):
            expanded=value.new_zeros(*value.shape[:-1],width);expanded[...,ki]=value
        elif name.endswith('.out_absorb'):
            expanded=value.new_zeros(*value.shape[:-2],width,value.shape[-1]);expanded[...,:rv,:]=value
        else:continue
        result['student'][name]=expanded
    cfg.update(rank=width,rank_v=width)
    result['serving_padding']={'source_rank_k':rk,'source_rank_v':rv,'physical_rank':width,'inference_only':True}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    ck=pad_export(torch.load(a.source,map_location='cpu',weights_only=False))
    tmp=out.with_suffix('.tmp');torch.save(ck,tmp);tmp.rename(out)
    print(ck['serving_padding'],flush=True)

if __name__=='__main__':main()
