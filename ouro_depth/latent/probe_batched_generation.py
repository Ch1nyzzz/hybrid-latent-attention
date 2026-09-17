"""GPU qualification of padded batched decode against independent serial engines."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .batched_engine import BatchedRollingEngine
from .generate import INSTR, LatentDecoder, BatchedLatentDecoder
from .register import LatentStudent
from .training_common import amp
from .vendor_model import load_teacher


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--student', required=True)
    p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    model = load_teacher(args.model, 4, device, torch.bfloat16)
    student = LatentStudent.from_checkpoint(torch.load(args.student, map_location='cpu', weights_only=True), device).eval()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(x) for x in Path(args.data).read_text().splitlines()][:32]
    texts = [tok.apply_chat_template([{'role':'user','content':r['problem']+INSTR}], tokenize=False, add_generation_prompt=True) for r in rows]
    prompts = [tok(t, return_tensors='pt', add_special_tokens=False).input_ids.to(device) for t in texts]
    filler = tok.encode('The quick brown fox jumps over the lazy dog. ', add_special_tokens=False)
    missing = 4096-prompts[3].shape[1]
    long_prompt = torch.cat((prompts[3].new_tensor([(filler*((missing+len(filler)-1)//len(filler)))[:missing]]),prompts[3]),1)
    check_prompts = prompts[:3]+[long_prompt]
    batch_decoder = BatchedLatentDecoder(model,student,10240,0)
    batch, actual = batch_decoder.prefill_batch(check_prompts)
    refs, expected = [], []
    with amp(device):
        for ids in check_prompts:
            ref = BatchedRollingEngine(model,student,False)
            pred,_ = ref.prefill(ids,last_logits_only=True)
            ref.detach_history();refs.append(ref);expected.append(pred[:,-1].float())
        observations = []
        for step in range(64):
            for i in range(4):
                if i==1 and step>=16:
                    continue
                lp=actual[i].float().log_softmax(-1);target=expected[i][0].float().log_softmax(-1)
                kl=(target.exp()*(target-lp)).sum().clamp_min(0)
                assert torch.isfinite(kl)
                observations.append(dict(row=i,step=step,kl=float(kl),top1_match=int(lp.argmax())==int(target.argmax())))
            tokens=actual.argmax(-1)[:,None]
            active=torch.tensor([[True],[step<15],[True],[True]],device=device)
            pred,_=batch.step(tokens,active);batch.detach_history();actual=pred[:,-1].float()
            expected_lengths=[ids.shape[1]+step+1 for ids in check_prompts]
            expected_lengths[1]=check_prompts[1].shape[1]+min(step+1,15)
            assert batch.prefix_mask.sum(-1).tolist()==expected_lengths
            for i,ref in enumerate(refs):
                if bool(active[i]):
                    pred,_=ref.step(tokens[i:i+1]);ref.detach_history();expected[i]=pred[:,-1].float()
    result=dict(mean_kl=sum(r['kl'] for r in observations)/len(observations),
                max_kl=max(r['kl'] for r in observations),
                top1_agreement=sum(r['top1_match'] for r in observations)/len(observations),
                positions=len(observations),prompt_lengths=[ids.shape[1] for ids in check_prompts],
                thresholds=dict(mean_kl=.001,max_kl=.005,top1_agreement=.98))
    result['passed']=result['mean_kl']<=.001 and result['max_kl']<=.005 and result['top1_agreement']>=.98
    (out/'numerical.json').write_text(json.dumps(dict(summary=result,observations=observations),indent=2))
    print(json.dumps({'BATCHED_NUMERICAL':result}),flush=True)
    if not result['passed']:
        raise RuntimeError('batched HF numerical qualification failed')
    del batch, refs, expected
    torch.cuda.empty_cache()
    # Include prompt prefill and decode, and grade no truncated probe as MATH500.
    probes=[]
    for name,inputs,tokens in [('math_prompts_b32',prompts,128),('long4k_b32',[long_prompt]*32,128)]:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        outputs=batch_decoder.generate(inputs,tokens,set(),temperature=0.)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        n_tokens=sum(map(len,outputs))
        row=dict(name=name,prompt_lengths=[x.shape[1] for x in inputs],tokens=n_tokens,seconds=elapsed,
                 tokens_per_second=n_tokens/elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        probes.append(row);print(json.dumps({'BATCHED_THROUGHPUT':row}),flush=True)
    (out/'throughput.json').write_text(json.dumps(probes,indent=2))
    print('BATCHED_GENERATION_QUALIFIED',flush=True)


if __name__=='__main__':
    main()
