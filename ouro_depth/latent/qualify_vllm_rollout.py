"""GPU gate: mixed prompts, two live weight versions, vLLM logprobs vs replay."""
import argparse
import json
from pathlib import Path

import torch
from .batched_engine import BatchedRollingEngine
from .decode_training import PromptIndex, token_logp
from .teacher import Teacher
from .training_common import amp, load_export
from .vllm_rollout import VLLMRollout
from .logprob_metrics import position_metrics, summarize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--student',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--num-prompts',type=int,default=4)
    p.add_argument('--kv-gib',type=float,default=6.)
    args=p.parse_args()
    device=torch.device('cuda:0')
    student,_=load_export(args.student,device)
    teacher=Teacher(args.model,student.cfg['loops'],device,dtype=torch.bfloat16)
    teacher.remove_hooks();student.eval()
    corpus=PromptIndex(Path(args.data)/'dev.jsonl',1024,2048)
    # Different full-prompt lengths exercise vLLM's packed prompt boundaries.
    rows=[corpus.sample_at(i,20260915) for i in range(args.num_prompts)];corpus.close()
    prompts=[torch.tensor(r['prompt_ids'],device=device)[None] for r in rows]
    eos=teacher.model.config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])-{None}
    worker=VLLMRollout(args.model,Path(args.output)/'worker',device=device,batch_size=args.batch_size,
        max_prompt=3008,max_new=64,seed=918,logprobs=4096,diagnostic_limit=4,kv_bytes=int(args.kv_gib*2**30))
    reports=[]
    try:
        for version in range(2):
            if version:
                # Change every writer/reader so this checks a real graph-safe reload.
                with torch.no_grad():
                    for param in student.parameters():param.mul_(.99)
            with amp(device),torch.no_grad():
                trajectories=worker.generate(student,prompts,eos_ids=eos,version=version)
                errors=[];details=[];positions=[]
                for prompt_index,t in enumerate(trajectories[:4]):
                    engine=BatchedRollingEngine(teacher.model,student,False,serving_numerics=True)
                    pred,_=engine.prefill(t.ids[:,:t.prompt],chunk_size=t.prompt,last_logits_only=True)
                    engine.detach_history();preds=[pred]
                    for i in range(t.prompt,t.ids.shape[1]-1):
                        pred,_=engine.step(t.ids[:,i:i+1]);engine.detach_history();preds.append(pred)
                    logits=torch.cat(preds,1)
                    lp=token_logp(logits,t.ids[:,t.prompt:])
                    distribution=torch.log_softmax(logits[0].float(),-1)
                    for i,candidate in enumerate(worker.last_topk[prompt_index]):
                        positions.append(dict(id=prompt_index, **position_metrics(distribution[i],candidate['ids'],candidate['lp'])))
                    row_errors=(lp-t.old_logp).abs().flatten()
                    worst=int(row_errors.argmax())
                    details.append(dict(position=worst,old_logp=float(t.old_logp[0,worst]),replay_logp=float(lp[0,worst])))
                    errors.extend(row_errors.tolist())
            report=dict(distribution=summarize(positions), version=version,rollout_batch=len(prompts),prompt_lengths=[p.shape[1] for p in prompts[:4]],
                        tokens=len(errors),worst_positions=details,max_logp_error=max(errors),mean_logp_error=sum(errors)/len(errors))
            reports.append(report);print('VLLM_ROLLOUT_QUALIFICATION '+json.dumps(report),flush=True)
            if not all(torch.isfinite(torch.tensor(errors))):
                raise RuntimeError('Nonfinite vLLM/replay numerical qualification')
        Path(args.output,'qualification.json').write_text(json.dumps(reports,indent=2))
        if any(not r['distribution']['passed'] for r in reports):
            raise RuntimeError('vLLM/replay numerical qualification failed')
    finally:worker.close()


if __name__=='__main__':main()
