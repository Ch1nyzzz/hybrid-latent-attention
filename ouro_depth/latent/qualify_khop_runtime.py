"""Bounded BF16 forward/optimizer smoke check, without gradient comparisons.

Runs before the full 8-GPU launch. Disposable updates never become training init.
"""
import argparse,json,math,time
from pathlib import Path
import torch
from .batched_engine import BatchedRollingEngine
from .decode_training import PromptIndex,Trajectory,token_logp,score_teacher
from .history_snapshot import collect_snapshot, load_rollout_snapshot
from .khop_replay import parallel_forward,replay_batch_khop
from .teacher import Teacher
from .training_common import load_export,TeacherTargets,synchronize_gradients


def main():
    p=argparse.ArgumentParser()
    for name in ('model','student','data','output','mode'):p.add_argument('--'+name,required=True)
    p.add_argument('--rollout-cache', action='store_true')
    p.add_argument('--soak-prompts', type=int, default=0)
    p.add_argument('--soak-max-new', type=int, default=2048)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.soak_prompts and not (a.mode=='opd' and a.rollout_cache):
        raise ValueError('The EOS soak requires OPD rollout-cache mode')
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');student,_=load_export(a.student,device);student.eval()
    teacher=Teacher(a.model,student.cfg['loops'],device,dtype=torch.bfloat16)
    model=teacher.model;teacher.remove_hooks()
    idx=PromptIndex(Path(a.data)/'dev.jsonl',1024,64)
    row=idx.sample_at(0,20260915);idx.close()
    ids=torch.tensor(row['input_ids'],device=device)[None];prompt=row['prompt_len']
    worker=None;opd=None
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if a.mode=='opd':
                from .verl_opd import VerlOPDLoss
                from .vllm_rollout import VLLMRollout
                opd=VerlOPDLoss()
                eos=model.config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])-{None}
                worker=VLLMRollout(a.model,out/'rollout',device=device,batch_size=2 if a.rollout_cache else 1,max_prompt=1024,
                    max_new=64,seed=20260915,kv_bytes=2*2**30,export_cache=a.rollout_cache)
                prompts=[ids[:,:prompt]]
                if a.rollout_cache:prompts.append(ids[:,:max(1,prompt//2)])
                generation_started=time.monotonic()
                trajectories=worker.generate(student,prompts,eos_ids=eos,version=0)
                generation_seconds=time.monotonic()-generation_started
                trajectory=trajectories[0]
                if a.rollout_cache:
                    for item in trajectories:load_rollout_snapshot(item,student)
                ids=trajectory.ids
                tlp=score_teacher(model,trajectory)
                logits=None;targets={}
            else:
                trajectory=Trajectory(ids,prompt,0)
                capture=Teacher.wrap(model)
                logits,targets=TeacherTargets(capture)(ids[:,:-1]);capture.remove_hooks();tlp=None
            with torch.no_grad():
                engine=BatchedRollingEngine(model,student,False,serving_numerics=True)
                first,_=engine.prefill(ids[:,:prompt],last_logits_only=True);engine.detach_history();serial=[first]
                for i in range(prompt,ids.shape[1]-1):
                    pred,_=engine.step(ids[:,i:i+1]);engine.detach_history();serial.append(pred)
                serial=torch.cat(serial,1)
                snapshot=load_rollout_snapshot(trajectory,student) if a.rollout_cache else collect_snapshot(model,student,ids,prompt,serving_numerics=True)
            captured=[]
            handle=model.lm_head.register_forward_hook(lambda module,inputs,output:captured.append(output.detach()))
            with torch.no_grad():
                if trajectory.response_length>1:
                    parallel_forward(model,student,ids,prompt,snapshot.rows,logits,targets,
                        lam_attn=.1,normalizer=trajectory.response_length,serving_numerics=True,
                        trajectory=trajectory,teacher_logp=tlp,opd_loss=opd)
                    parallel=torch.cat((snapshot.first_response_logits,captured[-1]),1)
                else:parallel=snapshot.first_response_logits
            handle.remove()
            labels=ids[:,prompt:]
            delta=(token_logp(parallel,labels)-token_logp(serial,labels)).abs()
            report=dict(mode=a.mode,response=trajectory.response_length,logits_max=float((parallel.float()-serial.float()).abs().max()),
                logp_max=float(delta.max()),logp_mean=float(delta.mean()))
            if not bool(torch.isfinite(parallel).all()) or report['logp_max']>.25 or report['logp_mean']>.05:
                raise RuntimeError('BF16 parallel/serial forward check failed: '+json.dumps(report))
            if opd is not None:
                drift=(token_logp(parallel,labels)-trajectory.old_logp).abs()
                report.update(rollout_logp_max=float(drift.max()),rollout_logp_mean=float(drift.mean()))
                if report['rollout_logp_max']>.25 or report['rollout_logp_mean']>.05:
                    raise RuntimeError('OPD behavior/replay check failed: '+json.dumps(report))
            del serial,parallel,snapshot,captured
            optimizer=torch.optim.AdamW(student.parameters(),lr=1e-6)
            probe=next(student.parameters());before=probe.detach().clone()
            started=time.monotonic()
            metrics=replay_batch_khop(model,student,trajectory,hops=3,normalizer=trajectory.response_length,
                teacher_logits=logits,targets=targets,teacher_logp=tlp,opd_loss=opd,
                serving_numerics=True,checkpointing=True,history_source='rollout' if a.rollout_cache else 'collect')
            norm=synchronize_gradients(student)
            if not math.isfinite(norm) or norm<=0:raise RuntimeError('No finite nonzero update')
            optimizer.step();torch.cuda.synchronize()
            report.update(metrics,grad_norm=norm,update_seconds=time.monotonic()-started,
                parameter_max_change=float((probe-before).abs().max()),passed=True)
            if report['parameter_max_change']==0:raise RuntimeError('Optimizer did not change probe parameter')
            if a.rollout_cache:
                if metrics['history_collect_seconds'] != 0:raise RuntimeError('Unexpected serial collection')
                report.update(cache_export_seconds=worker.last_cache_export_seconds,generation_including_startup_seconds=generation_seconds)
                changed=worker.generate(student,prompts,eos_ids=eos,version=1)
                for item in changed:load_rollout_snapshot(item,student)
                report.update(second_version=1,second_version_requests=len(changed),cache_bytes=sum(t.history_ref['bytes'] for t in trajectories))
                if a.soak_prompts:
                    # Full-length soak: early-EOS trajectories are where cache/trajectory
                    # length alignment breaks, and the short numerics check never hits them.
                    # Close the numerics worker first: two engines at gpu_memory .35 each
                    # plus the trainer would sit at the edge of an 80 GiB card.
                    worker.close();worker=None
                    soak_worker=VLLMRollout(a.model,out/'soak',device=device,batch_size=a.soak_prompts,
                        max_prompt=1024,max_new=a.soak_max_new,seed=20260917,kv_bytes=4*2**30,export_cache=True)
                    try:
                        soak_idx=PromptIndex(Path(a.data)/'dev.jsonl',1024,a.soak_max_new)
                        soak_prompts=[torch.tensor(soak_idx.sample_at(i,20260917)['prompt_ids'],device=device)[None]
                                      for i in range(a.soak_prompts)]
                        soak_idx.close()
                        soak_started=time.monotonic()
                        soaked=soak_worker.generate(student,soak_prompts,eos_ids=eos,version=0)
                        for item in soaked:load_rollout_snapshot(item,student)
                        stopped=sum(not t.truncated for t in soaked)
                        if not stopped:
                            raise RuntimeError('Soak batch produced no EOS-stopped trajectory')
                        report.update(soak_prompts=a.soak_prompts,soak_max_new=a.soak_max_new,
                            soak_eos_stopped=stopped,soak_seconds=time.monotonic()-soak_started,
                            soak_cache_bytes=sum(t.history_ref['bytes'] for t in soaked),
                            soak_responses=[t.response_length for t in soaked])
                    finally:
                        soak_worker.close()
            (out/'qualification.json').write_text(json.dumps(report,indent=2))
            print('KHOP_RUNTIME_PASS '+json.dumps(report),flush=True)
    finally:
        if worker is not None:worker.close()
        teacher.remove_hooks()

if __name__=='__main__':main()
