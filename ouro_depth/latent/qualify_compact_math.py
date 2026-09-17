"""Fixed-prefix compaction qualification and matched inference concurrency probes."""
import argparse,gc,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .batched_engine import BatchedRollingEngine
from .generate import LatentDecoder,INSTR
from .graph_generate import GraphRollingStep
from .register import LatentStudent
from .vendor_model import load_teacher


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    device=torch.device('cuda');torch.manual_seed(719)
    model=load_teacher('/trisol/input/model',4,device,torch.bfloat16)
    student=LatentStudent.from_checkpoint(torch.load('/trisol/input/models/model-0/student-600.pt',map_location='cpu',weights_only=True),device).eval()
    tok=AutoTokenizer.from_pretrained('/trisol/input/model',trust_remote_code=True)
    row=json.loads(Path('ouro_depth/matheval/data/math500.jsonl').read_text().splitlines()[0])
    prompt=tok(tok.apply_chat_template([{'role':'user','content':row['problem']+INSTR}],tokenize=False,add_generation_prompt=True),return_tensors='pt',add_special_tokens=False).input_ids.to(device)
    filler=tok.encode('The quick brown fox jumps over the lazy dog. ',add_special_tokens=False)
    report={'numerical':[],'benchmarks':[]}
    def emit(kind,row):
        report[kind].append(row);Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps({kind:row}),flush=True)
    def history(width):
        n=width-prompt.shape[1]
        ids=torch.cat([prompt.new_tensor([(filler*((n+len(filler)-1)//len(filler)))[:n]]),prompt],1)
        h,_=LatentDecoder(model,student,20000,0).prefill(ids)
        return h
    def runner(h,batch,compact):
        e=BatchedRollingEngine(model,student,False)
        e.seed_history(tuple(x.expand(batch,-1,-1).clone() for x in h),torch.ones(batch,h[0].shape[1],device=device,dtype=torch.bool))
        return GraphRollingStep(e,compact_finished=compact)
    for width in (250,4096):
        h=history(width);ref=runner(h,64,False);actual=runner(h,64,True)
        kl=[];top=[];sizes=set()
        for step in range(80):
            ids=(torch.arange(64,device=device)[:,None]+1000+step)
            valid=torch.zeros(64,1,device=device,dtype=torch.bool)
            chosen=list(range(64)) if step==0 else list(range(1,64,2)) if step<35 else [1,5,13,17,37,45,55,61] if step<67 else [5,55]
            valid[chosen]=True
            expected,_=ref.step(ids,valid);found,_=actual.step(ids,valid)
            a=expected[chosen,-1].float().log_softmax(-1);b=found[chosen,-1].float().log_softmax(-1)
            kl.extend((a.exp()*(a-b)).sum(-1).tolist());top.extend((a.argmax(-1)==b.argmax(-1)).tolist())
            sizes.add(actual.batch_size)
            if not torch.equal(actual.positions,ref.positions[actual.row_ids]) or not torch.equal(actual.prefix_mask,ref.prefix_mask[actual.row_ids]):raise RuntimeError('Compaction row/mask/position mismatch')
            del expected,found,a,b
        result=dict(width=width,positions=len(kl),mean_kl=sum(kl)/len(kl),max_kl=max(kl),top1=sum(top)/len(top),compute_batches=sorted(sizes))
        result['passed']=result['mean_kl']<=.001 and result['max_kl']<=.005 and result['top1']>=.98 and sizes=={64,32,8}
        emit('numerical',result)
        del ref,actual,h;gc.collect();torch.cuda.empty_cache()
    for width in (128,4096,8192):
        h=history(width if width<8192 else 4096)
        if width==8192:h=tuple(x.repeat(1,2,1) for x in h)
        shapes=[(32,32),(64,64)] if width<8192 else [(32,2),(8,2)]
        for batch,live in shapes:
            r=runner(h,batch,False)
            ids=torch.full((batch,1),1234,device=device,dtype=torch.long)
            valid=torch.zeros_like(ids,dtype=torch.bool);valid[:live]=True
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            for _ in range(128):
                logits,_=r.step(ids,valid);del logits
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            emit('benchmarks',dict(width=width,batch=batch,live=live,steps=128,seconds=seconds,useful_tokens_per_second=live*128/seconds,peak_gib=torch.cuda.max_memory_allocated()/2**30))
            del r;gc.collect();torch.cuda.empty_cache()
        del h
    if not all(r['passed'] for r in report['numerical']):raise RuntimeError('Compaction numerical gate failed')
    print('COMPACTION_QUALIFICATION_DONE',flush=True)
if __name__=='__main__':main()
