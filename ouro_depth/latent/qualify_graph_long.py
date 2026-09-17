"""Deployment gate: worst-case B32 cache growth and end-to-end stop handling."""
import argparse,gc,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .batched_engine import BatchedRollingEngine
from .generate import BatchedLatentDecoder,INSTR
from .graph_generate import GraphLatentDecoder,GraphRollingStep
from .register import LatentStudent
from .vendor_model import load_teacher


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--compact-finished',action='store_true')
    args=parser.parse_args()
    device=torch.device('cuda')
    model=load_teacher('/trisol/input/model',4,device,torch.bfloat16)
    student=LatentStudent.from_checkpoint(torch.load('/trisol/input/models/model-0/student-600.pt',map_location='cpu',weights_only=True),device).eval()
    tok=AutoTokenizer.from_pretrained('/trisol/input/model',trust_remote_code=True)
    rows=[json.loads(x) for x in Path('ouro_depth/matheval/data/math500.jsonl').read_text().splitlines()]
    prompts=[tok(tok.apply_chat_template([{'role':'user','content':r['problem']+INSTR}],tokenize=False,add_generation_prompt=True),return_tensors='pt',add_special_tokens=False).input_ids.to(device) for r in rows[:4]]
    limit=max(x.shape[1] for x in prompts)+9
    ref=BatchedLatentDecoder(model,student,limit,0)
    graph=GraphLatentDecoder(model,student,limit,0,compact_finished=args.compact_finished)
    expected=ref.generate(prompts,12,set())
    actual=graph.generate(prompts,12,set())
    if actual!=expected:raise RuntimeError('End-to-end greedy/context-limit mismatch')
    stop={expected[0][1]}
    if ref.generate(prompts,12,stop)!=graph.generate(prompts,12,stop):
        raise RuntimeError('End-to-end EOS mismatch')
    print('GRAPH_GREEDY_EOS_CONTEXT_PASS',flush=True)
    # Synthetic history exercises allocation, mask/cursor growth and graph-pool
    # lifetime at a larger physical cache than the planned MATH prompts require.
    history,_=ref.prefill(prompts[0]); width=10230; batch=args.batch
    engine=BatchedRollingEngine(model,student,False)
    engine.seed_history(tuple(x.new_zeros(batch,width,x.shape[-1]) for x in history),torch.ones(batch,width,device=device,dtype=torch.bool))
    del history,ref,graph
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    graph=GraphRollingStep(engine,compact_finished=args.compact_finished)
    tokens=torch.full((batch,1),1234,device=device,dtype=torch.long)
    valid=torch.ones_like(tokens,dtype=torch.bool);valid[1]=False
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(20):
        logits,_=graph.step(tokens,valid)
        if not bool(torch.isfinite(logits).all()):raise RuntimeError('Nonfinite long-context output')
        del logits
    torch.cuda.synchronize()
    if graph.capture_count!=2 or graph.positions[0].item()!=width+20 or graph.positions[1].item()!=width:
        raise RuntimeError('Long-context capture/position state mismatch')
    result=dict(passed=True,kind='synthetic-cache-memory-and-state',batch=batch,initial_width=width,final_width=graph.used,captures=graph.capture_count,peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.perf_counter()-start,greedy_eos_context_pass=True)
    Path('/trisol/output/graph-long-qualification.json').write_text(json.dumps(result,indent=2))
    print('GRAPH_LONG_QUALIFICATION '+json.dumps(result),flush=True)

if __name__=='__main__':main()
