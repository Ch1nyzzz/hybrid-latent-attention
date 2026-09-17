"""Bounded, matched GPU qualification of graph decode before MATH deployment."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from .generate import BatchedLatentDecoder, LatentDecoder
from .graph_generate import GraphRollingStep
from .batched_engine import BatchedRollingEngine
from .register import LatentStudent
from .vendor_model import load_teacher
from .training_common import amp


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--batch', type=int, default=32)
    args = p.parse_args()
    device = torch.device('cuda')
    model = load_teacher('/trisol/input/model', 4, device, dtype=torch.bfloat16)
    student = LatentStudent.from_checkpoint(torch.load('/trisol/input/models/model-0/student-600.pt', map_location='cpu', weights_only=True), device).eval()
    decoder = BatchedLatentDecoder(model, student, 20000, 0)
    torch.manual_seed(719)
    report = {'qualification': [], 'benchmarks': []}
    def emit(key, row):
        report[key].append(row)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps({key: row}), flush=True)
    for width, steps in [(250, 16), (4096, 32)]:
        prompts = [torch.randint(100, 20000, (1,n), device=device) for n in (width, 37)]
        ref, _ = decoder.prefill_batch(prompts)
        engine = BatchedRollingEngine(model, student, False)
        engine.seed_history(tuple(x.clone() for x in ref.prefix), ref.prefix_mask.clone())
        graph = GraphRollingStep(engine)
        divergences, matches, errors = [], [], []
        for i in range(steps):
            ids = torch.randint(100, 20000, (2,1), device=device)
            valid = torch.tensor([[True], [i < 3]], device=device)
            with amp(device):
                expected,_ = ref.step(ids,valid); ref.detach_history()
                actual,_ = graph.step(ids,valid)
            rows = valid[:,0]
            a,b=expected[rows,-1].float().log_softmax(-1),actual[rows,-1].float().log_softmax(-1)
            divergences.extend((a.exp()*(a-b)).sum(-1).tolist())
            matches.extend((a.argmax(-1)==b.argmax(-1)).tolist())
            errors.append(float((a-b).abs().max()))
            if graph.positions.tolist() != ref.positions.tolist() or not torch.equal(graph.prefix_mask, ref.prefix_mask):
                raise RuntimeError('Graph changed positions or stopped-row history mask')
        row=dict(width=width,positions=len(matches),mean_kl=sum(divergences)/len(divergences),max_kl=max(divergences),top1=sum(matches)/len(matches),max_logprob_error=max(errors),captures=graph.capture_count)
        row['passed']=row['mean_kl']<=.001 and row['max_kl']<=.005 and row['top1']>=.98
        if width==250 and graph.capture_count != 2:
            raise RuntimeError('Capacity crossing was not exercised')
        emit('qualification',row)
        if not row['passed']:raise RuntimeError('Graph numerical qualification failed')
        del ref,engine,graph,actual,expected
        gc.collect();torch.cuda.empty_cache()
    # Identical weights, packed prefix, input tokens and step count. Report capture
    # separately, and include it in total decode time (prefill shared by both).
    for width in (128,4096):
        prompt=torch.randint(100,20000,(1,width),device=device)
        history,_=LatentDecoder(model,student,20000,0).prefill(prompt)
        for mode in ('eager','graph'):
            e=BatchedRollingEngine(model,student,False)
            e.seed_history(tuple(x.expand(args.batch,-1,-1).clone() for x in history),torch.ones(args.batch,width,device=device,dtype=torch.bool))
            runner=GraphRollingStep(e) if mode=='graph' else e
            ids=torch.full((args.batch,1),1234,device=device,dtype=torch.long)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            with amp(device):
                runner.step(ids);runner.detach_history()
                torch.cuda.synchronize();first=time.perf_counter()-t
                start=time.perf_counter()
                for _ in range(127):
                    runner.step(ids);runner.detach_history()
                torch.cuda.synchronize()
            steady=time.perf_counter()-start
            emit('benchmarks',dict(mode=mode,width=width,batch=args.batch,steps=128,first_step_seconds=first,steady_seconds=steady,steady_tokens_per_second=args.batch*127/steady,total_seconds=first+steady,total_tokens_per_second=args.batch*128/(first+steady),peak_gib=torch.cuda.max_memory_allocated()/2**30))
            del runner,e
            gc.collect();torch.cuda.empty_cache()
        del history
    print('GRAPH_QUALIFICATION_DONE',flush=True)

if __name__=='__main__':main()
