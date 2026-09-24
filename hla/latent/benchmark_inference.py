"""Matched fixed-token inference workload, S6 versus full recurrent Ouro KV.

No scores/accuracy are inferred from this token replay benchmark. Both methods
use full, serial per-request prefill and CUDA graph decode. The baseline keeps
the original model forward and SDPA; only cache allocation is made static.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..vendor.modeling_ouro import UniversalTransformerCache
from .generate import BatchedLatentDecoder
from .graph_generate import GraphRollingStep
from .register import LatentStudent
from .vendor_model import load_teacher


class StaticOuroCache(UniversalTransformerCache):
    """All loop/layer KV pairs, written once at the current GPU cursor."""
    def __init__(self, keys, values, length):
        super().__init__(len(keys))
        self.key_cache, self.value_cache = keys, values
        self.length = length

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        pos = cache_kwargs['cache_position']
        self.key_cache[layer_idx].index_copy_(2, pos, key_states)
        self.value_cache[layer_idx].index_copy_(2, pos, value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.length


class GraphOuroStep:
    def __init__(self, model, cache, width):
        self.model, self.cache = model, cache
        self.cursor = torch.tensor([width], device='cuda', dtype=torch.long)
        self.ids = torch.zeros(cache.key_cache[0].shape[0], 1, device='cuda', dtype=torch.long)
        self.columns = torch.arange(cache.key_cache[0].shape[2], device='cuda')
        self.graph = None
        self.used = width

    def run(self):
        mask = (self.columns <= self.cursor[0])[None, None, None, :]
        out = self.model(self.ids, past_key_values=self.cache, use_cache=True,
                         cache_position=self.cursor, position_ids=self.cursor[None],
                         attention_mask={'full_attention': mask}, logits_to_keep=1,
                         exit_at_step=3).logits
        self.cursor.add_(1)
        return out

    @torch.no_grad()
    def step(self, ids):
        self.ids.copy_(ids)
        if self.graph is None:
            saved = self.cursor.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.run()
                    self.cursor.copy_(saved)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = self.run()
            torch.cuda.current_stream().wait_stream(stream)
            self.cursor.copy_(saved)
        self.graph.replay()
        self.used += 1
        return self.output, None


@torch.no_grad()
def prefill_ouro(model, prompts, capacity):
    keys, values = [], []
    for row, ids in enumerate(prompts):
        result = model(ids, use_cache=True, logits_to_keep=1, exit_at_step=3)
        cache = result.past_key_values
        if row == 0:
            for k, v in zip(cache.key_cache, cache.value_cache):
                shape = (len(prompts), k.shape[1], capacity, k.shape[3])
                keys.append(k.new_zeros(shape))
                values.append(v.new_zeros(shape))
        for dst, src in zip(keys, cache.key_cache):
            dst[row:row+1, :, :ids.shape[1]].copy_(src)
        for dst, src in zip(values, cache.value_cache):
            dst[row:row+1, :, :ids.shape[1]].copy_(src)
        del result, cache
    return GraphOuroStep(model, StaticOuroCache(keys, values, prompts[0].shape[1]), prompts[0].shape[1])


def clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def memory():
    return dict(allocated_gib=torch.cuda.memory_allocated()/2**30,
                reserved_gib=torch.cuda.memory_reserved()/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)


def make_tokens(tokenizer, width, steps, requests):
    # Deterministic contiguous token windows from real math text; long windows
    # can span documents. These are controlled workload inputs, not scored tasks.
    rows = [json.loads(s) for s in Path('hla/matheval/data/math500.jsonl').read_text().splitlines()]
    text = '\n\n'.join(r['problem']+'\n'+r.get('solution', r.get('answer', '')) for r in rows)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    size = (width+steps)*requests
    tokens = (tokens*((size+len(tokens)-1)//len(tokens)))[:size]
    ids = torch.tensor(tokens, dtype=torch.long).reshape(requests, width+steps)
    return ids, hashlib.sha256(ids.numpy().tobytes()).hexdigest()


@torch.no_grad()
def qualify(model, student, ids, method):
    """Graph vs native rolling execution on identical growing token histories."""
    prompt, continuation = ids[:1, :-16].cuda(), ids[:1, -16:].cuda()
    if method == 'ouro':
        ref = model(prompt, use_cache=True, logits_to_keep=1, exit_at_step=3).past_key_values
        graph = prefill_ouro(model, [prompt], ((prompt.shape[1]//256)+1)*256)
    else:
        decoder = BatchedLatentDecoder(model, student, 20000, 0)
        ref, _ = decoder.prefill_batch([prompt])
        actual, _ = decoder.prefill_batch([prompt])
        graph = GraphRollingStep(actual)
    kl, top = [], []
    for index in range(16):
        token = continuation[:, index:index+1]
        if method == 'ouro':
            out = model(token, past_key_values=ref, use_cache=True, logits_to_keep=1, exit_at_step=3)
            ref, expected = out.past_key_values, out.logits
        else:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                expected, _ = ref.step(token)
                ref.detach_history()
        actual, _ = graph.step(token)
        a, b = expected[:, -1].float().log_softmax(-1), actual[:, -1].float().log_softmax(-1)
        kl.extend((a.exp()*(a-b)).sum(-1).tolist())
        top.extend((a.argmax(-1)==b.argmax(-1)).tolist())
    result = dict(mean_kl=sum(kl)/len(kl), max_kl=max(kl), top1=sum(top)/len(top), positions=len(kl))
    result['passed'] = result['mean_kl'] < .002 and result['max_kl'] < .01 and result['top1'] >= .9375
    return result


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(719)
    model = load_teacher(args.model, 4, torch.device('cuda'), torch.bfloat16)
    student = None
    if args.method == 's6':
        student = LatentStudent.from_checkpoint(torch.load(args.student, map_location='cpu', weights_only=True), torch.device('cuda')).eval()
        student.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids, digest = make_tokens(tokenizer, args.width, args.steps, args.requests)
    output = dict(method=args.method, width=args.width, steps=args.steps, batch=args.batch,
                  requests=args.requests, input_sha256=digest, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, dtype='BF16', loops=4, backend='HF-CUDA-graph',
                  prompt_policy='serial full prompt', workload='fixed math-text token replay; no EOS; not task scoring',
                  repeats=[], qualification=None)
    path = Path(args.output)
    def persist():
        path.write_text(json.dumps(output, indent=2))
    # An independent single-row reference check on this context length.
    qids, _ = make_tokens(tokenizer, args.width, 16, 1)
    output['qualification'] = qualify(model, student, qids, args.method)
    persist()
    if not output['qualification']['passed']:
        raise RuntimeError(f"Numerical qualification failed: {output['qualification']}")
    clean()
    output['weights'] = memory()
    for repeat in range(args.repeats):
        clean()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        prefill_seconds = decode_seconds = first_step_seconds = 0.
        replay_seconds = replay_tokens = cache_bytes = 0
        batches = []
        for begin in range(0, args.requests, args.batch):
            batch_ids = ids[begin:begin+args.batch].cuda()
            prompts = [row[None, :args.width] for row in batch_ids]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.method == 'ouro':
                capacity = ((args.width+args.steps-1)//256+1)*256
                runner = prefill_ouro(model, prompts, capacity)
            else:
                decoder = BatchedLatentDecoder(model, student, 20000, 0)
                engine, _ = decoder.prefill_batch(prompts)
                runner = GraphRollingStep(engine, minimum_capacity=((args.width+args.steps-1)//256+1)*256)
                del engine
            torch.cuda.synchronize()
            pref = time.perf_counter()-t0
            prefill_seconds += pref
            t0 = time.perf_counter()
            logits, _ = runner.step(batch_ids[:, args.width:args.width+1])
            del logits
            torch.cuda.synchronize()
            first = time.perf_counter()-t0
            first_step_seconds += first
            decode_start = time.perf_counter()
            for offset in range(1, args.steps):
                logits, _ = runner.step(batch_ids[:, args.width+offset:args.width+offset+1])
                del logits
            torch.cuda.synchronize()
            replay = time.perf_counter()-decode_start
            replay_seconds += replay
            replay_tokens += len(prompts)*(args.steps-1)
            decode_seconds += first+replay
            tensors = (runner.cache.key_cache+runner.cache.value_cache if args.method == 'ouro' else list(runner.engine.prefix))
            allocated_cache = sum(t.numel()*t.element_size() for t in tensors)
            cache_bytes = max(cache_bytes, allocated_cache)
            batches.append(dict(requests=len(prompts), prefill_seconds=pref, capture_first_step_seconds=first,
                                replay_seconds=replay, cache_allocated_gib=allocated_cache/2**30))
            del tensors, runner, prompts, batch_ids
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        row = dict(repeat=repeat, total_seconds=seconds, prefill_seconds=prefill_seconds,
                   decode_including_capture_seconds=decode_seconds, capture_first_step_seconds=first_step_seconds,
                   replay_seconds=replay_seconds, replay_tokens_per_second=replay_tokens/replay_seconds,
                   replay_ms_per_step=1000*replay_seconds/((args.steps-1)*len(batches)),
                   output_tokens=args.requests*args.steps,
                   end_to_end_output_tokens_per_second=args.requests*args.steps/seconds,
                   max_cache_allocated_gib=cache_bytes/2**30, batches=batches, **memory())
        output['repeats'].append(row)
        persist()
        print('INFERENCE_PROGRESS '+json.dumps({k:v for k,v in row.items() if k!='batches'}), flush=True)
    output['complete'] = True
    persist()
    print('INFERENCE_RESULT '+json.dumps(output), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['ouro', 's6'], required=True)
    p.add_argument('--model', default='/trisol/input/model')
    p.add_argument('--student', default='/trisol/input/models/model-0/student-600.pt')
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--requests', type=int, required=True)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if min(args.width,args.batch,args.requests)>0 and args.steps>1:
        run(args)
    else:
        p.error('Require positive sizes and at least two decode steps')


if __name__ == '__main__':
    main()
