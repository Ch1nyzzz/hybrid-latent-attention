"""Persistent, isolated vLLM S6 generation worker. One request = one weight version.

The HF trainer keeps transformers 4.x; this subprocess uses the image's vLLM
runtime. Replies are atomic files so engine stdout cannot corrupt the protocol.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

from .serving_config import compilation_kwargs, kv_capacity, parse_engine_log

FDO = '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'


def apply_student(model, payload, version):
    """Latent tensors of one package; shared by the latent-only and full-parameter entries."""
    import torch
    body = model.model
    if payload['version'] != version or payload['cfg'] != body.latent_cfg:
        raise ValueError('Rollout weight version/config mismatch')
    state = payload['student']
    targets = {f'layers.{i}.{name}': value for i, layer in enumerate(body.layers)
               for name, value in layer.self_attn.latent.state_dict().items()}
    if state.keys() != targets.keys():
        raise ValueError('Rollout student keys differ')
    for name, target in targets.items():
        source = state[name]
        if source.shape != target.shape or not torch.isfinite(source).all():
            raise ValueError('Invalid rollout parameter: ' + name)
    with torch.no_grad():
        for name, target in targets.items():
            target.copy_(state[name])
    return {'version': version, 'tensors': len(targets)}


def unwrap_model(model):
    while not hasattr(model, 'model') and hasattr(model, 'unwrap'):
        model = model.unwrap()
    return model


def install_student(model, path, version):
    """Copy into existing tensors: captured CUDA graphs retain the same addresses."""
    import torch
    model = unwrap_model(model)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if 'backbone' in payload:
        raise ValueError('Full-parameter packages require update_s6_backbone')
    ack = apply_student(model, payload, version)
    torch.cuda.synchronize()
    return ack


def install_full_parameter(model, path, version):
    """One versioned backbone+latent payload under one ack; the eval init path shares the loader."""
    import torch
    from .backbone_sync import apply_backbone_update, package_backbone, prepare_backbone_update
    model = unwrap_model(model)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    backbone = package_backbone(payload)
    if backbone is None:
        raise ValueError('update_s6_backbone requires a full-parameter package')
    prepared = prepare_backbone_update(model, backbone)
    tensors = apply_student(model, payload, version)['tensors'] + apply_backbone_update(model, backbone, prepared=prepared)
    torch.cuda.synchronize()
    return {'version': version, 'tensors': tensors}


class S6RolloutWorkerExtension:
    """Named worker RPC: only a path and integer cross the vLLM boundary."""
    def update_s6_student(self, path, version):
        ack = install_student(self.model_runner.get_model(), path, version)
        self.s6_weight_version = version
        return ack

    def update_s6_backbone(self, path, version):
        ack = install_full_parameter(self.model_runner.get_model(), path, version)
        self.s6_weight_version = version
        return ack

    def begin_s6_cache_export(self, directory, version):
        from .cache_export import RolloutCacheExporter
        if getattr(self, 's6_weight_version', None) != version:
            raise ValueError('Export weight version was not acknowledged')
        if not hasattr(self, 's6_exporter'):
            self.s6_exporter = RolloutCacheExporter(self.model_runner)
        self.s6_exporter.begin(directory, version)
        return version

    def finish_s6_cache_export(self):
        return self.s6_exporter.finish()


def encode_outputs(outputs, prompts, max_new, eos_ids):
    """Reject reordered, partial, nonfinite or malformed sampled trajectories."""
    if len(outputs) != len(prompts):
        raise ValueError('Incomplete vLLM rollout batch')
    result = []
    for output, prompt in zip(outputs, prompts):
        if list(output.prompt_token_ids) != prompt or len(output.outputs) != 1:
            raise ValueError('vLLM prompt/output alignment mismatch')
        completion = output.outputs[0]
        tokens = list(completion.token_ids)
        if not tokens or len(tokens) > max_new or len(completion.logprobs or []) != len(tokens):
            raise ValueError('Missing sampled token logprobs')
        values = [float(step[token].logprob) for token, step in zip(tokens, completion.logprobs)]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite rollout logprob')
        if any(t in eos_ids for t in tokens[:-1]):
            raise ValueError('Tokens after EOS')
        if completion.finish_reason not in ('stop', 'length'):
            raise ValueError('Unsuccessful generation')
        if completion.finish_reason == 'stop' and tokens[-1] not in eos_ids:
            raise ValueError('Unrequested stop condition')
        if completion.finish_reason == 'length' and len(tokens) != max_new:
            raise ValueError('Unexpected context truncation')
        result.append(dict(tokens=tokens, logps=values, truncated=completion.finish_reason == 'length'))
    return result


def generate_with_request_ids(llm, prompts, sampling_params):
    """Keep the engine-assigned external->internal mapping, including random IDs."""
    processor = llm.llm_engine.input_processor
    assign = processor.assign_request_id
    request_ids = {}

    def assigned(request):
        result = assign(request)
        external = request.external_req_id
        if external in request_ids:
            raise RuntimeError('Duplicate external rollout request ID')
        request_ids[external] = request.request_id
        return result

    processor.assign_request_id = assigned
    try:
        outputs = llm.generate([{'prompt_token_ids': x} for x in prompts], sampling_params, use_tqdm=False)
    finally:
        processor.assign_request_id = assign
    return outputs, request_ids


def atomic_reply(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, allow_nan=False))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    from vllm import LLM, SamplingParams
    llm = None
    last_version = -1
    for line in sys.stdin:
        request = json.loads(line)
        try:
            version = request['version']
            if version <= last_version:
                raise ValueError('Rollout versions must increase')
            prompts = request['prompts']
            if not prompts or len(prompts) > config['batch_size'] or any(
                    not x or len(x) > config['max_prompt'] for x in prompts):
                raise ValueError('Rollout batch outside configured bounds')
            if llm is None:
                # Async scheduling schedules one step ahead of sampling: an EOS-finished
                # request gets one extra executed token, which would export P+n cache
                # rows for a P+n-1 trajectory. The exporter requires synchronous steps.
                llm = LLM(model=config['model'], trust_remote_code=True, dtype='bfloat16',
                    worker_extension_cls='ouro_depth.vllm_latent.rollout_worker.S6RolloutWorkerExtension',
                    hf_overrides={'latent_student': request['weights'],
                                  **({'latent_window': config['window']} if config.get('window') else {})},
                    attention_backend='TRITON_ATTN', enable_prefix_caching=False,
                    enable_chunked_prefill=False, async_scheduling=False, generation_config='vllm',
                    max_model_len=config['max_prompt']+config['max_new'],
                    max_num_batched_tokens=max(8192, config['max_prompt']+config['max_new']),
                    max_num_seqs=config['batch_size'], kv_cache_memory_bytes=config['kv_bytes'],
                    gpu_memory_utilization=config['gpu_memory'], seed=config['seed'],
                    max_logprobs=max(20, config.get('logprobs', 0)),
                    **compilation_kwargs(FDO, config['batch_size']))
                sys.stdout.flush(); sys.stderr.flush()
                pool = parse_engine_log(Path(config['log']).read_text(errors='replace'))['kv_cache_tokens']
                capacity = kv_capacity(pool, config['batch_size'], config['max_prompt']+config['max_new'])
                if config.get('window'):
                    # vLLM's startup pool estimate budgets the sliding-window exact layers at full length; bound the
                    # real need instead (latent rows at full length + the window pages + one full exact prompt in
                    # flight). The exporter still refuses any preempted request at runtime.
                    import torch
                    body = json.loads((Path(config['model'])/'config.json').read_text())
                    cfg = torch.load(request['weights'], map_location='cpu', weights_only=False)['cfg']
                    latent = 2 * body['num_hidden_layers'] * (cfg['rank'] + cfg['rank_v'] + 2 * cfg['rank1'])
                    exact = 2 * body['num_hidden_layers'] * cfg['loops'] * 2 * body['hidden_size']
                    need = config['batch_size'] * ((config['max_prompt']+config['max_new']) * latent
                                                    + (config['window'] + 48) * exact) + config['max_prompt'] * exact
                    capacity = dict(capacity, kv_fits=need <= config['kv_bytes'], window_need_bytes=need)
                if not capacity['kv_fits']:
                    raise RuntimeError('Insufficient KV capacity: preemption would change S6 semantics: '+str(capacity))
            entry = 'update_s6_backbone' if request.get('full_parameter') else 'update_s6_student'
            ack = llm.collective_rpc(entry, args=(request['weights'], version))
            if len(ack) != 1 or ack[0]['version'] != version:
                raise RuntimeError('Worker did not acknowledge current weights')
            # Prefix caching is disabled and the previous synchronous generate drained
            # every request; no old-version KV can be reused after this copy.
            sp = SamplingParams(n=1, temperature=1., top_p=1., top_k=-1,
                max_tokens=config['max_new'], logprobs=config.get('logprobs', 0), ignore_eos=True,
                stop_token_ids=request['eos_ids'], seed=config['seed']+version)
            if request.get('cache_directory'):
                export_ack = llm.collective_rpc('begin_s6_cache_export', args=(request['cache_directory'], version))
                if export_ack != [version]:
                    raise RuntimeError('Cache export not acknowledged')
            if request.get('cache_directory'):
                outputs, request_ids = generate_with_request_ids(llm, prompts, sp)
            else:
                outputs = llm.generate([{'prompt_token_ids': x} for x in prompts], sp, use_tqdm=False)
            result = encode_outputs(outputs, prompts, config['max_new'], set(request['eos_ids']))
            export_seconds = 0.
            if request.get('cache_directory'):
                exports = llm.collective_rpc('finish_s6_cache_export')
                if len(exports) != 1 or exports[0]['version'] != version:
                    raise RuntimeError('Invalid cache export response')
                snapshots = exports[0]['snapshots']
                if set(snapshots) != {request_ids[o.request_id] for o in outputs}:
                    raise RuntimeError('Incomplete rollout cache export')
                export_seconds = exports[0]['export_seconds']
                for output, row in zip(outputs, result):
                    row['history_ref'] = snapshots[request_ids[output.request_id]]
                    row['request_id'] = request_ids[output.request_id]
                    row['history_ref']['token_ids'] = list(output.prompt_token_ids) + row['tokens']
            diagnostics = None
            if config.get('logprobs', 0):
                diagnostics = [[{'ids': list(step), 'lp': [float(x.logprob) for x in step.values()]}
                                for step in output.outputs[0].logprobs] for output in outputs[:config.get('diagnostic_limit')]]
            last_version = version
            atomic_reply(request['reply'], dict(version=version, trajectories=result, weight_ack=ack, topk=diagnostics, cache_export_seconds=export_seconds))
        except Exception:
            atomic_reply(request['reply'], {'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
