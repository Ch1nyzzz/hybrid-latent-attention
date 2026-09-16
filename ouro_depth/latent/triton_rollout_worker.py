"""Run with the unmodified image Python environment, outside torchrun."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def load_student(model, *, snapshot, version):
    import torch
    ck = torch.load(snapshot, map_location='cpu', weights_only=False)
    if ck['weight_version'] != version or ck['cfg'] != model.model.latent_cfg:
        raise ValueError('Inference weight version/config mismatch')
    model.model._student_state = ck['student']
    loaded = model.model.load_latent_student()
    if loaded != len(ck['student']):
        raise ValueError('Incomplete latent weight reload')
    model.model._student_state = {}  # release CPU snapshot after all parameters were copied
    return {'weight_version': version, 'loaded_tensors': loaded}


class StudentWeightWorker:
    """Named RPC with primitive arguments; no pickled callable transport."""
    def reload_latent_student(self, snapshot, version):
        return load_student(self.get_model(), snapshot=snapshot, version=version)


def install_model():
    import fcntl
    import shutil
    import vllm
    from ouro_depth.vllm_latent.patch_triton import patch_installation
    root = Path(vllm.__file__).parent
    if vllm.__version__ != '0.26.0':
        raise RuntimeError(f'Only qualified vLLM 0.26.0 is supported, found {vllm.__version__}')
    # Multiple rank-local workers can start together; patch only under this lock.
    with open('/tmp/loop-scale-vllm-patch.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        patch_installation(root)
        source = Path(__file__).resolve().parents[1] / 'vllm_latent/ouro_latent.py'
        destination = root / 'model_executor/models/ouro.py'
        backup = destination.with_suffix('.py.before-loop-scale')
        if not backup.exists():
            shutil.copyfile(destination, backup)
        temporary = destination.with_suffix('.py.writing')
        shutil.copyfile(source, temporary)
        temporary.replace(destination)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--directory', required=True)
    p.add_argument('--gpu-memory', type=float, default=.25)
    p.add_argument('--max-seqs', type=int, default=16)
    args = p.parse_args()
    if os.environ.get('LATENT_FINALIZE_AFTER_READ') != '1':
        raise RuntimeError('Training rollout requires finalize-after-read semantics')
    install_model()
    from vllm import LLM, SamplingParams
    llm = None
    for line in sys.stdin:
        req = json.loads(line)
        response = Path(req['response'])
        try:
            if llm is None:
                llm = LLM(model=args.model, hf_overrides={'total_ut_steps': 4, 'latent_student': req['snapshot']},
                          trust_remote_code=True, dtype='bfloat16', enforce_eager=True,
                          attention_backend='TRITON_ATTN', enable_prefix_caching=False,
                          enable_chunked_prefill=False, enable_sleep_mode=True,
                          worker_extension_cls='ouro_depth.latent.triton_rollout_worker.StudentWeightWorker',
                          max_model_len=4096, max_num_batched_tokens=8192,
                          max_num_seqs=args.max_seqs, gpu_memory_utilization=args.gpu_memory)
            else:
                llm.wake_up()
            loaded = llm.collective_rpc('reload_latent_student',
                                        kwargs={'snapshot': req['snapshot'], 'version': req['weight_version']})
            if not loaded or any(r['weight_version'] != req['weight_version'] for r in loaded):
                raise RuntimeError('Worker weight synchronization failed')
            params = [SamplingParams(n=1, temperature=1., top_p=.7, max_tokens=req['max_new'],
                                     stop_token_ids=[0, 2], ignore_eos=False, seed=seed)
                      for seed in req['seeds']]
            outputs = llm.generate([{'prompt_token_ids': x} for x in req['prompts']], params, use_tqdm=False)
            completions = [list(o.outputs[0].token_ids) for o in outputs]
            llm.sleep(level=1)
            payload = {'weight_version': req['weight_version'], 'completions': completions,
                       'backend': 'TRITON_ATTN', 'loaded': loaded}
        except Exception as error:
            traceback.print_exc()
            payload = {'error': str(error), 'weight_version': req['weight_version']}
        temporary = response.with_suffix('.writing')
        temporary.write_text(json.dumps(payload))
        temporary.replace(response)
        if 'error' in payload:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
