"""Persistent, isolated vLLM 0.26 Triton sampler for current-weight I3.

The subprocess uses the image's Transformers, not the trainer's 4.56 overlay.
Only token IDs return to training; no inference cache is used for gradients.
"""
import json
import os
from pathlib import Path
import subprocess
import time

import torch


def validate_completions(payload, version, prompts, max_new, eos_ids=(0, 2)):
    if payload.get('weight_version') != version:
        raise RuntimeError('Sampler returned a stale parameter version')
    rows = payload['completions']
    if len(rows) != len(prompts):
        raise RuntimeError('Sampler returned an incorrect number of completions')
    for row in rows:
        if not 1 <= len(row) <= max_new or any(type(t) is not int or t < 0 for t in row):
            raise RuntimeError('Invalid generated token sequence')
        if any(t in eos_ids for t in row[:-1]):
            raise RuntimeError('Sampler generated tokens after EOS')
    return rows


class TritonRollout:
    def __init__(self, model_path, directory, *, python='python', gpu_memory=.25,
                 max_seqs=16, timeout=3600):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model_path, self.python = str(model_path), python
        self.gpu_memory, self.max_seqs, self.timeout = gpu_memory, max_seqs, timeout
        self.process = self.log = None

    def _start(self):
        env = os.environ.copy()
        local_rank = int(env.get('LOCAL_RANK', 0))
        devices = env.get('CUDA_VISIBLE_DEVICES')
        env['CUDA_VISIBLE_DEVICES'] = devices.split(',')[local_rank] if devices else str(local_rank)
        for name in list(env):
            if name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                        'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE') or name.startswith('TORCHELASTIC_'):
                env.pop(name, None)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
        env['LATENT_FINALIZE_AFTER_READ'] = '1'
        env['LATENT_MANUAL_PREFILL'] = '0'
        env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        env['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
        self.log = (self.directory / 'worker.log').open('a')
        self.process = subprocess.Popen(
            [self.python, '-m', 'ouro_depth.latent.triton_rollout_worker',
             '--model', self.model_path, '--directory', str(self.directory),
             '--gpu-memory', str(self.gpu_memory), '--max-seqs', str(self.max_seqs)],
            env=env, stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT, text=True)

    def generate(self, student, version, prompts, seeds, max_new):
        if len(prompts) != len(seeds) or not prompts:
            raise ValueError('One seed per nonempty generation prompt is required')
        if self.process is None:
            self._start()
        snapshot = self.directory / 'current-student.pt'
        temporary = snapshot.with_suffix('.writing')
        # vLLM stores BF16 inference weights; FP32 optimizer masters stay in training.
        torch.save({'cfg': student.cfg, 'student': {k: v.detach().to(device='cpu', dtype=torch.bfloat16)
                                                 for k, v in student.state_dict().items()},
                    'weight_version': version}, temporary)
        temporary.replace(snapshot)
        # Release idle replay allocations before the separate inference process
        # wakes. The live training parameters/optimizer remain on this device.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        response = self.directory / f'response-{version}.json'
        response.unlink(missing_ok=True)
        request = {'snapshot': str(snapshot), 'response': str(response), 'weight_version': version,
                   'prompts': prompts, 'seeds': seeds, 'max_new': max_new}
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while not response.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'Triton worker exited ({self.process.returncode}); see {self.directory / "worker.log"}')
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError('Triton generation timed out')
            time.sleep(.2)
        payload = json.loads(response.read_text())
        if 'error' in payload:
            raise RuntimeError(f'Triton generation failed: {payload["error"]}')
        return validate_completions(payload, version, prompts, max_new)

    def close(self):
        if self.process:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=20)
            self.process = None
        if self.log:
            self.log.close()
            self.log = None
