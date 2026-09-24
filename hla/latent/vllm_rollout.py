"""Trainer-side client for the project's fused vLLM S6 adapter (no HF fallback)."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

from .decode_training import Trajectory


def worker_environment(root, work, device, base_env):
    env = {k: v for k, v in base_env.items() if not (k in {
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
        'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'} or k.startswith('TORCHELASTIC_'))}
    visible = base_env.get('CUDA_VISIBLE_DEVICES')
    gpu = visible.split(',')[device] if visible else str(device)
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=f'{work}/shim:{root}',
        S6_VLLM_OURO='alias', S6_VLLM_OURO_FILE=str(root/'hla/vllm_latent/ouro_latent.py'),
        VLLM_CACHE_ROOT=str(work/'cache'), VLLM_USE_FLASHINFER_SAMPLER='0',
        VLLM_LOGGING_LEVEL='INFO', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
        # Eight engines starting at once raced on random ports (EADDRINUSE killed one rank's engine and hung the
        # others in NCCL); give each device its own range, clear of MATH500 (18000+) and torchrun (29500).
        VLLM_PORT=str(40000 + 100 * device))
    return env


class VLLMRollout:
    def __init__(self, model_path, work, *, device, batch_size, max_prompt, max_new,
                 seed, kv_bytes=6*2**30, gpu_memory=.35, timeout=3600, logprobs=0, diagnostic_limit=None, export_cache=False,
                 window=0):
        if device.type != 'cuda':
            raise ValueError('Production OPD generation requires the S6 vLLM CUDA adapter')
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.device, self.timeout = device, timeout
        self.last_version = -1
        self.export_cache = export_cache
        root = Path(__file__).resolve().parents[2]
        shim = self.work/'shim'; shim.mkdir(exist_ok=True)
        shutil.copyfile(root/'hla/vllm_latent/s6_sitecustomize.py', shim/'sitecustomize.py')
        log = self.work/'worker.log'
        config = dict(model=model_path, batch_size=batch_size, max_prompt=max_prompt,
                      max_new=max_new, seed=seed, kv_bytes=kv_bytes, gpu_memory=gpu_memory, log=str(log), logprobs=logprobs, diagnostic_limit=diagnostic_limit,
                      window=window)
        config_path = self.work/'config.json'; config_path.write_text(json.dumps(config))
        self.log = log.open('a')
        self.process = subprocess.Popen([sys.executable, '-m', 'hla.vllm_latent.rollout_worker',
            '--config', str(config_path)], stdin=subprocess.PIPE, stdout=self.log, stderr=self.log,
            text=True, start_new_session=True,
            env=worker_environment(root, self.work, device.index or 0, os.environ))

    def generate(self, student, prompts, *, eos_ids, version):
        if version <= self.last_version:
            raise ValueError('Cannot reuse an old rollout version')
        weights = self.work/'student.pt'
        temporary = self.work/'student.tmp'
        payload = dict(student={k: v.detach().cpu() for k, v in student.state_dict().items()},
                       cfg=student.cfg, version=version)
        torch.save(payload, temporary)
        temporary.replace(weights)
        reply = self.work/f'reply-{version}.json'
        if reply.exists():
            raise FileExistsError('Stale worker reply: '+str(reply))
        request = dict(version=version, weights=str(weights), reply=str(reply),
                       prompts=[p.detach().cpu().reshape(-1).tolist() for p in prompts], eos_ids=sorted(eos_ids))
        if self.export_cache:
            request['cache_directory'] = str((self.work/f'history-{version}').resolve())
        self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        deadline = time.monotonic()+self.timeout
        while not reply.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'vLLM worker exited ({self.process.returncode}); see {self.work}/worker.log')
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError('vLLM rollout exceeded timeout')
            time.sleep(.1)
        result = json.loads(reply.read_text())
        if 'error' in result:
            raise RuntimeError(result['error'])
        if result['version'] != version or len(result['trajectories']) != len(prompts):
            raise ValueError('Worker returned stale or incomplete trajectories')
        trajectories = []
        for prompt, row in zip(prompts, result['trajectories']):
            tail = torch.tensor(row['tokens'], device=self.device, dtype=torch.long)[None]
            trajectories.append(Trajectory(torch.cat((prompt, tail), 1), prompt.shape[1], version,
                torch.tensor(row['logps'], device=self.device, dtype=torch.float32)[None], row['truncated'],
                row.get('history_ref'), row.get('request_id')))
            if self.export_cache and (not row.get('history_ref') or not row.get('request_id')):
                raise ValueError('Missing required rollout cache')
        self.last_cache_export_seconds = result.get('cache_export_seconds', 0.)
        self.last_topk = result.get('topk')
        self.last_version = version
        reply.unlink()
        return trajectories

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=15)
        self.log.close()
        for directory in self.work.glob('history-*'):
            if directory.is_dir():
                shutil.rmtree(directory)
        # Temporary synchronization weights are not final training exports.
        for name in ('student.pt', 'student.tmp'):
            (self.work/name).unlink(missing_ok=True)

