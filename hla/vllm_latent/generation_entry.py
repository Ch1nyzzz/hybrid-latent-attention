"""Default generation entry: run vLLM in its own dependency environment."""
import os
from pathlib import Path
import shutil
import sys


def evaluation_command(args):
    if args.resume_from:
        raise ValueError('HF answer-resume imports are not compatible with vLLM; choose a fresh output directory')
    if args.prompt_chunk_size != 0:
        raise ValueError('vLLM S6 generation requires full prompt prefill (--prompt-chunk-size 0)')
    root = Path(__file__).resolve().parents[2]
    out = Path(args.output).resolve()
    shim = out/'s6shim'; shim.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root/'hla/vllm_latent/s6_sitecustomize.py', shim/'sitecustomize.py')
    env = dict(os.environ, PYTHONPATH=f'{shim}:{root}', VLLM_USE_FLASHINFER_SAMPLER='0',
               S6_VLLM_OURO='alias' if args.student else 'off',
               S6_VLLM_OURO_FILE=str(root/'hla/vllm_latent/ouro_latent.py'),
               VLLM_CACHE_ROOT=str(out/f'vllm-cache-{args.shard}'))
    argv = [sys.executable, '-m', 'hla.vllm_latent.matheval',
            '--model', str(Path(args.model_path).resolve()), '--data', str(Path(args.data).resolve()),
            '--output', str(out), '--loops', str(args.loops), '--max-new', str(args.max_new),
            '--n', str(args.n), '--temperature', str(args.temperature), '--top-p', str(args.top_p),
            '--seed', str(args.seed), '--shard', str(args.shard), '--nshards', str(args.nshards),
            '--limit', str(args.limit), '--max-model-len', str(args.max_model_len or 4096+args.max_new),
            '--max-num-seqs', str(max(args.batch, args.n)), '--auto-concurrency',
            '--engine-log', str(out/f'engine-{args.shard}.log')]
    argv += ['--student', str(Path(args.student).resolve())] if args.student else ['--base']
    return argv, env
