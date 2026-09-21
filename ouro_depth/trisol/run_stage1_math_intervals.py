"""Original Stage1 training alternates with isolated n=1 MATH500 on the same GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from ouro_depth.trisol.math500_intervals import evaluate,inference_env,FDO


def targets():return tuple(range(100,601,100))


def prepare_hf_runtime(root):
    """Resume skips run_stage1_recipe.sh, so install its HF runtime explicitly.

    Keep the parent/vLLM environment unchanged; only HF subprocesses prepend
    stage1_deps. Validate in a fresh interpreter, before loading any model.
    """
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
        '--find-links','/trisol/input/datasets/ds-1','--target','/work/stage1_deps',
        'transformers==4.56.2','huggingface_hub==0.34.4'],check=True)
    env=dict(os.environ,PYTHONPATH=f'/work/stage1_deps:{root}')
    code=("import json, transformers, huggingface_hub; "
          "from ouro_depth.vendor.configuration_ouro import OuroConfig; "
          "assert transformers.__version__ == '4.56.2'; "
          "assert huggingface_hub.__version__ == '0.34.4'; "
          "assert hasattr(OuroConfig(), 'pad_token_id'); "
          "print(json.dumps({'event':'HF_REFERENCE_RUNTIME_READY', "
          "'transformers':transformers.__version__, 'path':transformers.__file__}),flush=True)")
    subprocess.run([sys.executable,'-c',code],env=env,check=True)


def prepare_resume_export(source,out):
    """Recover an inference export without resetting the archived optimizer/RNG."""
    import torch
    from ouro_depth.latent.training_common import SEMANTICS
    source,out=Path(source),Path(out)
    marker=json.loads((source/'complete.json').read_text())
    payload=torch.load(source/'training.pt',map_location='cpu',weights_only=False)
    if marker['completed_steps']!=8 or payload['completed_steps']!=8:
        raise ValueError('This continuation requires the qualified step-8 checkpoint')
    if payload['semantics']!=SEMANTICS or payload['metadata']['stage']!=1:
        raise ValueError('Invalid Stage1 resume semantics')
    expected=(int(os.environ['S6_RANK_K']),int(os.environ['S6_RANK_V']),256)
    if tuple(payload['cfg'][k] for k in ('rank','rank_v','rank1'))!=expected:
        raise ValueError('Resume rank mismatch')
    if payload['metadata']['world']!=8 or len(payload['rng_by_rank'])!=8 or not payload['optimizer']['state']:
        raise ValueError('Missing 8-rank optimizer/RNG recovery state')
    export={k:payload[k] for k in ('student','cfg','metadata','semantics')}
    export['step']=8
    out.mkdir(parents=True,exist_ok=True)
    path=out/'student-8.pt'
    if path.exists():raise FileExistsError(path)
    torch.save(export,path)
    print(json.dumps(dict(event='stage1_resume_export',completed_steps=8,source=str(source),
                         optimizer_states=len(payload['optimizer']['state']),rng_ranks=8)),flush=True)


def run(train_root=None,root=None,out=None):
    train_root=Path(train_root or '/work/loop_scale');root=Path(root or Path(__file__).resolve().parents[2])
    out=Path(out or '/trisol/output');model='/trisol/input/model'
    data=str(root/'ouro_depth/matheval/data/math500.jsonl')
    geometry=['--rank',os.environ['S6_RANK_K'],'--rank-v',os.environ['S6_RANK_V'],'--rank1','256']
    common=geometry+['--data-dir','/work/expanded-corpus']
    train_env=dict(os.environ,PYTHONPATH=str(train_root))
    resume_source=os.environ.get('S6_STAGE1_RESUME')
    # The driver chooses each interval's checkpoint, not the shell's native
    # first-resume environment (which would remain at step8 for every interval).
    train_env.pop('TRISOL_RESUME',None)
    train_env.pop('TRISOL_RESUME_CHECKPOINT',None)
    def train(target,resume=None,qualify=False):
        env=dict(train_env)
        if qualify:env['S6_QUALIFY']='1'
        else:env.pop('S6_QUALIFY',None)
        args=['bash',str(train_root/'ouro_depth/trisol/run_stage1_recipe.sh'),*common,'--stop-after',str(target)]
        if resume:args+=['--resume',str(out/f'checkpoint-{resume:06d}') if isinstance(resume,int) else str(resume)]
        subprocess.run(args,env=env,cwd=train_root,check=True)
        if json.loads((out/f'checkpoint-{target:06d}/complete.json').read_text())['completed_steps']!=target:
            raise RuntimeError('Incomplete checkpoint')
    if resume_source:
        prepare_hf_runtime(root)
        prepare_resume_export(resume_source,out)
    else:
        train(2,qualify=True);train(8,2,True)
        subprocess.run([sys.executable,str(train_root/'ouro_depth/trisol/verify_s6_stage1_qualification.py'),str(out),
            '--data-dir','/work/expanded-corpus',*geometry],env=train_env,check=True)
    def padded(step):
        path=out/'serving'/f'student-{step}-padded.pt'
        subprocess.run([sys.executable,'-m','ouro_depth.latent.pad_serving_rank',str(out/f'student-{step}.pt'),str(path)],check=True)
        return path
    student=padded(8)
    # Full fixed-prefix qualification once per geometry, before the first scored generation.
    qual=out/'rank-serving-qualification';qual.mkdir(exist_ok=True)
    hfenv=dict(os.environ,PYTHONPATH=f'/work/stage1_deps:{root}',CUDA_VISIBLE_DEVICES='0')
    def hf(args,log):
        with (qual/log).open('w') as f:subprocess.run([sys.executable,*args],env=hfenv,stdout=f,stderr=subprocess.STDOUT,check=True)
    hf(['-m','ouro_depth.latent.hf_reference','--model-path',model,'--student',str(student),'--data',data,
        '--output',str(qual/'hf'),'--n-prompts','4','--max-new','64','--prompt-chunk-size','0','--long-prompt-tokens','4096'],'hf.log')
    work=qual/'vllm';work.mkdir(exist_ok=True)
    with (qual/'compare.log').open('w') as f:
        subprocess.run([sys.executable,'-m','ouro_depth.vllm_latent.compare','--model',model,'--student',str(student),
            '--out',str(work),'--ref',str(qual/'hf/hf_reference.json'),'--max-new','64','--max-model-len','10240',
            '--logprobs-k','4096','--backend','TRITON_ATTN','--compile-config',FDO,'--max-num-seqs','8',
            '--engine-log',str(work/'engine.log')],env=inference_env(root,work,0),stdout=f,stderr=subprocess.STDOUT,check=True)
    hf(['-m','ouro_depth.latent.qualify_vllm_math','--model',model,'--student',str(student),
        '--compare',str(work/'compare.json'),'--output',str(qual/'qualified.json'),
        '--max-kl',os.environ.get('S6_SERVING_MAX_KL','0.05')],'qualification.log')
    print('RANK_SERVING_QUALIFIED',flush=True)
    resume=resume_source or 8
    for step in targets():
        train(step,resume);student=padded(step)
        evaluate(root,model,str(student),data,out/'math500'/f'step-{step:06d}')
        resume=step
    print('STAGE1_MATH600_COMPLETE',flush=True)

if __name__=='__main__':run()
