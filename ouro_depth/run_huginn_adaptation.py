"""Explicit fixed256-step common Huginn adaptation, never the deeper-training test.

python -m ouro_depth.run_huginn_adaptation --root STUDY_ROOT
Optional --resume must name this run's latest committed checkpoint. Existing
attempts are never restarted implicitly. GPU5 only; no sealed-test access.
"""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import fcntl
from types import SimpleNamespace

NAME='huginn-shared-adaptation-s19872'
REVISION='bb6621b65e90b6a4b9b29ef88dc83866d450470c'
GPU_UUID='GPU-c43da7d3-3e7c-0e84-b609-5519eed23ae3'
ENDPOINTS=(128,256)
SOURCE_FILES=('run_huginn_adaptation.py','huginn_training.py','huginn_evaluation.py','huginn_adapter.py',
    'huginn_tokenization.py','train.py','model.py','curriculum.py','v3_plan.py','train_v3.py',
    'v3_eval_binding.py','compare_predictions.py','run_diagnostics.py',
    'vendor/configuration_ouro.py','vendor/modeling_ouro.py',
    'HUGINN-SHARED-ADAPTATION.md','HUGINN-NATIVE-INTERFACE.md')


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value

def prerequisites(root,smoke):
    imported=smoke._prerequisites(root)
    training=read(root/'artifacts/huginn-training-tiny-cpu-verification.json')
    paired=read(root/'artifacts/huginn-paired-eval-cpu-verification.json')
    capacity=read(root/'artifacts/huginn-accumulation-capacity-verification.json')
    native=read(root/'diagnostics/huginn-calibration/raw-native-bos/completed.json')
    audit=read(root/'artifacts/huginn-native-calibration-verification.json')
    probe=read(root/'diagnostics/v3-final-depth-dev/status.json')
    for receipt,n in ((training,3),(paired,1)):
        if (receipt.get('passed') is not True or receipt.get('tests_run')!=n or receipt.get('failures')!=0
                or receipt.get('errors')!=0 or receipt.get('gpu_used') is not False or receipt.get('research_data_used') is not False):
            raise ValueError('Existing CPU training/paired-evaluation verification must pass')
    if (capacity.get('unique_trainable_parameters')!=3564976800 or capacity.get('microbatch')!=2
            or capacity.get('sequence_length')!=256 or capacity.get('accumulation_steps')!=8
            or [(c['loops'],c['window'],c['optimizer_step_completed']) for c in capacity['cases']]!=[(32,8,True),(64,8,True)]):
        raise ValueError('Existing actual full-size accumulation capacity evidence is required')
    if (native.get('phase')!='completed' or native.get('input_interface')!='native_BOS'
            or native.get('calibration_passes') is not False or audit.get('calibration_passes') is not False
            or audit.get('input_interface')!='native_BOS' or audit.get('binding')!=native['binding']
            or native['manifest']['revision']!=REVISION or probe.get('phase')!='completed'):
        raise ValueError('Failed native calibration and completed registered Ouro probe required')
    return {'import':imported,'training_cpu':training,'paired_cpu':paired,'capacity':capacity,
            'native_completed':native,'native_audit':audit,'ouro_probe_completed':probe}


def prepared_data(root,calibration):
    directory=root/'data/huginn-shared-adaptation';manifest=read(directory/'manifest.json')
    expected={'rows':4096,'updates':256,'batch_size':16,'microbatch':2,'padding_width':256,
              'depth':32,'gradient_window':8,'selection_seed':19871,'training_seed':19872,
              'protocol':'ouro_depth/HUGINN-SHARED-ADAPTATION.md'}
    if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Adaptation declaration differs')
    rows=[json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines() if line.strip()]
    original=[json.loads(line) for line in (root/'data/v3-pointer/train.jsonl').read_text().splitlines() if line.strip()]
    plan=read(directory/'plan.json');dev_manifest,dev=calibration.checked_data(root)
    if (digest(directory/'train.jsonl')!=manifest['train_sha256'] or digest(directory/'plan.json')!=manifest['plan_sha256']
            or digest(root/'data/v3-pointer/train.jsonl')!=manifest['source_sha256']
            or rows!=[original[i] for i in manifest['original_train_indices']]
            or len(rows)!=4096 or len({r['id'] for r in rows})!=4096
            or Counter((r['difficulty'],r['answer']) for r in rows)!=Counter({(d,a):256 for d in (1,2) for a in 'ABCDEFGH'})):
        raise ValueError('Prepared4096 rows must be the unchanged balanced original subset')
    if ({r['id'] for r in rows}&{r['id'] for r in dev}
            or {r['metadata']['instance_key'] for r in rows}&{r['metadata']['instance_key'] for r in dev}):
        raise ValueError('Adaptation training overlaps calibration DEV')
    if len(plan)!=256:raise ValueError('Exactly256 planned updates required')
    seen=[];hops=[]
    for i,step in enumerate(plan):
        progress=i/256;rate=1e-5*(min(1.,max(.1,progress/.05))*(.1+.9*.5*(1+math.cos(math.pi*progress))))
        # Preserve the hashed plan's exact rates. The formula is a compatibility
        # check: platform libm cosine may differ by a few floating-point ULPs.
        if (set(step)!={'indices','depth','lr'} or step['depth']!=32 or len(step['indices'])!=16
                or any(type(j) is not int or not 0<=j<4096 for j in step['indices'])
                or type(step['lr']) not in (int,float) or not math.isclose(step['lr'],rate,rel_tol=1e-14,abs_tol=0.)):
            raise ValueError('Prepared batch/depth/LR differs from the fixed protocol')
        batch_hops={rows[j]['difficulty'] for j in step['indices']}
        if len(batch_hops)!=1:raise ValueError('Each effective batch must have one hop')
        hops.append(next(iter(batch_hops)));seen.extend(step['indices'])
    if sorted(seen)!=list(range(4096)) or any(set(hops[i:i+2])!={1,2} for i in range(0,256,2)):
        raise ValueError('Every example exactly once and one d1/d2 batch per pair required')
    return rows,plan,dev,{'manifest':manifest,'calibration_manifest':dev_manifest,
        'train_sha256':manifest['train_sha256'],'plan_sha256':manifest['plan_sha256'],'dev_sha256':dev_manifest['data_sha256']}


def source_identity(source):
    paths=[source/'ouro_depth'/name for name in SOURCE_FILES]
    paths+=list((source/'vendor').iterdir())+[source/'helpers/smoke_gpu.py',source/'helpers/run_calibration.py']
    files={str(p.relative_to(source)):digest(p) for p in paths if p.is_file()}
    return {'files':files,'fingerprint':fingerprint(files)}


def freeze(root,run,imported):
    source=run/'source';shutil.copytree(root/'ouro_depth',source/'ouro_depth',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    (source/'helpers').mkdir();(source/'vendor').mkdir()
    shutil.copy2(root/'diagnostics/huginn-engineering/smoke_gpu.py',source/'helpers/smoke_gpu.py')
    shutil.copy2(root/'diagnostics/huginn-calibration/run_calibration.py',source/'helpers/run_calibration.py')
    for name,item in imported['files'].items():
        if name.endswith('.safetensors'):continue
        origin=root/'huginn_model'/name;raw=origin.read_bytes()
        actual=hashlib.sha256(raw).hexdigest() if item['sha256'] else hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if actual!=(item['sha256'] or item['git_blob']):raise ValueError(f'Pinned small vendor file changed: {name}')
        shutil.copy2(origin,source/'vendor'/name)
    return source,source_identity(source)


def readiness(summary):
    if summary.get('count')!=256 or summary.get('depths')!=[32,64]:raise ValueError('Full fixed calibration exits required')
    accuracy={str(d):{str(r):summary['metrics'][f'pointer_chasing/d{d}']['by_depth'][str(r)]['accuracy'] for r in (32,64)} for d in (1,2)}
    passed=all(accuracy[str(d)][str(r)]>=floor for d,floor in ((1,.95),(2,.8)) for r in (32,64))
    return {'ready':passed,'raw_accuracy':accuracy,'thresholds':{'d1':.95,'d2':.8,'both_depths_required':True},
            'scope':'common simple-task initialization only; not deeper-training or hard-task success'}


def validate_predictions(runtime,prefix):
    """Bind raw correctness to the actual tokenizer's canonical answer IDs."""
    binding=runtime.calibration.validate_result(prefix,runtime.dev_rows)
    answers=runtime.answer_ids
    if len(answers)!=8 or any(type(token) is not int or token<0 for token in answers) or len(set(answers))!=8:
        raise ValueError('Eight distinct canonical answer token IDs required')
    answer_ids=dict(zip('ABCDEFGH',answers))
    predictions=[json.loads(line) for line in Path(str(prefix)+'.predictions.jsonl').read_text().splitlines()]
    checked=0
    for prediction,row in zip(predictions,runtime.dev_rows):
        for depth in ('32','64'):
            score=prediction['scores'][depth];token=score['prediction_token']
            if type(token) is not int or token<0 or score['correct']!=(token==answer_ids[row['answer']]):
                raise ValueError(f'Raw prediction token disagrees with correctness: {row["id"]} R{depth}')
            checked+=1
    if checked!=512:raise ValueError('Exactly 512 raw prediction token checks required')
    return {**binding,'raw_prediction_token_checks':checked,'answer_token_ids':answer_ids,
            'raw_correctness_matches_answer_tokens':True}


def endpoint(runtime,run,state,checkpoint):
    """Evaluate exact endpoints while restoring all training RNG even on failure."""
    core,model,context=runtime.core,runtime.model,runtime.context;u=state['update'];prefix=run/f'dev-{u}'
    receipt_path=run/f'endpoint-{u}.json'
    expected={'update':u,'checkpoint':str(checkpoint),'run_identity_fingerprint':fingerprint(context.identity)}
    if receipt_path.exists():
        receipt=read(receipt_path)
        if (any(receipt.get(k)!=v for k,v in expected.items()) or receipt['predictions_sha256']!=digest(str(prefix)+'.predictions.jsonl')
                or receipt['summary_sha256']!=digest(str(prefix)+'.json')):raise ValueError('Saved evaluation receipt changed')
        binding=validate_predictions(runtime,prefix)
        if (receipt.get('binding')!=binding or receipt.get('readiness')!=readiness(read(str(prefix)+'.json'))
                or receipt.get('selection_allowed')!=(u==256) or receipt.get('training_rng_restored') is not True):
            raise ValueError('Saved endpoint readiness or scope differs from its scores')
        return receipt
    if any(Path(str(prefix)+s).exists() for s in ('.json','.predictions.jsonl')):raise FileExistsError('Uncommitted evaluation output exists; inspect it')
    rng=core.get_rng_state(context)
    try:
        summary=runtime.evaluate(prefix)
        binding=validate_predictions(runtime,prefix)
    finally:core.set_rng_state(rng,context);model.train()
    receipt={**expected,'binding':binding,'summary_sha256':digest(str(prefix)+'.json'),
             'predictions_sha256':digest(str(prefix)+'.predictions.jsonl'),'readiness':readiness(summary),
             'selection_allowed':u==256,'training_rng_restored':True}
    write(receipt_path,receipt);return receipt


def training_loop(runtime,run,resume=None):
    core,model,optimizer,context=runtime.core,runtime.model,runtime.optimizer,runtime.context
    state=core.load_checkpoint(resume,model,optimizer,context) if resume else core.new_state(context)
    if resume:
        latest=read(run/'latest.json')
        if latest!={'checkpoint':str(resume),'state':state}:raise ValueError('Latest and loaded state differ')
        runtime.rollback(run,state['update'])
    else:runtime.seed()
    metrics=run/'metrics.jsonl'
    def log(record):
        with metrics.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    log({'event':'start','update':state['update'],'resume':str(resume) if resume else None})
    saved=Path(resume) if resume else None;last_eval=None
    if state['update'] in ENDPOINTS:last_eval=endpoint(runtime,run,state,saved)
    while state['update']<256:
        samples=runtime.samples() if state['update']==0 else None
        try:record=core.train_update(model,optimizer,context,state)
        except BaseException:
            write(run/'failed-update.json',state);raise
        if state['update']==1:
            evidence=runtime.displacement(samples)
            passed=not (record['missing_grad_count']!=0 or record['gradient_tensors']!=75 or record['gradient_elements']!=3564976800
                    or not math.isfinite(record['global_grad_norm_before_clip']) or record['global_grad_norm_before_clip']<=0
                    or any(not g['finite'] or g['norm_l2']<=0 for g in record['gradient_components'].values())
                    or not evidence or any(not e['finite'] for e in evidence) or not sum(e['changed_elements'] for e in evidence))
            write(run/'first-update-verification.json',{'update':1,'record':record,'core_parameter_samples':evidence,
                'passed':passed,'scope':'Real task optimization path only, not task accuracy or depth benefit'})
            if not passed:raise ValueError('First real task update lacks finite full/core gradient or parameter-displacement evidence')
        log({'event':'update','update':state['update'],**record,'counters':state.copy()})
        if state['update'] in ENDPOINTS:
            saved=core.save_checkpoint(run/f"checkpoint-{state['update']}",model,optimizer,context,state)
            write(run/'latest.json',{'checkpoint':str(saved),'state':state})
            last_eval=endpoint(runtime,run,state,saved)
            log({'event':'dev','update':state['update'],'endpoint':last_eval})
    if state['cursor']!=256 or state['examples']!=4096 or state['forward_token_rounds']!=4096*256*32 or state['gradient_token_rounds']!=4096*256*8:
        raise ValueError('Final consumed budget differs from the fixed plan')
    if not (run/'first-update-verification.json').is_file():raise ValueError('Missing first actual update evidence')
    result={'phase':'completed','termination':'fixed_plan','checkpoint':str(saved),'state':state,
            'readiness':last_eval['readiness'],'final_endpoint':last_eval,'only_final256_selectable':True,
            'training_complete':True,'deeper_training_success_claimed':False,'test_scored':False}
    write(run/'completed.json',result);return result


def load_runtime(root,source,rows,plan,dev,run_identity,status):
    # The entrypoint itself matches the frozen copy; all subsequently imported
    # package dependencies and the official vendor class resolve to that copy.
    import ouro_depth
    ouro_depth.__path__=[str(source/'ouro_depth')]
    import torch,numpy as np,transformers
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from ouro_depth import huginn_training as core
    from ouro_depth.huginn_tokenization import encode_native_rows
    from ouro_depth.train import evaluate
    from ouro_depth.train_v3 import _rollback_uncommitted_metrics
    for name in ('huginn_training','huginn_evaluation','huginn_adapter','huginn_tokenization','train','model','train_v3'):
        __import__('ouro_depth.'+name)
        if Path(sys.modules['ouro_depth.'+name].__file__).resolve()!=source/'ouro_depth'/f'{name}.py':raise ValueError('Dependency escaped frozen source')
    smoke=module(source/'helpers/smoke_gpu.py','adaptation_smoke_frozen')
    calibration=module(source/'helpers/run_calibration.py','adaptation_calibration_frozen')
    if torch.__version__!='2.11.0+cu130' or transformers.__version__!='4.56.2' or torch.cuda.device_count()!=1:raise ValueError('Pinned runtime/single GPU5 required')
    torch.cuda.set_device(0);torch.set_num_threads(8)
    cls=get_class_from_dynamic_module('raven_modeling_minimal.RavenForCausalLM',str(source/'vendor'),local_files_only=True)
    config=cls.config_class.from_pretrained(str(source/'vendor'),local_files_only=True)
    model,loading=cls.from_pretrained(str(root/'huginn_model'),config=config,torch_dtype=torch.float32,
        local_files_only=True,use_safetensors=True,low_cpu_mem_usage=True,output_loading_info=True)
    if any(loading.values()):raise ValueError(f'Official pretrained loading mismatch: {loading}')
    model=model.to('cuda:0');tokenizer=AutoTokenizer.from_pretrained(str(source/'vendor'),local_files_only=True)
    encoded,answers=encode_native_rows(rows,tokenizer,256);encoded_dev,dev_answers=encode_native_rows(dev,tokenizer,256)
    if answers!=dev_answers:raise ValueError('Training and calibration answer IDs differ')
    context=core.prepare_training(model,encoded,plan,run_identity,microbatch_size=2,padding_width=256,gradient_window=8,lr=1e-5,weight_decay=.01,clip=1.)
    parameters=smoke._parameters(model)
    if (parameters['unique_trainable_parameters']!=3564976800 or parameters['parameter_tensors']!=75
            or parameters['parameters_by_dtype']!={'torch.float32':3564976800}
            or not parameters['embedding_head_same_parameter'] or not parameters['embedding_head_same_storage']):
        raise ValueError('Official all-parameter FP32/tied-head boundary differs')
    optimizer=core.make_optimizer(model,context)
    if optimizer.state:raise ValueError('Fresh optimizer must start without Adam moments')
    status.update(loading_info=loading,parameters=parameters,optimizer_empty_before_optional_restore=True,answer_ids=answers,
                  training_context_identity=context.identity,autocast_dtype='bfloat16')
    def score(prefix):
        bridge=calibration.EvaluationBridge(model,encoded_dev,tokenizer.pad_token_id,18931,256)
        args=SimpleNamespace(device='cuda:0',pad_id=tokenizer.pad_token_id,eval_batch=2)
        return evaluate(bridge,encoded_dev,answers,args,[32,64],prefix)
    def seed():random.seed(19872);np.random.seed(19872);torch.manual_seed(19872)
    return SimpleNamespace(core=core,model=model,optimizer=optimizer,context=context,calibration=calibration,
        dev_rows=dev,answer_ids=answers,evaluate=score,rollback=_rollback_uncommitted_metrics,seed=seed,
        samples=lambda:smoke._core_samples(torch,model),displacement=smoke._updated_samples)


def execute(root,resume=None):
    root=Path(root).resolve();run=root/'runs'/NAME
    if 'torch' in sys.modules:raise RuntimeError('Use a standalone process before importing torch')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,GPU_UUID):raise ValueError('Only exact allocated GPU5 UUID is permitted')
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU_UUID,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with (root/'artifacts/huginn-shared-adaptation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'completed.json').exists():raise FileExistsError('Completed adaptation cannot be restarted')
        if run.exists() and not resume:raise FileExistsError('Existing attempt requires an explicit own-checkpoint resume')
        if resume:
            resume=Path(resume).resolve()
            if resume.parent!=run or resume.name not in ('checkpoint-128','checkpoint-256') or not (resume/'complete.json').is_file():
                raise ValueError('Resume must name an own committed registered checkpoint')
            if Path(read(run/'latest.json')['checkpoint']).resolve()!=resume:raise ValueError('Resume must use latest committed checkpoint')
        smoke=module(root/'diagnostics/huginn-engineering/smoke_gpu.py','adaptation_smoke_preflight')
        calibration=module(root/'diagnostics/huginn-calibration/run_calibration.py','adaptation_calibration_preflight')
        adoption=read(root/'artifacts/huginn-adaptation-adoption.json')
        if not isinstance(adoption,dict) or not adoption:raise ValueError('Explicit adoption record is required')
        prerequisites_value=prerequisites(root,smoke);rows,plan,dev,data=prepared_data(root,calibration)
        if prerequisites_value['native_completed']['manifest']!=data['calibration_manifest']:raise ValueError('Native-failure calibration identity changed')
        if not resume:
            run.mkdir();source,source_receipt=freeze(root,run,prerequisites_value['import']['import_source'])
            write(run/'plan.json',plan)
        else:
            source=run/'source';source_receipt=source_identity(source)
            if read(run/'plan.json')!=plan:raise ValueError('Frozen adaptation plan changed')
        if digest(Path(__file__))!=source_receipt['files']['ouro_depth/run_huginn_adaptation.py']:
            raise ValueError('Run the frozen entrypoint for resume; executing code differs')
        identity={'protocol':'common_huginn_task_adaptation','run':str(run),'model_directory':str(root/'huginn_model'),
            'model_revision':REVISION,'source':source_receipt,'data':data,'training_seed':19872,
            'eval_seed':18931,'endpoints':list(ENDPOINTS),'adoption':adoption,'prerequisites_fingerprint':fingerprint(prerequisites_value)}
        if resume and read(run/'identity.json')!=identity:raise ValueError('Run source/data/import/prerequisite identity changed')
        if not resume:write(run/'identity.json',identity)
        state={'phase':'preflight','pid':os.getpid(),'run':str(run),'source':str(source),'identity':identity,
            'gpu':5,'gpu_uuid':GPU_UUID,'resume':str(resume) if resume else None,'test_scored':False,
            'scope':'fixed common adaptation only; not the deeper-training experiment'}
        status=run/'status.json'
        if resume:shutil.copy2(status,run/f'status-before-resume-{time.time_ns()}.json')
        write(status,state)
        with (run/f"process-{'resume-'+str(time.time_ns()) if resume else 'initial'}.log").open('x',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                gpu_helper=module(source/'ouro_depth/run_diagnostics.py','adaptation_gpu_frozen')
                description=gpu_helper.assert_gpu_unused(5)
                if description.split(',')[1].strip()!=GPU_UUID:raise ValueError('Allocated GPU5 UUID changed')
                state.update(phase='loading',gpu_description=description);write(status,state)
                runtime=load_runtime(root,source,rows,plan,dev,identity,state)
                state['phase']='training';write(status,state)
                result=training_loop(runtime,run,resume)
                state.update(phase='completed',result=result);write(status,state);return result
            except BaseException as error:
                import traceback
                traceback.print_exc();state.update(phase='failed',error=repr(error));write(status,state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--resume',type=Path);args=parser.parse_args();execute(args.root,args.resume)
