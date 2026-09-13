"""Frozen raw-Huginn d1/d2 calibration only; no training or sealed-test access."""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_data(root):
    directory = root / 'diagnostics/huginn-calibration'
    manifest = read_json(directory / 'manifest.json')
    expected = {'scope': 'cold_start_task_format_calibration_only',
        'repository': 'tomg-group-umd/huginn-0125',
        'revision': 'bb6621b65e90b6a4b9b29ef88dc83866d450470c',
        'source_data': 'data/v3-pointer/dev.jsonl',
        'data_file': 'diagnostics/huginn-calibration/calibration-dev.jsonl',
        'count': 256, 'depths': [32, 64], 'eval_seed': 18931,
        'microbatch': 2, 'padding_width': 256,
        'test_data_read': False, 'training_performed': False,
        'thresholds': {'d1': 0.95, 'd2': 0.8, 'both_depths_required': True,
                       'metric': 'unrestricted_next_token_accuracy'}}
    if any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k,v in expected.items()):
        raise ValueError('Calibration manifest differs from the declared protocol')
    raw = (root / manifest['data_file']).read_bytes()
    source = (root / manifest['source_data']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['data_sha256'] or hashlib.sha256(source).hexdigest() != manifest['source_sha256']:
        raise ValueError('Calibration/source DEV identity changed')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    original = [json.loads(line) for line in source.splitlines() if line.strip()]
    if rows != [r for r in original if r['difficulty'] in (1, 2)]:
        raise ValueError('Calibration is not the complete ordered original d1/d2 subset')
    if len(rows) != 256 or len({r['id'] for r in rows}) != 256:
        raise ValueError('Expected256unique calibration questions')
    if Counter((r['difficulty'],r['answer']) for r in rows) != Counter({(d,a):16 for d in (1,2) for a in 'ABCDEFGH'}):
        raise ValueError('Calibration answer/hop balance changed')
    return manifest, rows


def validate_result(prefix, rows):
    # Reuse generic score/summary recomputation, not the Ouro-only4/8 loader.
    from ouro_depth.v3_eval_binding import _check_scores, _recompute, _compare
    summary = read_json(str(prefix)+'.json')
    predictions = [json.loads(l) for l in Path(str(prefix)+'.predictions.jsonl').read_text().splitlines() if l.strip()]
    if summary.get('evaluator_version') != 2 or summary.get('choice_tie_break') != 'ascending_token_id' or summary.get('depths') != [32,64] or summary.get('count') != 256:
        raise ValueError('Unexpected calibration evaluator result')
    if len(predictions) != len(rows):
        raise ValueError('Incomplete calibration results')
    for pred, truth in zip(predictions, rows):
        if any(type(pred.get(k)) is not type(truth[k]) or pred[k] != truth[k] for k in ('id','family','difficulty','answer')):
            raise ValueError('Prediction membership/metadata mismatch')
        _check_scores(pred, ['32','64'])
    _compare(summary['metrics'], _recompute(predictions,['32','64']), 'metrics')
    return {'all_256_ordered_IDs_and_metadata_match': True,
            'all_score_derived_metrics_match': True, 'test_data_read': False}


class EvaluationBridge:
    """Expose the existing scorer interface without changing Ouro code."""
    def __init__(self, model, encoded, pad_id, eval_seed, width, progress=None):
        self.model, self.pad_id, self.eval_seed, self.width = model, pad_id, eval_seed, width
        self.progress, self.completed = progress, 0
        self.lookup = {tuple(item['ids']): item['row']['id'] for item in encoded}
        if len(self.lookup) != len(encoded):
            raise ValueError('Prompts must have unique token sequences for ID binding')
    @property
    def training(self):
        return self.model.training
    def train(self, mode=True):
        self.model.train(mode)
        return self
    def eval(self):
        return self.train(False)
    def __call__(self, ids, mask, *, depths):
        import torch
        from ouro_depth.huginn_evaluation import paired_depth_logits
        lengths = mask.sum(dim=1).tolist()
        example_ids = [self.lookup[tuple(row[:n].tolist())] for row,n in zip(ids,lengths)]
        if ids.shape[1] > self.width:
            raise ValueError('Scorer input exceeds frozen padding width')
        amount = self.width - ids.shape[1]
        if amount:
            ids = torch.nn.functional.pad(ids,(0,amount),value=self.pad_id)
            mask = torch.nn.functional.pad(mask,(0,amount),value=0)
        result = paired_depth_logits(self.model, ids, mask, example_ids=example_ids,
            depths=depths, eval_seed=self.eval_seed, pad_token_id=self.pad_id)
        self.completed += len(example_ids)
        if self.progress is not None and self.completed % 32 == 0:
            self.progress(self.completed)
        return result


def execute(root, label, *, native_bos=False):
    root = root.resolve()
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', label) is None:
        raise ValueError('Use one safe attempt label')
    directory = root/'diagnostics/huginn-calibration'
    destination = directory/label
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if 'torch' in sys.modules:
        raise RuntimeError('Run standalone before importing torch')
    sys.path.insert(0,str(root/'diagnostics/huginn-engineering'))
    import smoke_gpu as smoke
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,smoke.GPU_UUID):
        raise ValueError('Only allocated GPU4 is permitted')
    os.environ.update(CUDA_VISIBLE_DEVICES=smoke.GPU_UUID,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with (root/'diagnostics/huginn-engineering/smoke.lock').open('a') as lock, contextlib.ExitStack() as stack:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        prerequisites = smoke._prerequisites(root)
        cpu = read_json(root/'artifacts/huginn-paired-eval-cpu-verification.json')
        # The completed CPU test was launched from root and recorded a relative
        # model directory. Resolve that existing receipt without rewriting it.
        cpu_source = Path(cpu.get('source_directory',''))
        cpu_source = (cpu_source if cpu_source.is_absolute() else root/cpu_source).resolve()
        if not (cpu.get('scope') == 'official_code_tiny_random_CPU_paired_evaluation_synthetic_tokens_only'
                and cpu.get('passed') is True and cpu.get('tests_run') == 1
                and cpu.get('failures') == 0 and cpu.get('errors') == 0
                and cpu.get('gpu_used') is False and cpu.get('research_data_used') is False
                and cpu.get('torch') == prerequisites['cpu']['torch']
                and cpu_source == (root/'huginn_model').resolve()
                and cpu.get('model_revision') == smoke.REVISION
                and cpu.get('latent_seed_scheme') == 'huginn-paired-latent-v1'):
            raise ValueError('Actual paired-evaluation CPU verification must pass for this pinned model')
        manifest,rows = checked_data(root)
        interface = None
        if native_bos:
            interface = read_json(directory/'native-interface.json')
            if (interface.get('scope') != 'native_BOS_task_format_followup_only'
                    or interface.get('only_input_change') != 'prepend_BOS_65504_via_official_tokenizer'
                    or interface.get('parent_data_sha256') != manifest['data_sha256']
                    or interface.get('thresholds') != manifest['thresholds']
                    or interface.get('output_rule') != 'unrestricted_next_token_space_prefixed_answer'
                    or interface.get('training_performed') is not False
                    or interface.get('test_data_read') is not False):
                raise ValueError('Native interface follow-up differs from its declaration')
        destination.mkdir()
        source = destination/'source'
        shutil.copytree(root/'ouro_depth',source/'ouro_depth',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        shutil.copy2(Path(__file__),destination/'run_calibration.py')
        for filename in ('manifest.json','calibration-dev.jsonl'):
            shutil.copy2(directory/filename,destination/filename)
        if native_bos:
            shutil.copy2(directory/'native-interface.json',destination/'native-interface.json')
            shutil.copy2(root/'ouro_depth/HUGINN-NATIVE-INTERFACE.md',destination/'HUGINN-NATIVE-INTERFACE.md')
        sys.path.insert(0,str(source))
        logfile=stack.enter_context((destination/'process.log').open('x',buffering=1))
        stack.enter_context(contextlib.redirect_stdout(logfile));stack.enter_context(contextlib.redirect_stderr(logfile))
        status_path=destination/'status.json'
        state={'phase':'preflight','scope':manifest['scope'],'pid':os.getpid(),'manifest':manifest,
            'source':str(source),'gpu_uuid':smoke.GPU_UUID,'model_saved':False,'training_performed':False,
            'input_interface': 'native_BOS' if native_bos else 'no_special_tokens',
            'interface_followup': interface,
            'test_data_read':False,'prerequisites':{'official_import':prerequisites['import_source'],
                'paired_eval_CPU':cpu},'questions_scored':0}
        started=time.monotonic();torch=None
        try:
            from ouro_depth.run_diagnostics import assert_gpu_unused
            description=assert_gpu_unused(smoke.GPU)
            if description.split(',')[1].strip()!=smoke.GPU_UUID:raise RuntimeError('GPU identity changed')
            state['gpu_description']=description;smoke._record(status_path,state)
            import torch
            from transformers import AutoTokenizer
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from ouro_depth.train import encode_rows,evaluate
            from types import SimpleNamespace
            if torch.__version__!=cpu['torch'] or torch.cuda.device_count()!=1:raise RuntimeError('CUDA/runtime mismatch')
            torch.cuda.set_device(0);torch.set_num_threads(8)
            model_dir=root/'huginn_model'
            cls=get_class_from_dynamic_module('raven_modeling_minimal.RavenForCausalLM',str(model_dir),local_files_only=True)
            state['phase']='loading_model';smoke._record(status_path,state)
            model,loading=cls.from_pretrained(str(model_dir),torch_dtype=torch.float32,local_files_only=True,
                use_safetensors=True,low_cpu_mem_usage=True,output_loading_info=True)
            if any(loading.values()):raise RuntimeError(f'Checkpoint loading discrepancy:{loading}')
            model=model.to('cuda:0').eval();model.requires_grad_(False)
            tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
            if native_bos:
                from ouro_depth.huginn_tokenization import encode_native_rows
                encoded,answers=encode_native_rows(rows,tokenizer,manifest['padding_width'])
            else:
                encoded,answers=encode_rows(rows,tokenizer,manifest['padding_width'])
            def progress(count):
                state['questions_with_both_depth_logits'] = count
                smoke._record(status_path,state)
            bridge=EvaluationBridge(model,encoded,tokenizer.pad_token_id,manifest['eval_seed'],manifest['padding_width'],progress)
            args=SimpleNamespace(device='cuda:0',pad_id=tokenizer.pad_token_id,eval_batch=manifest['microbatch'])
            state.update(phase='evaluating',loading_info=loading,parameters=smoke._parameters(model),
                answer_token_ids=answers,prompt_max_length=max(len(x['ids']) for x in encoded))
            smoke._record(status_path,state)
            prefix=destination/'pretrained-dev'
            result=evaluate(bridge,encoded,answers,args,manifest['depths'],str(prefix))
            binding=validate_result(prefix,rows)
            accuracy={str(d):{str(t):result['metrics'][f'pointer_chasing/d{d}']['by_depth'][str(t)]['accuracy'] for t in manifest['depths']} for d in (1,2)}
            passes=all(accuracy[str(d)][str(t)]>=manifest['thresholds'][f'd{d}'] for d in (1,2) for t in manifest['depths'])
            state.update(phase='completed',questions_scored=len(rows),binding=binding,accuracy=accuracy,
                calibration_passes=passes,skip_shared_warmup_supported=passes,
                elapsed_seconds=time.monotonic()-started,memory=smoke._memory(torch),
                interpretation='Task-format calibration only; no hard-task benefit, training-method or adaptive-halting conclusion.')
            with (destination/'completed.json').open('x') as handle:json.dump(state,handle,indent=2,allow_nan=False)
            smoke._record(status_path,state)
            return state
        except BaseException as error:
            traceback.print_exc();state.update(phase='failed',error=repr(error),error_type=type(error).__name__,elapsed_seconds=time.monotonic()-started)
            if torch is not None and torch.cuda.is_initialized():state['memory']=smoke._memory(torch)
            smoke._record(status_path,state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--attempt-label',required=True)
    parser.add_argument('--native-bos',action='store_true')
    args=parser.parse_args();execute(args.root,args.attempt_label,native_bos=args.native_bos)
