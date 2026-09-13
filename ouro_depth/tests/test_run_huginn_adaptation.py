"""Only new adaptation orchestration; fake optimizer/model, no torch/GPU.

The existing actual tiny-core accumulation/stochastic-resume proof is reused.
These tests target endpoint order, RNG restoration, final-only readiness and
explicit-run refusal. Saved-evaluation score validation uses synthetic rows.
"""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ouro_depth import run_huginn_adaptation as runner
from ouro_depth.v3_eval_binding import _recompute


class Model:
    def __init__(self):self.value=0.;self.training=True
    def train(self):self.training=True


class FakeCore:
    def __init__(self):self.saved={};self.saves=[];self.draws=[];self.bad_first=False
    def new_state(self,context):return {'update':0,'cursor':0,'examples':0,'forward_token_rounds':0,'gradient_token_rounds':0}
    def get_rng_state(self,context):return random.getstate()
    def set_rng_state(self,state,context):random.setstate(state)
    def train_update(self,model,opt,context,state):
        value=random.random();model.value+=value;self.draws.append(value)
        state['update']+=1;state['cursor']+=1;state['examples']+=16
        state['forward_token_rounds']+=16*256*32;state['gradient_token_rounds']+=16*256*8
        return {'update':state['update'],'loss':value,'missing_grad_count':0,'gradient_tensors':75,
            'gradient_elements':3564976800,'global_grad_norm_before_clip':1. if state['update']==1 else 0.,
            'gradient_components':{name:{'finite':True,'norm_l2':0. if self.bad_first else 1.} for name in ('core_block','adapter','prelude','coda')}}
    def save_checkpoint(self,path,model,opt,context,state):
        path.mkdir();runner.write(path/'complete.json',{'update':state['update']})
        self.saved[str(path)]=(copy.deepcopy(state),model.value,random.getstate());self.saves.append(state['update']);return path
    def load_checkpoint(self,path,model,opt,context):
        state,model.value,rng=copy.deepcopy(self.saved[str(path)]);random.setstate(rng);return state


class Controller(unittest.TestCase):
    def runtime(self,core,run,fail_at=None):
        model=Model();evaluations=[];context=SimpleNamespace(identity={'scope':'synthetic_controller_only'})
        answers=list(range(100,108))
        rows=[{'id':f'synthetic-{d}-{i}','difficulty':d,'family':'pointer_chasing','answer':'ABCDEFGH'[i%8]} for d in (1,2) for i in range(128)]
        calibration=runner.module(Path(runner.__file__).parents[1]/'diagnostics/huginn-calibration/run_calibration.py','stub_calibration_validator')
        def evaluate(prefix):
            u=int(prefix.name.removeprefix('dev-'));evaluations.append(u);model.training=False;random.random()
            if u==fail_at:raise RuntimeError('Synthetic interruption during evaluation')
            values=[]
            for row in rows:
                scores={}
                for d in (32,64):
                    good=not (u==256 and row['difficulty']==2 and d==64)
                    choice=row['answer'] if good else ('B' if row['answer']=='A' else 'A')
                    token=answers['ABCDEFGH'.index(row['answer'])] if good else 10
                    scores[str(d)]={'correct':good,'choice_correct':good,'choice':choice,'prediction_token':token,
                        'choice_tied':False,'choice_tie_aware_correct':float(good),'nll':1.,'choice_nll':1.,'answer_mass':.5}
                values.append({**row,'scores':scores})
            summary={'count':256,'depths':[32,64],'evaluator_version':2,'choice_tie_break':'ascending_token_id',
                'metrics':_recompute(values,['32','64'])}
            runner.write(str(prefix)+'.json',summary)
            Path(str(prefix)+'.predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))
            return summary
        runtime=SimpleNamespace(core=core,model=model,optimizer=object(),context=context,dev_rows=rows,answer_ids=answers,calibration=calibration,
            evaluate=evaluate,seed=lambda:random.seed(19872),samples=lambda:object(),
            displacement=lambda samples:[{'finite':True,'changed_elements':1}],rollback=lambda *args:None)
        return runtime,evaluations

    def test_registered_endpoints_rng_resume_and_final_failure_remain_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();full=root/'full';full.mkdir();core=FakeCore();runtime,evaluations=self.runtime(core,full)
            continuous=runner.training_loop(runtime,full);final_rng=random.getstate();final_model=runtime.model.value
            self.assertEqual(evaluations,[128,256]);self.assertEqual(core.saves,[128,256])
            self.assertTrue(runner.read(full/'endpoint-128.json')['readiness']['ready'])
            self.assertFalse(runner.read(full/'endpoint-128.json')['selection_allowed'])
            self.assertTrue(continuous['training_complete']);self.assertFalse(continuous['readiness']['ready'])
            self.assertEqual(continuous['phase'],'completed');self.assertEqual(continuous['state']['examples'],4096)
            interrupted=root/'interrupted';interrupted.mkdir();other=FakeCore();partial,calls=self.runtime(other,interrupted,fail_at=128)
            with self.assertRaises(RuntimeError):runner.training_loop(partial,interrupted)
            self.assertTrue(partial.model.training)
            cp=interrupted/'checkpoint-128';saved_rng=other.saved[str(cp)][2]
            self.assertEqual(random.getstate(),saved_rng)
            self.assertFalse((interrupted/'completed.json').exists())
            resumed,resumed_evals=self.runtime(other,interrupted);random.seed(1);random.random()
            result=runner.training_loop(resumed,interrupted,cp)
            self.assertEqual(resumed_evals,[128,256]);self.assertEqual(other.saves,[128,256])
            self.assertEqual(other.draws,core.draws);self.assertEqual(resumed.model.value,final_model)
            self.assertEqual(random.getstate(),final_rng);self.assertEqual(result['state'],continuous['state'])
            # A committed endpoint is reused; changed output never silently passes.
            before=random.getstate();runner.endpoint(resumed,interrupted,result['state'],interrupted/'checkpoint-256')
            self.assertEqual(resumed_evals,[128,256]);self.assertEqual(random.getstate(),before)
            receipt_path=interrupted/'endpoint-256.json';original=runner.read(receipt_path);changed=copy.deepcopy(original)
            changed['readiness']['ready']=True;runner.write(receipt_path,changed)
            with self.assertRaises(ValueError):runner.endpoint(resumed,interrupted,result['state'],interrupted/'checkpoint-256')
            runner.write(receipt_path,original)
            with (interrupted/'dev-256.predictions.jsonl').open('a') as f:f.write('{}\n')
            with self.assertRaises(ValueError):runner.endpoint(resumed,interrupted,result['state'],interrupted/'checkpoint-256')

    def test_raw_token_only_corruption_rejected_for_new_and_committed_endpoints(self):
        def corrupt(prefix):
            path=Path(str(prefix)+'.predictions.jsonl')
            values=[json.loads(line) for line in path.read_text().splitlines()]
            # Leave every correctness/choice flag and the summary unchanged.
            values[0]['scores']['32']['prediction_token']=10
            path.write_text(''.join(json.dumps(row)+'\n' for row in values))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fresh=root/'fresh';fresh.mkdir();runtime,_=self.runtime(FakeCore(),fresh)
            evaluate=runtime.evaluate
            def corrupted_evaluate(prefix):
                summary=evaluate(prefix);corrupt(prefix);return summary
            runtime.evaluate=corrupted_evaluate
            with self.assertRaisesRegex(ValueError,'Raw prediction token'):
                runner.endpoint(runtime,fresh,{'update':128},fresh/'checkpoint-128')
            self.assertFalse((fresh/'endpoint-128.json').exists())
            committed=root/'committed';committed.mkdir();runtime,_=self.runtime(FakeCore(),committed)
            receipt=runner.endpoint(runtime,committed,{'update':128},committed/'checkpoint-128')
            self.assertEqual(receipt['binding']['raw_prediction_token_checks'],512)
            self.assertTrue(receipt['binding']['raw_correctness_matches_answer_tokens'])
            self.assertEqual(receipt['binding']['answer_token_ids'],dict(zip('ABCDEFGH',runtime.answer_ids)))
            corrupt(committed/'dev-128')
            # Update only the file digest so the semantic check is reached on resume.
            receipt['predictions_sha256']=runner.digest(committed/'dev-128.predictions.jsonl')
            runner.write(committed/'endpoint-128.json',receipt)
            with self.assertRaisesRegex(ValueError,'Raw prediction token'):
                runner.endpoint(runtime,committed,{'update':128},committed/'checkpoint-128')

    def test_first_gradient_acceptance_and_early_run_gpu_refusals(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{},clear=False):
            root=Path(folder).resolve();(root/'artifacts').mkdir();(root/'runs').mkdir()
            output=root/'bad';output.mkdir();core=FakeCore();core.bad_first=True;runtime,_=self.runtime(core,output)
            with self.assertRaises(ValueError):runner.training_loop(runtime,output)
            self.assertEqual(core.saves,[])
            self.assertFalse(runner.read(output/'first-update-verification.json')['passed'])
            run=root/'runs'/runner.NAME;run.mkdir()
            os.environ['CUDA_VISIBLE_DEVICES']=runner.GPU_UUID
            with patch.object(runner,'module',side_effect=AssertionError('No prerequisite/model/GPU work for refusal')):
                with self.assertRaises(FileExistsError):runner.execute(root)
                with self.assertRaises(ValueError):runner.execute(root,root/'foreign/checkpoint-128')
                runner.write(run/'completed.json',{'training_complete':True})
                with self.assertRaises(FileExistsError):runner.execute(root,run/'checkpoint-256')
            os.environ['CUDA_VISIBLE_DEVICES']='5'
            with self.assertRaises(ValueError):runner.execute(root)


if __name__=='__main__':unittest.main()
