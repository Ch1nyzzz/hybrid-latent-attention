"""New candidate loss dispatch plus actual tiny CPU checkpoint/next-update test.

Reuse V4 fixture utilities, not its tests. Evaluations emit synthetic score
fixtures; training is actual tiny Ouro with dropout and full checkpointed BPTT.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ouro_depth import train_extension as t
from ouro_depth.tests import test_train_v4 as fixtures
from ouro_depth.model import OuroDepthModel
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM
from ouro_depth.v3_eval_binding import _recompute

RESULTS={}


class ExtensionCPU(unittest.TestCase):
    equal=fixtures.V4TrainerCPU.equal

    def test_new_loss_resume_and_registered_control_endpoint_continuation(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.ExitStack() as stack:
            root=Path(temporary);old_threads=torch.get_num_threads();torch.set_num_threads(1)
            stack.callback(torch.set_num_threads,old_threads)
            stack.enter_context(patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA in CPU validation')))
            stack.enter_context(patch.object(torch.cuda,'get_rng_state_all',side_effect=AssertionError('No CUDA RNG read')))
            stack.enter_context(patch.object(torch.cuda,'set_rng_state_all',side_effect=AssertionError('No CUDA RNG restore')))
            stack.enter_context(patch.object(torch.cuda,'manual_seed_all'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            data=root/'data';data.mkdir()
            rows=[{'id':f'synthetic-{i}','family':'pointer_chasing','difficulty':(1,2,3,4,6,8)[i//6],
                   'prompt':f'example{i}:','answer':'ABCDEFGH'[i%8]} for i in range(36)]
            for name in ('train.jsonl','dev.jsonl'):(data/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            fixtures.seed(482)
            config=OuroConfig(vocab_size=80,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32,attention_dropout=.1,
                pad_token_id=0,bos_token_id=1,eos_token_id=2,use_cache=False,tie_word_embeddings=False)
            config._attn_implementation='sdpa';base=OuroForCausalLM(config).float()
            def fresh():return OuroDepthModel(copy.deepcopy(base),mode='full',checkpointing=True)
            initializer=fresh().save_trainable(root/'initializer')
            plan_path=root/'plan.json'
            def args(name,arm='extension'):
                return SimpleNamespace(output=str(root/name),data_dir=str(data),model_path=str(root/'same-base'),
                    checkpoint=str(initializer),plan_path=str(plan_path),resume=None,device='cpu',arm=arm,
                    seed=20260916,mode='full',batch_size=3,micro_batch=2,eval_batch=2,max_length=16,
                    padding_width=8,max_updates=60,phase_updates=[6,12,12],warmup_updates=24,
                    lr=1e-6,weight_decay=.01,clip=1.,depths=[4,6,8,16],pad_id=0,train_limit=0,dev_limit=0)
            setup=args('setup');setup.plan_path=None
            plan,_,_=t.prepare_plan(fixtures.TinyTokenizer(),setup,1)
            plan_path.write_text(json.dumps(plan))
            evaluation_calls=[]
            def fake_evaluation(model,encoded,answers,a,depths,prefix):
                values=[]
                for item in encoded:
                    row=item['row'];score={'prediction_token':item['target'],'choice':row['answer'],
                        'correct':True,'choice_correct':True,'choice_tied':False,'choice_tie_aware_correct':1.,
                        'nll':.2,'choice_nll':.1,'answer_mass':.9}
                    values.append({k:row[k] for k in ('id','answer','family','difficulty')}|{'scores':{str(d):score.copy() for d in depths}})
                result={'count':len(values),'depths':depths,'evaluator_version':2,
                        'choice_tie_break':'ascending_token_id','metrics':_recompute(values,list(map(str,depths)))}
                t.write_json(str(prefix)+'.json',result)
                Path(str(prefix)+'.predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
                evaluation_calls.append((Path(a.output).name,Path(prefix).name))
                return result
            stack.enter_context(patch.object(t,'evaluate',side_effect=fake_evaluation))
            original_collate=t.collate_fixed
            def run(model,a,trace):
                def collate(*x,**kw):
                    trace.append((random.random(),float(np.random.random())))
                    return original_collate(*x,**kw)
                with patch.object(t,'collate_fixed',side_effect=collate):return t.train(model,fixtures.TinyTokenizer(),a)
            full=fresh();full_args=args('full');full_trace=[];fixtures.seed(full_args.seed)
            forward_calls=[]
            handle=full.register_forward_pre_hook(lambda module,a,kw:forward_calls.append(tuple(kw['depths'])),with_kwargs=True)
            continuous=run(full,full_args,full_trace);handle.remove();continuous_rng=fixtures.rng()
            continuous_saved=torch.load(Path(continuous['checkpoint'])/'training.pt',map_location='cpu',weights_only=False)
            self.assertEqual(len(forward_calls),60)
            self.assertEqual(forward_calls,[tuple(map(int,r['loss_weights'])) for r in plan['arms']['extension'] for _ in range(2)])
            partial_args=args('resumed');partial_args.max_updates=6;partial_trace=[];fixtures.seed(partial_args.seed)
            partial=run(fresh(),partial_args,partial_trace)
            self.assertEqual(partial['termination'],'max_updates');self.assertIsNone(partial['dev'])
            self.assertFalse((Path(partial_args.output)/'completed.json').exists())
            resume_args=copy.copy(partial_args);resume_args.max_updates=60;resume_args.resume=partial['checkpoint']
            checkpoint=Path(partial['checkpoint']);payload_path=checkpoint/'training.pt';original=payload_path.read_bytes()
            payload=torch.load(payload_path,map_location='cpu',weights_only=False)
            for fault in ('lr','state','loss_definition'):
                value=copy.deepcopy(payload)
                if fault=='lr':value['optimizer']['param_groups'][0]['lr']=1e-6
                elif fault=='state':value['state']['plan_cursor']['cursor']+=1
                else:value['identity']['loss_definition']['easy_weights']['T4']=.5
                try:
                    torch.save(value,payload_path)
                    with self.subTest(fault=fault),self.assertRaises(ValueError):run(fresh(),resume_args,[])
                finally:payload_path.write_bytes(original)
            foreign=copy.copy(resume_args);foreign.resume=str(root/'foreign/checkpoint-6')
            with self.assertRaises(ValueError):run(fresh(),foreign,[])
            changed=Path(partial_args.output)/'source/ouro_depth/train_extension.py';original_source=changed.read_bytes()
            try:
                changed.write_bytes(original_source+b'\n# modified source\n')
                with self.assertRaises(ValueError):run(fresh(),resume_args,[])
            finally:changed.write_bytes(original_source)
            resumed=fresh();resumed_trace=[];fixtures.seed(999);torch.randn(19);random.random();np.random.random(7)
            resumed_result=run(resumed,resume_args,resumed_trace)
            resumed_saved=torch.load(Path(resumed_result['checkpoint'])/'training.pt',map_location='cpu',weights_only=False)
            self.equal(full.state_dict(),resumed.state_dict());self.equal(continuous_rng,fixtures.rng())
            self.assertEqual(full_trace,partial_trace+resumed_trace)
            for key in ('state','optimizer','torch_rng','cuda_rng','python_rng','numpy_rng'):
                self.equal(continuous_saved[key],resumed_saved[key])
            def updates(output):return [json.loads(line) for line in (Path(output)/'metrics.jsonl').read_text().splitlines() if json.loads(line)['event']=='update']
            a,b=updates(full_args.output),updates(resume_args.output)
            keys=('update','depth','difficulty','loss','terminal_ce','shallow_ce','lr','grad_norm','compute_units','auxiliary_exit_used')
            self.assertEqual([{k:r[k] for k in keys} for r in a],[{k:r[k] for k in keys} for r in b])
            for record,event in zip(plan['arms']['extension'],a):
                self.assertEqual(event['missing_grad_count'],0);self.assertGreater(event['grad_norm'],0)
                self.assertEqual(event['auxiliary_exit_used'],len(record['loss_weights'])>1)
                if event['auxiliary_exit_used']:
                    self.assertAlmostEqual(event['loss'],.25*event['shallow_ce']+.75*event['terminal_ce'],places=6)
                else:self.assertEqual(event['loss'],event['terminal_ce'])
            self.assertEqual(evaluation_calls,[('full','dev-final'),('resumed','dev-final')])
            # The exposure-matched control endpoint must be committed and reused
            # on resume, then followed by the larger equal-cost endpoint.
            control_args=args('control','control');control_args.max_updates=30;fixtures.seed(control_args.seed)
            control=run(fresh(),control_args,[])
            self.assertEqual(control['termination'],'max_updates')
            self.assertTrue((Path(control_args.output)/'endpoint-30.json').exists())
            self.assertEqual(evaluation_calls[-1],('control','dev-30'))
            control_resume=copy.copy(control_args);control_resume.resume=control['checkpoint'];control_resume.max_updates=60
            control_done=run(fresh(),control_resume,[])
            self.assertEqual(control_done['state']['update'],48)
            self.assertEqual(control_done['state']['compute_units'],continuous['state']['compute_units'])
            self.assertEqual([x for x in evaluation_calls if x[0]=='control'],[('control','dev-30'),('control','dev-final')])
            self.assertEqual(control_done['termination'],'budget')
            with self.assertRaises(FileExistsError):run(fresh(),control_resume,[])
            RESULTS.update({'extension_updates':30,'resume_after':6,'control_endpoints':[30,48],
                'model_adam_rng_and_next_updates_exact':True,'nondivisible_microbatch_sizes':[2,1],
                'actual_auxiliary_updates':sum(r['auxiliary_exit_used'] for r in a),
                'all_shared_gradients_present_and_finite':True,'global_gradient_norm_nonzero':True,'single_wrapper_call_per_microbatch':True,
                'evaluation_receipts_reused_on_resume':True,'bad_lr_cursor_loss_source_foreign_checkpoint_rejected':True,
                'evaluation_stubbed':True,'old_multi_exit_equivalence_repeated':False})


if __name__=='__main__':unittest.main()
