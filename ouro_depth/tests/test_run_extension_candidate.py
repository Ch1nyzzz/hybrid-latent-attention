"""One stdlib stub integration: adoption, raw gate, frozen launches, no retry.

No tokenizer/model/GPU or actual study data is accessed. Scoring files are
synthetic; the production evaluation binding is exercised once per validation.
"""
from collections import Counter
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ouro_depth.extension_plan import build_plan
from ouro_depth.v3_eval_binding import _recompute

SPEC=importlib.util.spec_from_file_location('extension_candidate_runner',Path(__file__).resolve().parents[2]/'artifacts/run_extension_candidate.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class RunnerStub(unittest.TestCase):
    def test_explicit_adoption_raw_screen_common_source_commands_and_no_retry(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.ExitStack() as stack:
            root=Path(temporary).resolve()
            for name in ('artifacts','diagnostics','runs','ouro_depth'):(root/name).mkdir()
            (root/'ouro_depth/synthetic_source.txt').write_text('frozen synthetic dependency')
            # Verify the actual adoption reader against synthetic final receipts.
            adoption={'status':'adopted','v4_final_bound':True,'v4_development_eligible':False,'candidate_protocol':m.PROTOCOL}
            m.write_json(root/'artifacts/extension-candidate-adoption.json',adoption)
            meta={'protocol':'pointer_v4','budget':3067084800,'candidates':{}}
            comparison={'protocol':'pointer_v4','decision_scope':'development','split':'dev',
                'decision':{'development_eligible':False},'count_validation':{'total':1280},'inputs':{
                'initializer':{'prefix':str(root/'diagnostics/v4-initializer-dev/initializer-dev')}}}
            for arm,u in (('fixed4',2400),('fixed8',1200)):
                run=root/'runs'/f'v4-{arm}-s20260915';cp=run/f'checkpoint-{u}';cp.mkdir(parents=True)
                identity={'protocol':'pointer_v4','arm':arm};state={'update':u,'compute_units':3067084800};dev={'synthetic':True}
                for path,value in ((run/'identity.json',identity),(cp/'identity.json',identity),
                    (run/'completed.json',{'termination':'budget','checkpoint':str(cp),'state':state,'dev':dev}),
                    (run/'latest.json',{'checkpoint':str(cp),**state}),(run/'dev-final.json',dev)):m.write_json(path,value)
                meta['candidates'][arm]={'checkpoint':str(cp),'identity':identity,'update':u,'compute_units':3067084800}
                comparison['inputs'][arm]={'prefix':str(run/'dev-final')}
            m.write_json(root/'artifacts/v4-final-metadata.json',meta)
            m.write_json(root/'artifacts/v4-final-dev-comparison.json',comparison)
            self.assertEqual(m._adoption(root)['adoption'],adoption)
            comparison['decision']['development_eligible']=True
            m.write_json(root/'artifacts/v4-final-dev-comparison.json',comparison)
            with patch.object(m,'_initial',side_effect=AssertionError('Must fail before old initializer/data/GPU read')):
                with self.assertRaises(ValueError):m.execute(root,'initialize')
            comparison['decision']['development_eligible']=False
            m.write_json(root/'artifacts/v4-final-dev-comparison.json',comparison)
            data=root/'data';data.mkdir();hops=(1,2,3,4,6,8,9,10,11,12)
            rows=[{'id':f'synthetic-{d}-{i}','family':'pointer_chasing','difficulty':d,'answer':'ABCDEFGH'[i%8]}
                  for d in hops for i in range(128)]
            (data/'dev.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            training=[r for r in rows if r['difficulty'] in (1,2,3,4,6,8)]
            plan=build_plan(training,padding_width=208)
            initial=root/'runs/v3-fixed4-s20260914/checkpoint-1566'
            stack.enter_context(patch.object(m,'_initial',return_value=(initial,{'synthetic':True})))
            stack.enter_context(patch.object(m,'_prepared',return_value=(data,plan,{'synthetic':True})))
            source_receipt={'files':{'synthetic_source.txt':'synthetic'},'fingerprint':'synthetic'}
            stack.enter_context(patch.object(m,'_source_identity',return_value=source_receipt))
            source=stack.enter_context(patch.object(m,'_source',wraps=m._source))
            available=stack.enter_context(patch.object(m,'_available',return_value='mocked available GPU'))
            stack.enter_context(patch.object(m,'_complete_arm',side_effect=lambda r,a,p,f:str(r/'runs'/m.NAMES[a]/f"checkpoint-{len(p['arms'][a])}")))
            calls=[];polls=Counter();next_pid=100
            def evaluation(prefix,bad_d2=False):
                predictions=[]
                for r in rows:
                    correct=not (bad_d2 and r['difficulty']==2)
                    choice=r['answer'] if correct else ('B' if r['answer']=='A' else 'A')
                    score={'prediction_token':m.ANSWER_IDS[choice],'choice':choice,'correct':correct,'choice_correct':correct,'choice_tied':False,
                        'choice_tie_aware_correct':float(correct),'nll':.1,'choice_nll':.1,'answer_mass':.8}
                    predictions.append({**r,'scores':{str(d):score.copy() for d in m.DEPTHS}})
                Path(str(prefix)+'.predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
                m.write_json(str(prefix)+'.json',{'count':1280,'evaluator_version':2,'choice_tie_break':'ascending_token_id',
                    'depths':m.DEPTHS,'metrics':_recompute(predictions,list(map(str,m.DEPTHS)))})
            class Child:
                def __init__(self,command,**kw):
                    nonlocal next_pid
                    next_pid+=1;self.pid=next_pid;self.command=command;self.returncode=None;calls.append((command,kw))
                    self.arm=command[command.index('--arm')+1] if '--arm' in command else 'initializer'
                    if self.arm=='initializer':evaluation(Path(command[command.index('--output')+1]))
                def poll(self):
                    polls[self.arm]+=1
                    # Failed control must not prevent the healthy arm being waited to completion.
                    if self.arm=='extension' and polls[self.arm]==1:return None
                    self.returncode=1 if self.arm=='control' else 0;return self.returncode
            stack.enter_context(patch.object(m.subprocess,'Popen',side_effect=Child))
            stack.enter_context(patch.object(m.time,'sleep'))
            initialized=m.execute(root,'initialize')
            self.assertEqual(initialized['phase'],'completed');self.assertEqual(source.call_count,1)
            self.assertEqual(calls[0][1]['env']['CUDA_VISIBLE_DEVICES'],m.GPUS[5])
            self.assertIn('dev.jsonl',calls[0][0]);self.assertIn('4,6,8,16',calls[0][0])
            before=available.call_count
            with self.assertRaises(FileExistsError):m.execute(root,'initialize')
            self.assertEqual(available.call_count,before)
            prefix=root/'diagnostics/extension-candidate-initializer-dev/initializer-dev'
            predictions_path=Path(str(prefix)+'.predictions.jsonl')
            predictions=predictions_path.read_text().splitlines();bad=json.loads(predictions[0])
            bad['scores']['4']['prediction_token']=0;predictions[0]=json.dumps(bad)
            predictions_path.write_text('\n'.join(predictions)+'\n')
            with self.assertRaisesRegex(ValueError,'raw token'):m._screen(prefix,data)
            evaluation(prefix,bad_d2=True)
            self.assertFalse(m._screen(prefix,data)['passed'])
            with self.assertRaises(ValueError):m.execute(root,'train')
            self.assertFalse((root/'artifacts/extension-candidate-train-status.json').exists())
            evaluation(prefix)
            with self.assertRaises(RuntimeError):m.execute(root,'train')
            state=m.read_json(root/'artifacts/extension-candidate-train-status.json')
            self.assertEqual([r['state'] for r in state['runs']],['failed','completed'])
            self.assertEqual(state['live_pids'],[]);self.assertGreaterEqual(polls['extension'],2)
            self.assertEqual(source.call_count,1)
            for command,kwargs in calls[1:]:
                arm=command[command.index('--arm')+1];output=Path(command[command.index('--output')+1])
                for key,value in {'--seed':'20260916','--batch-size':'16','--micro-batch':'8','--lr':'1e-6',
                    '--warmup-updates':'24','--padding-width':'208','--max-updates':'384' if arm=='control' else '240'}.items():
                    self.assertEqual(command[command.index(key)+1],value)
                self.assertIn('ouro_depth.train_extension',command)
                self.assertEqual(m.read_json(output/'frozen-plan.json'),plan)
                self.assertEqual((kwargs['cwd']/'ouro_depth/synthetic_source.txt').read_text(),'frozen synthetic dependency')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],m.GPUS[4 if arm=='control' else 5])
                self.assertEqual(kwargs['env']['PYTHONPATH'],str(kwargs['cwd']))
            with self.assertRaises(FileExistsError):m.execute(root,'train')
            self.assertEqual(len(calls),3)
            self.assertFalse(any('test.jsonl' in str(command) for command,_ in calls))


if __name__=='__main__':unittest.main()
