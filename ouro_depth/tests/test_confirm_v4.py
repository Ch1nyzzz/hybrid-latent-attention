"""V4 confirmation access gates with generated fixtures and fake children only.

No pretrained weights, actual research corpus, GPU query, or model call.
"""
from collections import Counter
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ouro_depth import confirm_v4 as c
from ouro_depth.launch_v4 import _initial_command, REVISION
from ouro_depth.v3_eval_binding import _recompute

HOPS=(1,2,3,4,6,8,9,10,11,12)


def rows(split,n):
    hops=HOPS[:6] if split=='train' else HOPS
    return [{'id':f'synthetic-{split}-{d}-{i}','split':split,'family':'pointer_chasing',
             'difficulty':d,'answer':'ABCDEFGH'[i%8],'prompt':f'synthetic prompt {d}/{i}:'}
            for d in hops for i in range(n)]


def predictions(data,role):
    counts=Counter(r['difficulty'] for r in data)
    result=[]
    for row in data:
        i=int(row['id'].rsplit('-',1)[1]);n=counts[row['difficulty']]
        scores={}
        for depth in c.DEPTHS[role]:
            fraction=.25
            if role=='fixed4' and depth==6: fraction=.375
            if role=='fixed8' and depth==16: fraction=.875
            correct=i<int(n*fraction) if row['difficulty']>=9 else depth!=16
            choice=row['answer'] if correct else 'ABCDEFGH'[('ABCDEFGH'.index(row['answer'])+1)%8]
            scores[str(depth)]={'prediction_token':101+'ABCDEFGH'.index(choice),'choice':choice,
                'correct':correct,'choice_correct':correct,'choice_tied':False,
                'choice_tie_aware_correct':float(correct),'nll':.2 if correct else 2.,
                'choice_nll':.1 if correct else 1.5,'answer_mass':.75}
        result.append({k:row[k] for k in ('id','answer','family','difficulty')}|{'scores':scores})
    return result


def write_predictions(prefix,values,depths):
    summary={'count':len(values),'depths':list(depths),'evaluator_version':2,
        'choice_tie_break':'ascending_token_id','metrics':_recompute(values,list(map(str,depths)))}
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    c.write_json(str(prefix)+'.json',summary)
    Path(str(prefix)+'.predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    return summary


class V4ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='synthetic-v4-confirm-')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        for name in ('artifacts/v4-plan','data/v4-pointer','diagnostics/v3-final-depth-dev',
                     'diagnostics/v4-initializer-dev','artifacts/v4-training/source/ouro_depth'):
            (self.root/name).mkdir(parents=True)
        package=Path(c.__file__).resolve().parent
        files=tuple(dict.fromkeys(c.TRAIN_FILES+c.CONTROL_FILES))
        for destination in (self.root/'ouro_depth',self.root/'artifacts/v4-training/source/ouro_depth',
                            self.root/'diagnostics/v4-initializer-dev/source/ouro_depth'):
            for name in files:
                target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(package/name,target)
        self.data={split:rows(split,n) for split,n in (('train',4000),('dev',128),('test',512))}
        directory=self.root/'data/v4-pointer'
        for split,values in self.data.items():
            (directory/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
        self.manifest={'dataset_type':'pointer_fixed_depth_v4','seed':19931,'node_count':25,
            'sealed_splits':['test'],'split_counts':{'train':24000,'dev':1280,'test':5120},
            'persisted_verification':{'internal_split_overlap':0,'reference_overlap':0,
                'split_sha256':{s:c.digest(directory/f'{s}.jsonl') for s in self.data}}}
        c.write_json(directory/'manifest.json',self.manifest)
        self.plan=c.build_plan(self.data['train'],padding_width=208)
        c.write_json(self.root/'artifacts/v4-plan/plan.json',self.plan)
        source={'repository':'ByteDance/Ouro-1.4B','revision':REVISION}
        c.write_json(self.root/'artifacts/model_source.json',source)
        self.initializer=self.root/'diagnostics/diagnostic-onehop-s20260913/checkpoint-416'
        self.initializer.mkdir(parents=True)
        weight=self.initializer/'trainable.pt';weight.write_bytes(b'INERT INITIALIZER FIXTURE, not a weight archive')
        stat=weight.stat()
        previous={'path':str(weight),'sha256':c.INITIAL_SHA,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        c.write_json(self.root/'diagnostics/v3-final-depth-dev/frozen.json',{'weights':{'initializer':previous}})
        c.write_json(self.initializer.parent/'completed.json',{'termination':'budget','checkpoint':str(self.initializer),
            'state':{'update':416,'compute_units':500539392}})
        self.model={'model_source':source,'initializer_weight_identity':previous,
            'prior_verified_digest_reused_with_unchanged_size_mtime':True}
        self.predictions={role:predictions(self.data['dev'],role) for role in c.ROLES}
        initial_prefix=c._prefixes(self.root)['initializer']
        write_predictions(initial_prefix,self.predictions['initializer'],c.DEPTHS['initializer'])
        initial_dir=initial_prefix.parent
        c.write_json(initial_dir/'launch.json',{'command':_initial_command(self.root,self.initializer,directory,initial_prefix),
            'model':self.model,'dataset_manifest':self.manifest,'gpu_uuid':c.GPU_UUIDS[4],
            'cwd':str(initial_dir/'source')})
        c.write_json(initial_dir/'completed.json',{'phase':'completed','exit_code':0})
        common=c._source_identity(self.root/'artifacts/v4-training/source/ouro_depth')
        self.identities={};self.states={};items=[]
        for arm,name in c.NAMES.items():
            run=self.root/'runs'/name
            shutil.copytree(self.root/'artifacts/v4-training/source',run/'source')
            state=c._expected_counters(self.plan,arm);state['valid_tokens']=state['padded_tokens']-1
            checkpoint=run/f'checkpoint-{state["update"]}';checkpoint.mkdir()
            for name in ('trainable.pt','training.pt'): (checkpoint/name).write_bytes((arm+' inert '+name).encode())
            identity={'format_version':1,'protocol':'pointer_v4','arm':arm,'seed':20260915,'mode':'full',
                'batch_size':16,'micro_batch':8,'lr':1e-5,'weight_decay':.01,'clip':1.,'fixed4_updates':2400,
                'max_length':768,'eval_batch':8,'eval_every':400,'save_every':400,'depths':list(c.DEPTHS[arm]),
                'plan_fingerprint':self.plan['fingerprint'],'padding_width':208,'num_layers':24,
                'trainable_parameters':1233324032,'device_type':'cuda','model_path':str(self.root/'base_model'),
                'initial_checkpoint':str(self.initializer/'trainable.pt'),'initial_checkpoint_sha256':c.INITIAL_SHA,
                'source':common,'train_file_sha256':self.manifest['persisted_verification']['split_sha256']['train'],
                'dev_file_sha256':self.manifest['persisted_verification']['split_sha256']['dev'],
                'optimizer':{'name':'AdamW','betas':[.9,.95],'eps':1e-8,'foreach':False,'fused':False},
                'encoded_train_sha256':'synthetic','pad_id':0,'torch':'synthetic-pinned'}
            summary=write_predictions(run/'dev-final',self.predictions[arm],c.DEPTHS[arm])
            gpu=4 if arm=='fixed4' else 5;pid=52000+gpu
            launch={'command':c._training_command(self.root,arm,self.initializer),'cwd':str(run/'source'),
                'output':str(run),'initializer':str(self.initializer),'model':self.model,
                'dataset_manifest':self.manifest,'plan_file':str(run/'frozen-plan.json'),
                'common_source':str(self.root/'artifacts/v4-training/source'),'gpu':gpu,'gpu_uuid':c.GPU_UUIDS[gpu],
                'pid':pid,'state':'running'}
            for path,value in ((run/'identity.json',identity),(checkpoint/'identity.json',identity),
                (run/'plan.json',self.plan),(run/'frozen-plan.json',self.plan),(run/'launch.json',launch),
                (run/'latest.json',{'checkpoint':str(checkpoint),**state}),
                (run/'completed.json',{'termination':'budget','budget':self.plan['budget'],
                    'planned_updates':state['update'],'plan_fingerprint':self.plan['fingerprint'],
                    'state':state,'checkpoint':str(checkpoint),'dev':summary})):
                c.write_json(path,value)
            items.append({'arm':arm,'name':run.name,'pid':pid,'gpu':gpu,'gpu_uuid':c.GPU_UUIDS[gpu],
                          'state':'completed','exit_code':0,'checkpoint':str(checkpoint)})
            self.identities[arm]=identity;self.states[arm]=state
        c.write_json(self.root/'artifacts/v4-launch.json',{'phase':'completed','plan_fingerprint':self.plan['fingerprint'],
            'runs':items,'test_scored':False})
        self.available=patch.object(c,'_available_gpu',side_effect=lambda gpu:f'{gpu}, {c.GPU_UUIDS[gpu]}, SYNTHETIC, 0').start()
        self.addCleanup(patch.stopall)
        patch.object(c,'_wait_released_gpu',side_effect=lambda gpu:(f'{gpu}, {c.GPU_UUIDS[gpu]}, SYNTHETIC, 0',1)).start()
        self.popen=patch.object(c.subprocess,'Popen',side_effect=AssertionError('Unexpected real process')).start()
        patch.object(c.time,'sleep').start()

    @contextlib.contextmanager
    def sealed_forbidden(self):
        original=Path.open
        def guarded(path,*args,**kwargs):
            if path==self.root/'data/v4-pointer/test.jsonl': raise AssertionError('Sealed fixture accessed before gate')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded): yield

    def mutate_json(self,path,mutate):
        original=path.read_bytes();value=json.loads(original);mutate(value);c.write_json(path,value)
        return lambda:path.write_bytes(original)

    def test_candidate_identity_rejects_final_plan_source_launch_and_checkpoint_changes_without_test_access(self):
        with self.sealed_forbidden():
            metadata=c.candidate_metadata(self.root)
        self.assertEqual(metadata['budget'],3067084800)
        run=self.root/'runs'/c.NAMES['fixed8']
        cases=[(run/'completed.json',lambda v:v['state'].__setitem__('compute_units',1)),
            (run/'completed.json',lambda v:v['state']['plan_cursor'].__setitem__('cursor',1199)),
            (run/'completed.json',lambda v:v.__setitem__('checkpoint',str(run/'checkpoint-800'))),
            (run/'latest.json',lambda v:v.__setitem__('update',1199)),
            (run/'identity.json',lambda v:v.__setitem__('initial_checkpoint_sha256','changed')),
            (run/'identity.json',lambda v:v.__setitem__('encoded_train_sha256','another-corpus')),
            (run/'plan.json',lambda v:v['arms']['fixed8'][0].__setitem__('depth',4)),
            (run/'launch.json',lambda v:v['command'].__setitem__(-1,'800')),
            (run/'checkpoint-1200/identity.json',lambda v:v.__setitem__('arm','fixed4'))]
        for path,mutate in cases:
            with self.subTest(path=path.name):
                undo=self.mutate_json(path,mutate)
                try:
                    with self.sealed_forbidden(),self.assertRaises(ValueError): c.candidate_metadata(self.root)
                finally: undo()
        path=run/'source/ouro_depth/model.py';original=path.read_bytes()
        try:
            path.write_bytes(original+b'\n# changed source\n')
            with self.sealed_forbidden(),self.assertRaises(ValueError): c.candidate_metadata(self.root)
        finally:path.write_bytes(original)
        self.available.assert_not_called();self.popen.assert_not_called()

    def test_failed_dev_gate_or_false_binding_precedes_any_test_read_or_weight_hash(self):
        metadata=c.candidate_metadata(self.root)
        with patch.object(c,'candidate_metadata',return_value=metadata),self.sealed_forbidden(),\
             patch.object(c,'_file_identity',side_effect=AssertionError('Premature artifact hash')):
            with patch.object(c,'validate_development',side_effect=ValueError('bad DEV binding')):
                with self.assertRaisesRegex(ValueError,'bad DEV binding'): c.prepare(self.root)
            with patch.object(c,'compare_prefixes',return_value={'decision_scope':'development','decision':{'development_eligible':False}}):
                with self.assertRaisesRegex(ValueError,'DEV gate failed'): c.prepare(self.root)
        self.assertFalse((self.root/c.DESTINATION).exists())
        self.available.assert_not_called();self.popen.assert_not_called()

    def test_saved_dev_summary_and_truth_membership_are_bound_before_selection(self):
        metadata=c.candidate_metadata(self.root)
        prefix=c._prefixes(self.root)['fixed8'];path=Path(str(prefix)+'.json')
        undo=self.mutate_json(path,lambda v:v['metrics']['all']['by_depth']['16'].__setitem__('accuracy',.999))
        try:
            with self.sealed_forbidden(),self.assertRaisesRegex(ValueError,'numeric mismatch'):
                c.validate_development(self.root,metadata)
        finally:undo()
        prediction_path=Path(str(prefix)+'.predictions.jsonl');original=prediction_path.read_text()
        values=[json.loads(l) for l in original.splitlines()];values[0]['id']='foreign-synthetic-ID'
        prediction_path.write_text(''.join(json.dumps(r)+'\n' for r in values))
        with self.sealed_forbidden(),self.assertRaisesRegex(ValueError,'ID sets differ'):
            c.validate_development(self.root,metadata)

    def fake_launch(self,spawned,live_second=False):
        def launch(command,**kwargs):
            output=Path(command[command.index('--output')+1]);split=command[command.index('--eval-file')+1].split('.')[0]
            role=output.name.rsplit('-',1)[0]
            status=c.read_json(self.root/c.DESTINATION/'status.json')
            if split=='test':
                self.assertEqual(len([t for t in status['tasks'] if t['split']=='dev' and t['state']=='completed']),3)
                self.assertTrue(all(t['reload_check']['accepted'] for t in status['tasks'] if t['split']=='dev'))
            self.assertEqual(kwargs['cwd'],str(self.root/c.DESTINATION/'source'))
            self.assertIn(kwargs['env']['CUDA_VISIBLE_DEVICES'],c.GPU_UUIDS.values())
            values=self.predictions[role] if split=='dev' else predictions(self.data['test'],role)
            write_predictions(output,values,c.DEPTHS[role])
            spawned.append((split,role))
            pid=61000+len(spawned)
            return SimpleNamespace(pid=pid,poll=lambda:None if live_second and pid==61002 else 0)
        return launch

    def test_complete_fake_dispatch_binds_dev_before_test_and_uses_real_holm_report(self):
        destination,frozen=c.prepare(self.root)
        self.assertEqual(len(frozen['commands']),6)
        self.assertEqual([t['split'] for t in frozen['commands']],['dev']*3+['test']*3)
        spawned=[];self.popen.side_effect=self.fake_launch(spawned)
        with contextlib.redirect_stdout(io.StringIO()): status=c.execute(self.root)
        self.assertEqual(status['phase'],'completed');self.assertTrue(status['test_scoring_started'])
        self.assertEqual(spawned,[(s,r) for s in ('dev','test') for r in c.ROLES])
        self.assertTrue(all(t['evaluation_binding']['count']==(1280 if t['split']=='dev' else 5120) for t in status['tasks']))
        result=c.read_json(destination/'comparison-test.json')
        self.assertEqual(result['decision_scope'],'heldout_test')
        self.assertTrue(result['decision']['confirmation_supported'])
        self.assertEqual(len(result['primary_comparisons']),5)
        self.assertTrue(result['decision']['all_five_holm_p_below_0_05'])
        self.assertTrue(result['decision']['own_exit_task_floors_passed'])
        self.assertTrue(result['decision']['d1_retention_passed'])
        before=len(spawned)
        with self.assertRaises(FileExistsError): c.execute(self.root)
        self.assertEqual(len(spawned),before)

    def test_numeric_reload_drift_blocks_all_test_children_and_preserves_live_pid(self):
        destination,_=c.prepare(self.root);spawned=[]
        self.popen.side_effect=self.fake_launch(spawned,live_second=True)
        original=c._checked_evaluation
        def drift(*args):
            binding,values=original(*args)
            if args[0]['role']=='initializer': next(iter(values.values()))['scores']['4']['nll']+=1e-13
            return binding,values
        with patch.object(c,'_checked_evaluation',side_effect=drift),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'Reloaded DEV changed'):c.execute(self.root)
        status=c.read_json(destination/'status.json')
        self.assertEqual(status['phase'],'failed');self.assertFalse(status['test_scoring_started'])
        self.assertEqual(status['live_pids'],[61002])
        self.assertTrue(all(split=='dev' for split,role in spawned));self.assertEqual(len(spawned),2)
        check=status['tasks'][0]['reload_check']
        self.assertFalse(check['accepted']);self.assertFalse(check['numeric_scores_exactly_equal'])
        self.assertEqual(check['numeric_differences']['4']['nll']['count'],1)
        self.assertEqual(check['numeric_differences']['4']['nll']['outside_tolerance'],0)
        original_rows={r['id']:r for r in self.predictions['initializer']}
        altered=copy.deepcopy(original_rows);next(iter(altered.values()))['scores']['4']['prediction_token']+=1
        self.assertFalse(c._reload_check(original_rows,altered,c.DEPTHS['initializer'])['accepted'])
        altered=copy.deepcopy(original_rows);next(iter(altered.values()))['scores']['4']['answer_mass']=float('nan')
        with self.assertRaises(ValueError):c._reload_check(original_rows,altered,c.DEPTHS['initializer'])

    def test_frozen_paths_existing_output_and_same_stat_weight_change_fail_before_hardware(self):
        destination,frozen=c.prepare(self.root)
        frozen_path=destination/'frozen.json';original=frozen_path.read_bytes()
        for fault in ('command','source','data','weight'):
            value=copy.deepcopy(frozen)
            if fault=='command':value['commands'][3]['command'][-1]='512'
            elif fault=='source':value['source']+='-foreign'
            elif fault=='data':value['data_files']['test']['path']=str(self.root/'data/v3-pointer/test.jsonl')
            else:value['weights']['fixed8']['path']=str(self.root/'runs'/c.NAMES['fixed8']/'checkpoint-800/trainable.pt')
            try:
                c.write_json(frozen_path,value)
                with self.assertRaises(ValueError):c.execute(self.root)
            finally:frozen_path.write_bytes(original)
        output=Path(frozen['commands'][-1]['prefix']+'.predictions.jsonl');output.write_text('existing')
        with patch.object(c,'candidate_metadata',side_effect=AssertionError('Premature metadata read')):
            with self.assertRaises(FileExistsError):c.execute(self.root)
        output.unlink()
        identity=frozen['weights']['fixed8'];path=Path(identity['path']);stat=path.stat();contents=path.read_bytes()
        path.write_bytes(bytes([contents[0]^1])+contents[1:]);os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'weight bytes changed'):c.execute(self.root)
        self.available.assert_not_called();self.popen.assert_not_called()
        self.assertFalse((destination/'status.json').exists())


if __name__=='__main__': unittest.main()
