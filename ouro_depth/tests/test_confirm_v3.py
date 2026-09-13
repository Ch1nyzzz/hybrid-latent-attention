"""Synthetic v3 candidate/provenance and process-dispatch tests; no model/data access."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ouro_depth import confirm_v3 as c
from ouro_depth.v3_plan import build_plan
from ouro_depth.tests.test_v3_plan import synthetic_rows


class V3ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan(synthetic_rows(), seed=20260914, budget=2_000_000_000,
                              batch_size=16, padding_width=208)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ouro-v3-confirm-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        for name, options in (
            ('assert_gpu_unused', {'side_effect': lambda gpu: f'{gpu}, GPU-fixture-{gpu}, A100, 4'}),
            ('compare_prefixes', {'return_value': {'decision_scope': 'development',
                'decision': {'confirmation_eligible': True, 'full_method_supported': False}}}),
            ('validate_development', {'return_value': {'validated_synthetic_fixture': True}}),
            ('subprocess.Popen', {'side_effect': AssertionError('Unexpected launch')}),
        ):
            target = c
            if '.' in name:
                parent, name = name.split('.')
                target = getattr(c, parent)
            p = patch.object(target, name, **options)
            setattr(self, name, p.start())
            self.addCleanup(p.stop)

    def fixture(self, name='case'):
        root = self.base/name
        source = root/'ouro_depth'
        source.mkdir(parents=True)
        (source/'__init__.py').write_text('')
        (source/'train.py').write_text('# Inert source, never executed.\n')
        (root/'artifacts/v3-plan').mkdir(parents=True)
        c.write_json(root/'artifacts/v3-plan/plan.json', self.plan)
        c.write_json(root/'artifacts/model_source.json', {'repository': 'ByteDance/Ouro-1.4B',
                                                       'revision': c.MODEL_REVISION})
        data = root/'data/v3-pointer'
        data.mkdir(parents=True)
        for split in ('train', 'dev', 'test'):
            (data/f'{split}.jsonl').write_text('Synthetic fixture '+split+'\n')
        hashes = {s: c.digest(data/f'{s}.jsonl') for s in ('train','dev','test')}
        c.write_json(data/'manifest.json', {'split_counts': {'train': 24000, 'dev': 1280, 'test': 5120},
            'persisted_verification': {'split_sha256': hashes}})
        initializer = root/'diagnostics/diagnostic-onehop-s20260913/checkpoint-416'
        initializer.mkdir(parents=True)
        (initializer/'trainable.pt').write_bytes(b'initializer fixture')
        c.write_json(initializer.parent/'completed.json', {'termination': 'budget',
            'checkpoint': str(initializer), 'state': {'update':416, 'compute_units':500539392}})
        identity = {'format_version': 1, 'protocol': 'pointer_v3', 'seed':20260914,
            'budget':2_000_000_000, 'mode':'full', 'batch_size':16, 'micro_batch':8,
            'padding_width':208, 'plan_fingerprint': self.plan['fingerprint'], 'train_limit':0,
            'dev_limit':1280, 'device_type':'cuda', 'lr':1e-5, 'weight_decay':0.01,
            'clip':1.0, 'warmup_fraction':0.05, 'max_length':768,
            'model_path':str(root/'base_model'), 'initial_checkpoint':str(initializer),
            'initial_checkpoint_sha256':c.digest(initializer/'trainable.pt'),
            'train_file_sha256':hashes['train'], 'dev_file_sha256':hashes['dev']}
        for role, name in c.NAMES.items():
            run = root/'runs'/name
            state = c._expected_counters(self.plan, c.ARMS[role])
            state['valid_tokens'] = state['padded_tokens']-1
            checkpoint = run/f'checkpoint-{state["update"]}'
            checkpoint.mkdir(parents=True)
            (checkpoint/'trainable.pt').write_bytes((role+' weights fixture').encode())
            (checkpoint/'training.pt').write_bytes(b'optimizer fixture')
            ident = {**identity, 'arm':c.ARMS[role]}
            dev = {'count':1280, 'depths':[4,6,8], 'evaluator_version':2, 'role':role}
            for path, value in ((run/'identity.json',ident),(checkpoint/'identity.json',ident),
                (run/'plan.json',self.plan),(run/'frozen-plan.json',self.plan),
                (run/'dev-final.json',dev),(run/'latest.json',{'checkpoint':str(checkpoint),**state}),
                (run/'completed.json',{'checkpoint':str(checkpoint),'state':state,'dev':dev,
                    'termination':'budget','plan_fingerprint':self.plan['fingerprint'],
                    'planned_updates':state['update']})):
                c.write_json(path,value)
        return root

    def assert_no_launch(self):
        self.assert_gpu_unused.assert_not_called()
        self.Popen.assert_not_called()

    def test_incomplete_or_mismatched_final_candidates_rejected_before_hashing(self):
        for case in ('missing','counter','cursor','identity','warmup'):
            with self.subTest(case=case):
                root=self.fixture(case); run=root/'runs'/c.NAMES['fixed']
                path=run/'completed.json'
                if case=='missing': path.unlink()
                elif case in ('counter','cursor'):
                    value=c.read_json(path)
                    if case=='counter': value['state']['compute_units']-=1
                    else: value['state']['plan_cursor']['cursor']-=1
                    c.write_json(path,value)
                elif case=='identity':
                    path=run/'identity.json';value=c.read_json(path);value['lr']=2e-5;c.write_json(path,value)
                else:
                    path=root/'diagnostics/diagnostic-onehop-s20260913/completed.json'
                    value=c.read_json(path);value['state']['compute_units']=1;c.write_json(path,value)
                with patch.object(c,'_file_identity',side_effect=AssertionError('Premature artifact hashing')):
                    with self.assertRaises((ValueError,FileNotFoundError)): c.prepare(root)
                self.assertFalse((root/c.DESTINATION).exists())
        self.assert_no_launch();self.compare_prefixes.assert_not_called()

    def test_ineligible_dev_never_hashes_or_launches(self):
        root=self.fixture()
        self.compare_prefixes.return_value={'decision_scope':'development','decision':{'confirmation_eligible':False}}
        with patch.object(c,'_file_identity',side_effect=AssertionError('Premature artifact hashing')):
            with self.assertRaises(ValueError): c.prepare(root)
        self.assertFalse((root/c.DESTINATION).exists());self.assert_no_launch()

    def test_invalid_development_binding_stops_before_selection_or_test_hash(self):
        root=self.fixture()
        self.validate_development.side_effect=ValueError('Mismatched DEV artifact')
        with patch.object(c,'_file_identity',side_effect=AssertionError('Premature artifact hashing')):
            with self.assertRaisesRegex(ValueError,'Mismatched DEV'):c.prepare(root)
        self.compare_prefixes.assert_not_called();self.assert_no_launch()
        self.assertFalse((root/c.DESTINATION).exists())

    def test_prepare_binds_four_models_and_original_generated_data(self):
        root=self.fixture();destination,frozen=c.prepare(root)
        self.assertEqual(len(frozen['commands']),4)
        self.assertEqual(set(frozen['weights']),{'initializer','fixed','conditional','independent'})
        self.assertEqual(frozen['primary_hops'],[9,10,11,12])
        self.assertTrue((destination/'frozen.json').is_file())
        for task in frozen['commands']:
            command=task['command']
            self.assertEqual(command[command.index('--eval-file')+1],'test.jsonl')
            self.assertEqual(command[command.index('--depths')+1],'4,6,8')
            self.assertEqual(command[command.index('--checkpoint')+1],frozen['metadata']['candidates'][task['role']]['checkpoint'])
        self.assert_no_launch()

    def test_changed_test_before_preparation_rejected_against_generation_manifest(self):
        root=self.fixture();(root/'data/v3-pointer/test.jsonl').write_text('changed fixture\n')
        with self.assertRaisesRegex(ValueError,'generation manifest'):c.prepare(root)
        self.assertFalse((root/c.DESTINATION).exists());self.assert_no_launch()

    def test_changed_weight_or_data_after_freeze_rejected_even_with_preserved_stat(self):
        for role in ('conditional','test'):
            root=self.fixture(role);destination,frozen=c.prepare(root)
            identity=frozen['data_files']['test'] if role=='test' else frozen['weights'][role]
            path=Path(identity['path']);stat=path.stat();data=path.read_bytes()
            path.write_bytes(bytes([data[0]^1])+data[1:]);os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'bytes changed'):c.execute(root,destination,frozen)
            self.assertFalse((destination/'status.json').exists())
        self.assert_no_launch()

    def test_existing_last_output_preflights_before_any_hash_or_process(self):
        root=self.fixture();destination,frozen=c.prepare(root)
        Path(frozen['commands'][-1]['prefix']+'.predictions.jsonl').write_text('existing fixture')
        with patch.object(c,'_file_identity',side_effect=AssertionError('Premature artifact hashing')):
            with self.assertRaises(FileExistsError):c.execute(root,destination,frozen)
        self.assert_no_launch()

    def test_changed_commands_source_or_bound_paths_rejected_before_hardware(self):
        root=self.fixture();destination,original=c.prepare(root)
        for case in ('checkpoint','eval_file','prefix','source','weight_path','data_path'):
            with self.subTest(case=case):
                frozen=copy.deepcopy(original)
                if case in ('checkpoint','eval_file'):
                    command=frozen['commands'][0]['command']
                    flag='--checkpoint' if case=='checkpoint' else '--eval-file'
                    command[command.index(flag)+1]='wrong-fixture'
                elif case=='prefix': frozen['commands'][-1]['prefix']+='-wrong'
                elif case=='source': frozen['source']+='-wrong'
                elif case=='weight_path': frozen['weights']['conditional']['path']+='-wrong'
                else: frozen['data_files']['test']['path']+='-wrong'
                with patch.object(c,'_file_identity',side_effect=AssertionError('Premature artifact hashing')):
                    with self.assertRaises(ValueError):c.execute(root,destination,frozen)
        self.assert_no_launch()

    def test_dispatch_four_frozen_tasks_to_verified_gpu_uuids(self):
        root=self.fixture();destination,frozen=c.prepare(root);spawned=[]
        def launch(command,**kwargs):
            prefix=command[command.index('--output')+1]
            self.assertEqual(kwargs['cwd'],frozen['source'])
            self.assertIn(kwargs['env']['CUDA_VISIBLE_DEVICES'],('GPU-fixture-4','GPU-fixture-5'))
            c.write_json(prefix+'.json',{'count':5120,'depths':[4,6,8],'evaluator_version':2})
            spawned.append(command)
            return SimpleNamespace(pid=41000+len(spawned),poll=lambda:0)
        self.Popen.side_effect=launch
        self.compare_prefixes.return_value={'decision_scope':'heldout_test','decision':{'confirmation_eligible':None}}
        with patch.object(c.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            c.execute(root,destination,frozen)
        self.assertEqual(spawned,[task['command'] for task in frozen['commands']])
        status=c.read_json(destination/'status.json')
        self.assertEqual(status['phase'],'completed');self.assertEqual(len(status['tasks']),4)
        self.assertEqual(self.compare_prefixes.call_args.kwargs['split'],'test')
        self.assertTrue(all(call.args[0] in (4,5) for call in self.assert_gpu_unused.call_args_list))


if __name__=='__main__':unittest.main()
