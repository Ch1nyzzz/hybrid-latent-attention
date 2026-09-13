"""Only new heterogeneous confirmation schema and preparation boundaries.

All files live in temporary directories; no actual reference corpus is opened,
and generate_dataset is never executed. Forty valid random synthetic pointer
rows exercise the persisted auditor; simple metadata fixtures test split mixing.
"""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ouro_depth import prepare_huginn_depth_data as prepared
from ouro_depth.data import _pointer_instance, _instance_key, _key
from ouro_depth.prepare_diagnostics import _write_json, _write_rows


class HuginnDepthData(unittest.TestCase):
    def test_fixed_production_schema_and_unseen_assignment(self):
        self.assertEqual({s: sum(q.values()) for s,q in prepared.COUNTS.items()},
                         {'train':24000, 'dev':1280, 'test':3072})
        self.assertEqual(sum(sum(q.values()) for q in prepared.COUNTS.values()), 28352)
        self.assertEqual(set(prepared.COUNTS['test']), {'1','2','6','8','9','10','11','12'})
        self.assertEqual(prepared.COUNTS['test']['1'],256)
        self.assertEqual(prepared.COUNTS['test']['9'],512)
        generated={}
        for split, depths, repeats in [('train',prepared.TRAIN_DEPTHS,1), ('dev',prepared.TRAIN_DEPTHS,1),
                                      ('test',prepared.TRAIN_DEPTHS,1), ('ood',prepared.UNSEEN_DEPTHS,3)]:
            generated[split]=[{'id':f'{split}-{d}-{a}-{j}', 'family':'pointer_chasing',
                'split':split, 'difficulty':d, 'answer':a, 'metadata':{'instance_key':f'{split}-{d}-{a}-{j}'}}
                for d in depths for a in prepared.LETTERS for j in range(repeats)]
        original=copy.deepcopy(generated)
        rows,mixing=prepared._assemble_splits(generated,train_per_depth=8,dev_per_depth=8,
                                             seen_test_per_depth=8,unseen_test_per_depth=16)
        self.assertEqual(generated,original)
        self.assertEqual({s:len(r) for s,r in rows.items()},{'train':48,'dev':80,'test':96})
        self.assertEqual(mixing['IID_test_discarded_by_predeclared_hop'],{'3':8,'4':8})
        retained={r['id'] for r in rows['test']}
        self.assertTrue(all((r['id'] in retained)==(r['difficulty'] in prepared.GUARD_DEPTHS)
                            for r in generated['test']))
        for split in ('dev','test'):
            for r in rows[split]:
                if r['difficulty'] not in prepared.UNSEEN_DEPTHS:continue
                index=int(r['id'].split('-')[-1])
                self.assertEqual(index==0,split=='dev')
                source=next(x for x in generated['ood'] if x['id']==r['id'])
                self.assertEqual({**r,'split':'ood'},source)
        for fault in ('split','hop','answer','stream'):
            bad=copy.deepcopy(generated)
            if fault=='split':bad['test'][0]['split']='dev'
            elif fault=='hop':bad['ood'][0]['difficulty']=8
            elif fault=='answer':bad['ood'][0]['answer']='Z'
            else:bad['other']=[]
            with self.subTest(fault=fault),self.assertRaises(ValueError):
                prepared._assemble_splits(bad,train_per_depth=8,dev_per_depth=8,
                                         seen_test_per_depth=8,unseen_test_per_depth=16)

    def test_small_persisted_heterogeneous_audit_and_corruption(self):
        quotas={'train':{'1':8},'dev':{'9':8},'test':{'1':8,'9':16}}
        rng=random.Random(619);rows={s:[] for s in prepared.SPLITS}
        for split,q in quotas.items():
            for d,n in q.items():
                for j in range(n):
                    r=_pointer_instance(rng,int(d),prepared.LETTERS[j%8],25)
                    key=_instance_key('pointer_chasing',r['metadata']['facts'])
                    r['metadata']['instance_key']=key
                    r.update(family='pointer_chasing',difficulty=int(d),split=split,
                             id=_key({'instance_key':key,'query':r['metadata']['query']})[:24])
                    rows[split].append(r)
        with tempfile.TemporaryDirectory() as folder,patch.object(prepared,'COUNTS',quotas):
            dest=Path(folder)
            mixing={'mixing_seeds':{s:prepared.SEED+101+j for j,s in enumerate(prepared.SPLITS)},
                'IID_test_discarded_by_predeclared_hop':{'3':256,'4':256},'filter_uses_model_outputs':False}
            manifest=prepared._manifest({'synthetic-reference':1},{},mixing)
            for s,values in rows.items():_write_rows(dest/f'{s}.jsonl',values)
            _write_json(dest/'manifest.json',manifest)
            result=prepared.audit_persisted(dest,{'unrelated-reference'},{'synthetic-reference':1},{},
                                           {s:[r['id'] for r in values] for s,values in rows.items()})
            self.assertEqual(result['verified_rows'],40)
            self.assertEqual(result['splits']['test']['by_difficulty']['1']['count'],8)
            self.assertEqual(result['splits']['test']['by_difficulty']['9']['count'],16)
            self.assertEqual(result['internal_split_overlap'],0)
            self.assertEqual(set(result['split_sha256']),set(prepared.SPLITS))
            for fault in ('split','duplicate','metadata','count','reference'):
                bad=copy.deepcopy(rows);excluded=set()
                if fault=='split':bad['test'][0]['split']='dev'
                elif fault=='duplicate':bad['test'][1]=copy.deepcopy(bad['test'][0])
                elif fault=='metadata':bad['test'][0]['metadata']['context_size']=24
                elif fault=='count':bad['test'].pop()
                else:excluded.add(bad['test'][0]['metadata']['instance_key'])
                with self.subTest(fault=fault),self.assertRaises(ValueError):
                    prepared._audit_rows(bad,quotas,excluded)
            for field,value in [('candidate_status','adopted'),('seed',28622),
                                ('count_per_difficulty',{'train':{'1':8},'dev':{'9':8},'test':{'1':16,'9':8}})]:
                changed={**manifest,field:value};_write_json(dest/'manifest.json',changed)
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Manifest differs'):
                    prepared.audit_persisted(dest,set(),{'synthetic-reference':1},{})
            manifest['persisted_verification']={'split_sha256':{s:'wrong' for s in prepared.SPLITS}}
            _write_json(dest/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'generated digest'):
                prepared.audit_persisted(dest,set(),{'synthetic-reference':1},{})
            manifest['persisted_verification']={**result,'generated_to_persisted_id_order':'exact','verified_rows':39}
            _write_json(dest/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'verification metadata'):
                prepared.audit_persisted(dest,{'unrelated-reference'},{'synthetic-reference':1},{})

    def test_extension_references_are_metadata_only_and_unique(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for j,relative in enumerate(prepared.EXTENSION_REFERENCES):
                p=root/relative;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps({'metadata':{'instance_key':f'extension-{j}'}})+'\n')
            def prior(_):return {'earlier'},{'prior':1},{'subset':{'additional_instances':0}}
            with patch.object(prepared,'previous_exclusion_index',side_effect=prior):
                excluded,counts,subsets=prepared.exclusion_index(root)
                self.assertEqual(excluded,{'earlier','extension-0','extension-1','extension-2'})
                self.assertEqual(sum(counts.values()),4)
                sealed=root/prepared.EXTENSION_REFERENCES[-1]
                sealed.write_text(json.dumps({'metadata':{'instance_key':'earlier'}})+'\n')
                with self.assertRaisesRegex(ValueError,'overlaps'):prepared.exclusion_index(root)
                sealed.write_text(json.dumps({'metadata':{'instance_key':None}})+'\n')
                with self.assertRaises(ValueError):prepared.exclusion_index(root)

    def test_refuse_overwrite_and_generator_failure_never_publishes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'data/huginn-depth-pointer';output.mkdir(parents=True)
            with patch.object(prepared,'exclusion_index',side_effect=AssertionError('References must not be read')):
                with self.assertRaises(FileExistsError):prepared.prepare_huginn_depth_data(root)
            output.rmdir()
            with patch.object(prepared,'exclusion_index',return_value=(set(),{},{})), \
                 patch.object(prepared,'generate_dataset',side_effect=RuntimeError('synthetic generation failure')) as generate:
                with self.assertRaisesRegex(RuntimeError,'synthetic generation failure'):
                    prepared.prepare_huginn_depth_data(root)
                self.assertEqual({k:generate.call_args.kwargs[k] for k in
                    ('train_count','dev_count','test_count','ood_count','seed')},
                    {'train_count':48000,'dev_count':1536,'test_count':3072,'ood_count':5120,'seed':28621})
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob('.huginn-depth-stage-*')),[])


if __name__=='__main__':unittest.main()
