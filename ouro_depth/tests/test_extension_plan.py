"""New candidate schedule/loss/budget risks only; pure synthetic stdlib rows."""
import copy
from collections import Counter
import json
import unittest
from ouro_depth.extension_plan import build_plan, PlanCursor, validate_plan, fingerprint


def rows():
    return [{'id':f'synthetic-{d}-{i}','family':'pointer_chasing','difficulty':d,
             'answer':'ABCDEFGH'[i%8],'prompt':f'placeholder-{i}'} for d in (1,2,3,4,6,8) for i in range(37)]


class ExtensionPlanTests(unittest.TestCase):
    def test_exact_shared_prefix_cost_exposure_lr_loss_and_boundaries(self):
        plan=build_plan(rows(),padding_width=208)
        control,extension=plan['arms']['control'],plan['arms']['extension']
        self.assertEqual((len(control),len(extension)),(384,240))
        self.assertEqual(plan['endpoints'],{'control':[240,384],'extension':[240]})
        self.assertEqual(plan['budget'],490733568)
        for records in (control,extension):
            self.assertEqual(sum(r['depth'] for r in records),1536)
            self.assertEqual(records[-1]['cumulative_compute'],plan['budget'])
        for i,(a,b) in enumerate(zip(control,extension)):
            for field in ('indices','ids','difficulty','lr'):self.assertEqual(a[field],b[field])
            self.assertEqual(b['depth'],4 if i<48 else 6 if i<144 else 8)
            self.assertEqual(b['lr'],1e-6*min((i+1)/24,1))
            expected={str(b['depth']):1.} if b['depth']==4 or b['difficulty']>2 else {'4':.25,str(b['depth']):.75}
            self.assertEqual(b['loss_weights'],expected)
            self.assertEqual(a['loss_weights'],{'4':1.})
        for records,multiplier in ((control,1024),(extension,640),(control[:240],640)):
            self.assertEqual(Counter({d:sum(16 for r in records if r['difficulty']==d) for d in (1,2,3,4,6,8)}),
                             Counter({d:multiplier for d in (1,2,3,4,6,8)}))
        for i in range(0,384,6):self.assertEqual(sorted(r['difficulty'] for r in control[i:i+6]),[1,2,3,4,6,8])
        for difficulty in (1,2,3,4,6,8):
            used=[i for r in control if r['difficulty']==difficulty for i in r['indices']]
            for start in range(0,len(used),37):self.assertEqual(len(used[start:start+37]),len(set(used[start:start+37])))

    def test_determinism_answer_independence_and_exact_cursor_restore(self):
        original=rows();plan=build_plan(original,padding_width=8,batch_size=2,num_layers=1,phase_updates=(6,12,12))
        self.assertEqual(plan,build_plan(original,padding_width=8,batch_size=2,num_layers=1,phase_updates=(6,12,12)))
        changed=copy.deepcopy(original)
        for row in changed:row['answer']='A'
        alternate=build_plan(changed,padding_width=8,batch_size=2,num_layers=1,phase_updates=(6,12,12))
        self.assertEqual(plan['arms'],alternate['arms']);self.assertNotEqual(plan['row_fingerprint'],alternate['row_fingerprint'])
        cursor=PlanCursor(plan,'extension')
        for _ in range(7):cursor.advance()
        restored=PlanCursor(plan,'extension');restored.load_state_dict(json.loads(json.dumps(cursor.state_dict())))
        while cursor.peek() is not None:
            self.assertEqual(cursor.peek(),restored.peek());cursor.advance();restored.advance()
        self.assertIsNone(restored.peek())
        before=restored.state_dict()
        for key,value in (('arm','control'),('cursor',31),('cursor',True),('plan_fingerprint','foreign')):
            bad={**before,key:value}
            with self.assertRaises(ValueError):restored.load_state_dict(bad)
            self.assertEqual(restored.state_dict(),before)

    def test_rehashed_loss_lr_endpoint_depth_and_sampler_tampering_is_rejected(self):
        plan=build_plan(rows(),padding_width=8,batch_size=2,num_layers=1,phase_updates=(6,12,12))
        mutations=[lambda p:p['arms']['extension'][6].__setitem__('depth',8),
            lambda p:p['arms']['extension'][7].__setitem__('loss_weights',{'6':.5}),
            lambda p:p['arms']['control'][0].__setitem__('lr',1e-6),
            lambda p:p['endpoints']['control'].__setitem__(0,24),
            lambda p:p['arms']['control'][0].__setitem__('compute_units',1),
            lambda p:p['shared_stream'][0]['indices'].__setitem__(0,0)]
        for mutate in mutations:
            changed=copy.deepcopy(plan);mutate(changed);changed['fingerprint']=fingerprint({k:v for k,v in changed.items() if k!='fingerprint'})
            with self.assertRaises(ValueError):validate_plan(changed)
        for kwargs in ({'phase_updates':(6,6,6)},{'phase_updates':(1,12,12)},{'lr':1e-5},{'warmup_updates':0}):
            with self.assertRaises(ValueError):build_plan(rows(),padding_width=8,**kwargs)


if __name__=='__main__':unittest.main()
