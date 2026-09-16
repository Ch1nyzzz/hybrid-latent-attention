"""Numerical and masking contracts for the Stage1/2 throughput paths."""
from contextlib import nullcontext
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from ouro_depth.latent import train_stage1_recipe as stage1
from ouro_depth.latent.batched_recipe import (prepare_batch, backward_batch,
                                             masked_fkl, memory_bounded_fkl)
from ouro_depth.tests.test_rolling_engine import fixture
from ouro_depth.tests.test_train_recipe import teacher_wrapper


class PrefillOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_stage1_packing_preserves_groups_and_bounds_padding(self):
        rows = [dict(input_ids=[3]*n, record_id=i) for i,n in enumerate([8,9,8,10,12,9,12,8])]
        original = list(stage1.length_batches(rows, 2))
        packs = list(stage1.packed_length_batches(rows, 2, 4, 1.6))
        actual = {i: [r['record_id'] for r in g] for pack in packs for i,g in pack}
        self.assertEqual(actual, {i: [r['record_id'] for r in g] for i,g in enumerate(original)})
        self.assertLess(len(packs), len(original))
        for pack in packs:
            count=sum(len(g) for _,g in pack)
            longest=max(len(g[0]['input_ids']) for _,g in pack)
            work=sum(len(g)*len(g[0]['input_ids'])**2 for _,g in pack)
            self.assertLessEqual(count,4)
            self.assertLessEqual(count*longest**2,1.6*work)

    def test_stage1_same_masks_group_normalizers_loss_and_gradients(self):
        model, initial, ids = fixture()
        teacher = teacher_wrapper(model).teacher
        records = [dict(input_ids=ids[i%2, :n].tolist()) for i,n in enumerate([7,5,7,6,5])]
        results=[]
        for execution in ('legacy','packed'):
            student=deepcopy(initial)
            # FP32 verifies algebra separately from BF16 GEMM shape rounding.
            with patch('ouro_depth.latent.train_stage1.torch.autocast', return_value=nullcontext()):
                metrics,norm=stage1.train_update(student,teacher,records,global_batch=5,micro_batch=2,
                    seed=31,step=7,rank=0,p_lockstep=.5,exit_target='reuse',device=torch.device('cpu'),
                    execution=execution,packed_batch=5,padding_ratio=2)
            results.append((torch.tensor(metrics),{n: p.grad for n,p in student.named_parameters()}))
        torch.testing.assert_close(results[0][0],results[1][0],atol=3e-7,rtol=3e-6)
        for name,grad in results[0][1].items():
            other=results[1][1][name]
            if grad is None:self.assertIsNone(other,name)
            else:torch.testing.assert_close(grad,other,atol=3e-7,rtol=5e-5,msg=name)

    def test_teacher_batch_preserves_valid_logits_outputs_and_denominators(self):
        model,_,ids=fixture()
        teacher=teacher_wrapper(model)
        examples=[(ids[:1],6),(ids[1:2,:4],3),(ids[1:2,:6],5)]
        for stage in (1,2):
            a=prepare_batch(examples,teacher,stage,1)
            b=prepare_batch(examples,teacher,stage,3)
            torch.testing.assert_close(a.logits,b.logits,atol=2e-7,rtol=3e-5)
            for key in a.targets:
                torch.testing.assert_close(a.targets[key],b.targets[key],atol=2e-7,rtol=3e-5)
                torch.testing.assert_close(a.denominators[key],b.denominators[key],atol=1e-9,rtol=3e-5)

    def test_memory_bounded_kl_value_gradient_and_saved_tensor_budget(self):
        torch.manual_seed(7)
        original=torch.randn(3,67,41)
        teacher=torch.randn_like(original).requires_grad_()
        valid=torch.arange(67)[None] < torch.tensor([67,41,2])[:,None]
        results=[];saved=[]
        for fn in (masked_fkl,memory_bounded_fkl):
            student=original.clone().requires_grad_()
            tensors=[]
            with torch.autograd.graph.saved_tensors_hooks(lambda x:(tensors.append(x.numel()),x)[1],lambda x:x):
                loss=fn(student,teacher,valid)
            (loss*.37).backward()
            results.append((loss.detach(),student.grad));saved.append(sum(tensors))
        torch.testing.assert_close(results[0][0],results[1][0],atol=0,rtol=0)
        torch.testing.assert_close(results[0][1],results[1][1],atol=5e-8,rtol=3e-5)
        self.assertIsNone(teacher.grad)
        self.assertLess(saved[1],saved[0])
        self.assertTrue((results[1][1][~valid]==0).all())

    def test_optimized_prefill_loss_all_gradients_and_update(self):
        model,initial,ids=fixture()
        teacher=teacher_wrapper(model)
        examples=[(ids[:1],6),(ids[1:2,:4],3),(ids[1:2,:6],5)]
        result=[]
        for fast in (False,True):
            student=deepcopy(initial)
            opt=torch.optim.AdamW(student.parameters(),lr=1e-4)
            batch=prepare_batch(examples,teacher,1,3 if fast else 1)
            report=backward_batch(model,student,batch,stage=1,mode='main',window=2,first_window=1,
                normalizers=(14,0,14),checkpointing=True,
                prefill_backend='sdpa' if fast else 'math',low_memory_kl=fast)
            grads={n:None if p.grad is None else p.grad.clone() for n,p in student.named_parameters()}
            opt.step()
            result.append((report,grads,deepcopy(student.state_dict())))
        self.assertAlmostEqual(result[0][0]['objective'],result[1][0]['objective'],places=6)
        for name,grad in result[0][1].items():
            if grad is None:self.assertIsNone(result[1][1][name],name)
            else:torch.testing.assert_close(grad,result[1][1][name],atol=3e-7,rtol=1e-4,msg=name)
        for name,value in result[0][2].items():
            torch.testing.assert_close(value,result[1][2][name],atol=3e-6,rtol=1e-4,msg=name)


if __name__=='__main__':unittest.main()
