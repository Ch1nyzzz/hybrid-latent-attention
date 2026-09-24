"""Focused correctness checks for the experimental adjoint benchmark."""
import unittest
import torch
from hla.latent.benchmark_khop import kgrad


class KHopTest(unittest.TestCase):
    def test_finite_hops_restore_gradient_double(self):
        theta=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
        a=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        c0=theta.sin();c1=a*c0+theta;c2=a*c1+c0.square()
        actual=(c0+c1.square()+c2.sin())
        truth=torch.autograd.grad(actual,(theta,a))
        leaves=torch.stack([c0,c1,c2]).detach().requires_grad_()
        computed=torch.stack([theta.sin(),a*leaves[0]+theta,a*leaves[1]+leaves[0].square()])
        loss=leaves[0]+leaves[1].square()+leaves[2].sin()
        got=kgrad(loss,[computed],[leaves],[theta,a],3)
        for g,t in zip(got,truth):torch.testing.assert_close(g,t,atol=1e-12,rtol=1e-12)

    def test_s6_parallel_gradient_recovery(self):
        from contextlib import nullcontext
        from hla.vendor.configuration_ouro import OuroConfig
        from hla.vendor.modeling_ouro import OuroForCausalLM
        from hla.latent.register import LatentStudent
        from hla.latent import diag_khop_gradient as diag
        torch.manual_seed(7)
        cfg=OuroConfig(vocab_size=41,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,
            total_ut_steps=4,use_cache=False,pad_token_id=0,bos_token_id=1,eos_token_id=2)
        cfg._attn_implementation='eager'
        model=OuroForCausalLM(cfg).double().eval().requires_grad_(False)
        student=LatentStudent(2,16,2,8,4,8,8,8).double()
        ids=torch.randint(3,41,(1,17))
        original=diag.khop_gradients
        def use_new(loss,computed,leaves,params,hops):
            output=[]
            for k in range(hops+1):
                gs=kgrad(loss,computed,leaves,params,k,retain_graph=True)
                output.append([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gs)])
            return output
        # First test production benchmark VJP directly against the original estimator on a rebuilt graph.
        for checkpoint in (False,True):
            report,_,_=diag.run_sequence(model,student,ids,5,windows=[4],hops=[0,1,2,3,11],lam_attn=.1,
                normalizer=12.,autocast=nullcontext,parallel_checkpoint=checkpoint)
            self.assertLess(report['forward_check']['row_max_abs_error'],2e-6)
            self.assertLess(report['versus_full']['hop11']['all']['relative_l2'],2e-6)
        diag.khop_gradients=use_new
        try:
            again,_,_=diag.run_sequence(model,student,ids,5,windows=[],hops=[11],lam_attn=.1,
                normalizer=12.,autocast=nullcontext,parallel_checkpoint=True)
            self.assertLess(again['versus_full']['hop11']['all']['relative_l2'],2e-6)
        finally:diag.khop_gradients=original


if __name__=='__main__':unittest.main()
