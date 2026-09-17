"""Two real Gloo ranks: S6 variable-token updates equal serial accumulation."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run_rank(rank, rendezvous):
    # Spawned workers do not load pytest conftest automatically.
    from ouro_depth.tests import conftest
    from ouro_depth.tests.test_s6_engine import fixture, targets_fn
    from ouro_depth.latent.batched_recipe import prepare_batch, backward_batch
    from ouro_depth.latent.training_common import make_optimizer, synchronize_gradients
    model,student,teacher,ids=fixture()
    reference=deepcopy(student)
    optimizer,serial_optimizer=make_optimizer(student),make_optimizer(reference)
    records=[(ids[:1],3),(ids[1:,:4],2)]
    dist.init_process_group('gloo',init_method=Path(rendezvous).as_uri(),rank=rank,world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        for stage in (2,3):
            optimizer.zero_grad(set_to_none=True);serial_optimizer.zero_grad(set_to_none=True)
            normalizer=sum(x.shape[1]-1-(p if stage==3 else 0) for x,p in records)
            def backward(st,rows):
                batch=prepare_batch(rows,targets_fn(teacher),stage)
                backward_batch(model,st,batch,stage=stage,normalizer=normalizer,chunk_size=3,
                               horizon_tokens=3,window=3,prompt_chunk_size=3)
            backward(student,[records[rank]])
            # Rank 1 has no historical writer contribution: one prefill chunk
            # or just one decode input. Global active mask must still reduce it.
            if rank==1:
                for sl in student.layers:
                    for linear in [*sl.cand_s,sl.cand1]:
                        assert linear.weight.grad is None or not linear.weight.grad.count_nonzero()
            synchronize_gradients(student)
            for row in records:backward(reference,[row])
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True)
            for (name,p),(_,q) in zip(student.named_parameters(),reference.named_parameters()):
                if q.grad is None:assert p.grad is None,name
                else:torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=2e-7,msg=name)
            optimizer.step();serial_optimizer.step()
            for name,value in student.state_dict().items():
                torch.testing.assert_close(value,reference.state_dict()[name],rtol=3e-5,atol=2e-7,msg=name)
    finally:dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo unavailable')
def test_two_rank_updates_match_serial(tmp_path):
    mp.spawn(_run_rank,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)
