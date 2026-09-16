"""Two real Gloo ranks: variable-token gradient sums equal a serial update."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ouro_depth.latent.train_recipe import (
    backward_example, initialize_decode_readers, make_optimizer,
    synchronize_gradients,
)
from ouro_depth.tests.test_rolling_engine import fixture
from ouro_depth.tests.test_train_recipe import teacher_wrapper


def _run_rank(rank, rendezvous, batched=False):
    torch.set_num_threads(1)
    model, student, batch = fixture()
    serial_model, serial_student, _ = fixture()
    teacher, serial_teacher = teacher_wrapper(model), teacher_wrapper(serial_model)
    optimizer, serial_optimizer = make_optimizer(student), make_optimizer(serial_student)
    original = deepcopy(student.state_dict())
    # Rank 1 has fewer valid tokens. During stage 2 its continuation has only
    # the boundary prediction, so decode readers/finalizer are locally unused.
    # Rank 0 has several actual decode inputs and two TBPTT backward windows.
    records = [(batch[:1], 3), (batch[1:2, :5], 4)]
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        for stage in (1, 2):
            if stage == 2:
                initialize_decode_readers(student, optimizer)
                initialize_decode_readers(serial_student, serial_optimizer)
            optimizer.zero_grad(set_to_none=True)
            serial_optimizer.zero_grad(set_to_none=True)
            ids, prompt = records[rank]
            if stage == 1:
                prompt = ids.shape[1] - 1
            local_counts = torch.tensor([
                ids.shape[1] - 1 if stage == 1 else prompt - 1,
                0 if stage == 1 else ids.shape[1] - prompt,
                ids.shape[1] - 1,
            ], dtype=torch.float64)
            dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
            normalizers = local_counts.tolist()
            assert normalizers == ([10, 0, 10] if stage == 1 else [5, 5, 10])
            if batched:
                from ouro_depth.latent.batched_recipe import prepare_batch, backward_batch
                microbatch = prepare_batch([(ids, prompt)], teacher, stage)
                backward_batch(model, student, microbatch, stage=stage, mode='main',
                               window=2, first_window=2, normalizers=normalizers)
            else:
                logits, targets = teacher(ids[:, :-1])
                backward_example(model, student, ids, prompt, logits, targets,
                                 stage=stage, mode='main', window=2, first_window=2,
                                 normalizers=normalizers)
            if stage == 1 or rank == 1:
                assert student.layers[0].finalize_mlp[2].weight.grad is None
                assert student.layers[0].q_absorb_d.grad is None
            else:
                assert student.layers[0].finalize_mlp[2].weight.grad.norm().item() > 0
                assert student.layers[0].q_absorb_d.grad.norm().item() > 0
            synchronize_gradients(student)
            if stage == 2:
                # The rank with no local finalizer path must receive the other
                # rank's globally active gradient, rather than skip the reduce.
                assert student.layers[0].finalize_mlp[2].weight.grad.norm().item() > 0
                assert student.layers[0].q_absorb_d.grad.norm().item() > 0

            # Independent serial accumulation over the SAME two examples and
            # global denominators. Deliberately bypass synchronize_gradients:
            # this reference runs within an initialized process group.
            for serial_ids, serial_prompt in records:
                if stage == 1:
                    serial_prompt = serial_ids.shape[1] - 1
                logits, targets = serial_teacher(serial_ids[:, :-1])
                backward_example(serial_model, serial_student, serial_ids, serial_prompt,
                                 logits, targets, stage=stage, mode='main',
                                 window=2, first_window=2, normalizers=normalizers)
            torch.nn.utils.clip_grad_norm_(serial_student.parameters(), 1.0,
                                           error_if_nonfinite=True)
            for (name, actual), (expected_name, expected) in zip(
                    student.named_parameters(), serial_student.named_parameters()):
                assert name == expected_name
                if expected.grad is None:
                    assert actual.grad is None, name
                else:
                    torch.testing.assert_close(actual.grad, expected.grad,
                                               rtol=2e-5, atol=2e-7, msg=name)
            optimizer.step()
            serial_optimizer.step()
            for name, actual in student.state_dict().items():
                torch.testing.assert_close(actual, serial_student.state_dict()[name],
                                           rtol=2e-5, atol=2e-7, msg=name)
            if stage == 1:
                for name, parameter in student.named_parameters():
                    if name.endswith('_d') or 'finalize_mlp' in name:
                        assert parameter.grad is None, name
                        assert parameter not in optimizer.state, name
                        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0, msg=name)
                assert not torch.equal(student.layers[0].cand.weight, original['layers.0.cand.weight'])
            else:
                assert student.layers[0].finalize_mlp[2].weight in optimizer.state
                assert student.layers[0].q_absorb_d in optimizer.state
    finally:
        dist.destroy_process_group()


class DistributedRecipeTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_batched_rank_updates_match_serial_with_locally_unused_parameters(self):
        with tempfile.TemporaryDirectory(prefix='batched-recipe-gloo-') as directory:
            mp.spawn(_run_rank, args=(str(Path(directory) / 'rendezvous'), True),
                     nprocs=2, join=True)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_two_rank_updates_match_serial_with_variable_lengths_and_unused_parameters(self):
        with tempfile.TemporaryDirectory(prefix='recipe-gloo-') as directory:
            mp.spawn(_run_rank, args=(str(Path(directory) / 'rendezvous'),),
                     nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
