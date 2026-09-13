"""Actual CPU tiny-Ouro resume coverage for the pointer_v2 trainer integration."""

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ouro_depth import train as trainer
from ouro_depth.model import OuroDepthModel
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM


class _TinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" "):
            return [2 + "ABCDEFGH".index(text[1:])]
        prompt, separator, answer = text.partition(" ")
        number = int(prompt.removeprefix("example").removesuffix(":"))
        ids = [1, 11 + number, 20 + number]
        return ids + ([2 + "ABCDEFGH".index(answer)] if separator else [])


class _InterruptedAtTwo(Exception):
    pass


class PointerV2TrainerResumeTests(unittest.TestCase):
    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_cpu_pointer_v2_resume_restores_samples_phases_optimizer_rng_and_identity(self):
        """Four real updates span task phases; save/restore occurs after update two."""
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            old_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            stack.callback(torch.set_num_threads, old_threads)
            # The test must not initialize GPU state even on the training host.
            stack.enter_context(patch.object(torch.cuda, "get_rng_state_all", return_value=[]))
            stack.enter_context(patch.object(torch.cuda, "set_rng_state_all"))
            stack.enter_context(patch.object(torch.cuda, "manual_seed_all"))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=0))
            stack.enter_context(patch.object(trainer, "evaluate", return_value={"metrics": {}, "count": 6}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

            data = root / "data"
            data.mkdir()
            rows = [
                {"id": f"v2-row-{index}", "prompt": f"example{index}:",
                 "answer": "ABCDEFGH"[index], "family": "pointer_chasing", "difficulty": difficulty}
                for index, difficulty in enumerate((1, 2, 3, 4, 6, 8))
            ]
            for split in ("train", "dev"):
                (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            torch.manual_seed(802)
            config = OuroConfig(
                vocab_size=43, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                max_position_embeddings=32, attention_dropout=0.1,
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
                use_cache=False, tie_word_embeddings=False,
            )
            config._attn_implementation = "sdpa"
            initial = OuroForCausalLM(config).float()

            def fresh_model():
                return OuroDepthModel(copy.deepcopy(initial), mode="full", checkpointing=True)

            def arguments(name):
                # train() receives already initialized weights; main() owns loading
                # this named initializer. Its path still belongs in resume identity.
                return SimpleNamespace(
                    output=str(root / name), data_dir=str(data), model_path=str(root / "same-base"),
                    checkpoint=str(root / "declared-initializer" / "checkpoint-7"),
                    arm="v2curriculum", task_schedule="pointer_v2", seed=73, mode="full", lora_rank=4,
                    device="cpu", batch_size=2, micro_batch=1, eval_batch=2,
                    lr=1e-3, weight_decay=0.01, clip=1.0, warmup_fraction=0.05,
                    budget=1400, max_updates=4, backprop_loops=None,
                    no_checkpointing=False, train_limit=0, max_length=32,
                    dev_limit=0, pad_id=0, eval_every=0, save_every=2,
                    depths=[4, 8], resume=None,
                )

            original_sample = trainer.PointerCurriculumSampler.batch_indices

            def trace_samples(trace):
                def wrapped(sampler, batch_size, progress):
                    indices = original_sample(sampler, batch_size, progress)
                    trace.append((progress, tuple(indices)))
                    return indices
                return patch.object(trainer.PointerCurriculumSampler, "batch_indices", new=wrapped)

            continuous_args = arguments("continuous")
            continuous_model = fresh_model()
            continuous_samples = []
            trainer.seed_all(continuous_args.seed)
            with trace_samples(continuous_samples):
                trainer.train(continuous_model, _TinyTokenizer(), continuous_args)

            interrupted_args = arguments("interrupted")
            interrupted_model = fresh_model()
            interrupted_samples = []
            original_checkpoint = trainer.checkpoint

            def save_then_interrupt(model, optimizer, output, state):
                saved = original_checkpoint(model, optimizer, output, state)
                if state["update"] == 2:
                    raise _InterruptedAtTwo()
                return saved

            trainer.seed_all(interrupted_args.seed)
            with trace_samples(interrupted_samples), patch.object(trainer, "checkpoint", side_effect=save_then_interrupt):
                with self.assertRaises(_InterruptedAtTwo):
                    trainer.train(interrupted_model, _TinyTokenizer(), interrupted_args)

            output = Path(interrupted_args.output)
            source = output / "checkpoint-2"
            saved_at_two = torch.load(source / "training.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved_at_two["state"]["update"], 2)
            self.assertFalse((output / "completed.json").exists())
            sampler_at_two = saved_at_two["state"]["task_sampler_state"]
            self.assertIsNotNone(sampler_at_two)
            for key in ("category_rng_state", "pool_rng_state", "cursors", "epochs"):
                self.assertIn(key, sampler_at_two)
            identity_path = output / "identity.json"
            identity_before = identity_path.read_text()
            identity = json.loads(identity_before)
            self.assertEqual(identity["task_schedule"], "pointer_v2")
            self.assertEqual(identity["initial_checkpoint"], str(Path(interrupted_args.checkpoint).resolve()))

            wrong_schedule = copy.copy(interrupted_args)
            wrong_schedule.resume = str(source)
            wrong_schedule.task_schedule = "flat"
            with self.assertRaisesRegex(ValueError, "requires the pointer_v2 task schedule"):
                trainer.train(fresh_model(), _TinyTokenizer(), wrong_schedule)
            self.assertEqual(identity_path.read_text(), identity_before)

            wrong_initializer = copy.copy(interrupted_args)
            wrong_initializer.resume = str(source)
            wrong_initializer.checkpoint = str(root / "different-initializer" / "checkpoint-7")
            with self.assertRaisesRegex(ValueError, "configuration/data mismatch.*initial_checkpoint"):
                trainer.train(fresh_model(), _TinyTokenizer(), wrong_initializer)
            self.assertEqual(identity_path.read_text(), identity_before)

            resumed_model = fresh_model()
            resumed_samples = []
            interrupted_args.resume = str(source)
            torch.manual_seed(9999)
            torch.randn(97)  # Missing restoration must be observable through dropout.
            with trace_samples(resumed_samples):
                trainer.train(resumed_model, _TinyTokenizer(), interrupted_args)

            self.assert_tree_equal(continuous_model.state_dict(), resumed_model.state_dict())
            self.assertEqual(continuous_samples, interrupted_samples + resumed_samples)
            self.assertEqual(len(continuous_samples), 8)  # Four updates, two microbatches each.
            continuous_final = json.loads((Path(continuous_args.output) / "completed.json").read_text())
            resumed_final = json.loads((output / "completed.json").read_text())
            self.assert_tree_equal(continuous_final["state"], resumed_final["state"])
            state = resumed_final["state"]
            self.assertEqual(state["update"], 4)
            self.assertEqual(state["examples"], 8)
            self.assertGreaterEqual(len(state["stage_histogram"]), 2)
            self.assertEqual(sum(state["stage_histogram"].values()), 8)
            self.assertEqual(sum(state["task_histogram"].values()), 8)
            self.assertEqual(sum(state["depth_histogram"].values()), 8)
            self.assertGreaterEqual(len(state["task_histogram"]), 2)

            continuous_checkpoint = torch.load(Path(continuous_final["checkpoint"]) / "training.pt", map_location="cpu", weights_only=False)
            resumed_checkpoint = torch.load(Path(resumed_final["checkpoint"]) / "training.pt", map_location="cpu", weights_only=False)
            self.assert_tree_equal(continuous_checkpoint, resumed_checkpoint)
            final_sampler = resumed_checkpoint["state"]["task_sampler_state"]
            self.assertNotEqual(sampler_at_two["category_rng_state"], final_sampler["category_rng_state"])
            self.assertGreater(sum(final_sampler["epochs"].values()), 0)
            for optimizer_state in resumed_checkpoint["optimizer"]["state"].values():
                self.assertEqual(optimizer_state["step"].item(), 4)
                self.assertTrue(torch.isfinite(optimizer_state["exp_avg"]).all())
                self.assertTrue(torch.isfinite(optimizer_state["exp_avg_sq"]).all())
            self.assertTrue(any(
                not torch.equal(saved_at_two["optimizer"]["state"][key]["exp_avg"], value["exp_avg"])
                for key, value in resumed_checkpoint["optimizer"]["state"].items()
            ))

            def update_trace(path):
                records = [json.loads(line) for line in path.read_text().splitlines()]
                updates = [record for record in records if record["event"] == "update"]
                fields = ("update", "depth", "task_stage", "task_counts", "loss", "grad_norm", "lr", "compute_units", "examples")
                return [{field: record[field] for field in fields} for record in updates]

            continuous_trace = update_trace(Path(continuous_args.output) / "metrics.jsonl")
            resumed_trace = update_trace(output / "metrics.jsonl")
            self.assert_tree_equal(continuous_trace, resumed_trace)
            self.assertEqual([record["update"] for record in resumed_trace], [1, 2, 3, 4])
            self.assertGreaterEqual(len({record["task_stage"] for record in resumed_trace}), 2)
            self.assertTrue(all(math.isfinite(record["loss"]) and record["grad_norm"] > 0 for record in resumed_trace))


if __name__ == "__main__":
    unittest.main()
